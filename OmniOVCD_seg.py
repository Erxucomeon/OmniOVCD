import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image
from scipy import ndimage
from skimage import measure

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@MODELS.register_module()
class OmniOVCDSeg(BaseSegmentor):
    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 change_detection_method='instance_method',
                 enable_vis=False,
                 vis_dir='./vis',
                 merge_classes_above_threshold=None,
                 **kwargs):
        super().__init__()
        
        self.device = device
        # Initialize SAM3 model
        model = build_sam3_image_model(
            bpe_path=f"./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='sam3_checkpoint/sam3.pt', 
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        
        
        self.change_detection_method = change_detection_method
       
        self.instance_iou_threshold = kwargs.get('instance_iou_threshold', 0.3)  
        self.t12_min_instance_area = kwargs.get('t12_min_instance_area', 20)  
        self.cd_min_instance_area = kwargs.get('cd_min_instance_area', 0)  
        self.use_instance_level_cd = kwargs.get('use_instance_level_cd', False) 
        
        
        self.enable_vis = enable_vis
        self.vis_dir = vis_dir
        
        self.merge_classes_above_threshold = merge_classes_above_threshold

    def _extract_fpn_features(self, image):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            
            if "backbone_out" not in inference_state:
                raise ValueError("backbone_out not found in inference_state")
            
            backbone_out = inference_state["backbone_out"]
            
            if "backbone_fpn" not in backbone_out:
                raise ValueError("backbone_fpn not found in backbone_out")
            
            fpn_features = backbone_out["backbone_fpn"]
            original_size = (image.size[1], image.size[0])  # (H, W)
            
            return fpn_features, original_size

    def compute_fpn_similarity_map(self, fpn_features1, fpn_features2, target_size):
        def process_fpn_features(fpn_features, target_size):
            if not fpn_features or len(fpn_features) == 0:
                raise ValueError("FPN features list is empty")
            
            upsampled_features = []
            
            for feat in fpn_features:
                # feat shape: [B, C, H, W]
                B, C, H, W = feat.shape
                if C > 320:
                    feat_part1 = feat[:, :320, :, :]  # [B, 320, H, W]
                    feat_part2 = feat[:, 320:, :, :]  # [B, C-320, H, W]
                    
                    feat_part1_upsampled = F.interpolate(
                        feat_part1, size=target_size, mode='bilinear', align_corners=False
                    )
                    feat_part2_upsampled = F.interpolate(
                        feat_part2, size=target_size, mode='bilinear', align_corners=False
                    )
                    
                    feat_upsampled = torch.cat([feat_part1_upsampled, feat_part2_upsampled], dim=1)
                else:
                    feat_upsampled = F.interpolate(
                        feat, size=target_size, mode='bilinear', align_corners=False
                    )
                
                upsampled_features.append(feat_upsampled)
            
            concatenated_features = torch.cat(upsampled_features, dim=1)  
            return concatenated_features
        
        feat1 = process_fpn_features(fpn_features1, target_size)  
        feat2 = process_fpn_features(fpn_features2, target_size)  
        
        B, C, H, W = feat1.shape
        feat1_hwc = feat1[0].permute(1, 2, 0) 
        feat2_hwc = feat2[0].permute(1, 2, 0) 
        
        dot_product = torch.sum(feat1_hwc * feat2_hwc, dim=2) 
        
        # norm1 = ||feat1||, norm2 = ||feat2||
        norm1 = torch.norm(feat1_hwc, dim=2) 
        norm2 = torch.norm(feat2_hwc, dim=2)  
        
        cosine_sim = dot_product / (norm1 * norm2 + 1e-8)  
        similarity_map = cosine_sim * 0.5 + 0.5  
        
        return similarity_map

    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                if self.use_transformer_decoder:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = inference_state['object_score'][inst_id]
                            # instance_mask = inference_state['masks'][inst_id].squeeze()
                            
                            # Handle potential dimension mismatch if SAM3 output differs slightly
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape), 
                                    size=(h, w), 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * instance_score)
                    
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape != (h, w):
                            semantic_logits = F.interpolate(
                                semantic_logits, 
                                size=(h, w), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                    
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
                
                if self.use_presence_score:
                    presence_score = inference_state["presence_score"]
                    if not self.use_transformer_decoder and not self.use_sem_seg:
                        seg_logits[query_idx] = presence_score
                    else:
                        seg_logits[query_idx] = seg_logits[query_idx] * presence_score
                
        return seg_logits

    def slide_inference(self, image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        
        # Initialize accumulators
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)
        
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                
                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))
                
                # Inference on crop
                crop_seg_logit = self._inference_single_view(crop_img)
                
                # Accumulate results
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        
        preds = preds / count_mat
        return preds

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        for i, meta in enumerate(batch_img_metas):
            image1_path = meta.get('img1_path')
            image1 = Image.open(image1_path).convert('RGB')
            ori_shape = meta['ori_shape']

            if 'img_path' not in meta and image1_path is not None:
                data_samples[i].set_metainfo(dict(img_path=image1_path))

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image1.size[0] or self.slide_crop < image1.size[1]):
                seg1_logits = self.slide_inference(image1, self.slide_stride, self.slide_crop)
            else:
                seg1_logits = self._inference_single_view(image1)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg1_logits.shape[-2:] != ori_shape:
                seg1_logits = F.interpolate(
                    seg1_logits.unsqueeze(0), 
                    size=ori_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Post-processing
            if self.num_cls != self.num_queries:
                seg1_logits = seg1_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg1_logits = (seg1_logits * cls_index).max(1)[0]
                seg1_pred = seg1_logits.argmax(0, keepdim=True)

            seg1_pred = torch.argmax(seg1_logits, dim=0)
            
            # Apply probability threshold
            max_vals = seg1_logits.max(0)[0]
            seg1_pred[max_vals < self.prob_thd] = self.bg_idx
            
            if self.merge_classes_above_threshold is not None:
                threshold = self.merge_classes_above_threshold
                seg1_pred[seg1_pred >= threshold] = 1
                seg1_pred[seg1_pred < threshold] = self.bg_idx


            # img2
            image2_path = meta.get('img2_path')
            image2 = Image.open(image2_path).convert('RGB')
            ori_shape = meta['ori_shape']

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image2.size[0] or self.slide_crop < image2.size[1]):
                seg2_logits = self.slide_inference(image2, self.slide_stride, self.slide_crop)
            else:
                seg2_logits = self._inference_single_view(image2)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg2_logits.shape[-2:] != ori_shape:
                seg2_logits = F.interpolate(
                    seg2_logits.unsqueeze(0), 
                    size=ori_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Post-processing
            if self.num_cls != self.num_queries:
                seg2_logits = seg2_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg2_logits = (seg2_logits * cls_index).max(1)[0]
                seg2_pred = seg2_logits.argmax(0, keepdim=True)

            seg2_pred = torch.argmax(seg2_logits, dim=0)
            
            # Apply probability threshold
            max_vals = seg2_logits.max(0)[0]
            seg2_pred[max_vals < self.prob_thd] = self.bg_idx
            
            if self.merge_classes_above_threshold is not None:
                threshold = self.merge_classes_above_threshold
                seg2_pred[seg2_pred >= threshold] = 1
                seg2_pred[seg2_pred < threshold] = self.bg_idx

            method = self.change_detection_method
            
            if method == 'instance_method':
                seg1_binary = (seg1_pred == 1).long() 
                seg2_binary = (seg2_pred == 1).long() 
                
                change_pred = self._instance_level_change_detection(seg1_binary, seg2_binary)
            
            if self.use_instance_level_cd and method != 'instance_method':
                seg1_binary = (seg1_pred == 1).long() 
                seg2_binary = (seg2_pred == 1).long() 
                
                change_pred = self._instance_level_change_detection(seg1_binary, seg2_binary, change_pred)

            if self.cd_min_instance_area > 0:
                change_pred = self._filter_cd_instances_by_area(change_pred)

            if self.enable_vis:
                self._save_visualization_results(image1_path, seg1_pred, seg2_pred, change_pred, ori_shape)
            
            pred_sem_seg_data = change_pred.unsqueeze(0)  

            data_samples[i].set_data({
                'seg1_logits': PixelData(**{'data': seg1_logits}), 
                'seg2_logits': PixelData(**{'data': seg2_logits}), 
                'pred_sem_seg': PixelData(**{'data': pred_sem_seg_data})
            })
            
        
        return data_samples

    def _extract_instances(self, mask):
        mask_np = mask.cpu().numpy().astype(np.uint8) 
        
        labeled_mask = measure.label(mask_np, connectivity=2, background=0)  
        
        instances = []
        num_instances = labeled_mask.max()
        
        for instance_id in range(1, num_instances + 1):
            instance_mask = (labeled_mask == instance_id).astype(np.uint8)
            instances.append(instance_mask)
        
        return instances
    
    def _instance_level_change_detection(self, seg1_pred, seg2_pred, change_pred=None):
        t1_instances = self._extract_instances(seg1_pred)
        t2_instances = self._extract_instances(seg2_pred)
        
        seg1_np = seg1_pred.cpu().numpy().astype(np.uint8)  # [H, W]
        seg2_np = seg2_pred.cpu().numpy().astype(np.uint8)  # [H, W]
        
        if change_pred is not None:
            change_np = change_pred.cpu().numpy().astype(np.uint8)
        else:
            change_np = np.zeros_like(seg1_np, dtype=np.uint8)
        
        for t1_instance in t1_instances:
            t1_area = t1_instance.sum()
            if t1_area == 0:
                continue
            
            if t1_area < self.t12_min_instance_area:
                continue
            
            has_overlap = False
            for t2_instance in t2_instances:
                intersection = t1_instance * t2_instance
                intersection_area = intersection.sum()
                
                if intersection_area > 0:
                    overlap_ratio = intersection_area / t1_area
                    
                    if overlap_ratio >= self.instance_iou_threshold:
                        has_overlap = True
                        break
            
            if not has_overlap:
                change_np = np.maximum(change_np, t1_instance)
        
        for t2_instance in t2_instances:
            t2_area = t2_instance.sum()
            if t2_area == 0:
                continue
            if t2_area < self.t12_min_instance_area:
                continue
            has_overlap = False
            for t1_instance in t1_instances:
                intersection = t2_instance * t1_instance
                intersection_area = intersection.sum()
                if intersection_area > 0:
                    overlap_ratio = intersection_area / t2_area
                    
                    if overlap_ratio >= self.instance_iou_threshold:
                        has_overlap = True
                        break
            if not has_overlap:
                change_np = np.maximum(change_np, t2_instance)
        
        change_pred = torch.from_numpy(change_np).long().to(seg1_pred.device)
        
        return change_pred

    def _filter_cd_instances_by_area(self, change_pred):
        if self.cd_min_instance_area <= 0:
            return change_pred
        
        change_np = change_pred.cpu().numpy().astype(np.uint8) 
        
        original_mask = change_np.copy()
        
        labeled_mask = measure.label(change_np, connectivity=2, background=0) 
        
        filtered_mask = np.zeros_like(change_np, dtype=np.uint8)
        
        num_instances = labeled_mask.max()
        for instance_id in range(1, num_instances + 1):
            instance_mask = (labeled_mask == instance_id).astype(np.uint8)
            instance_area = instance_mask.sum()
            
            if instance_area >= self.cd_min_instance_area:
                filtered_mask = np.maximum(filtered_mask, instance_mask)
        
        change_pred = torch.from_numpy(filtered_mask).long().to(change_pred.device)
        
        return change_pred

    def _save_visualization_results(self, image1_path, seg1_pred, seg2_pred, change_pred, ori_shape):
        import os
        from pathlib import Path
        
        image1_name = Path(image1_path).stem
        
        t1_dir = Path(self.vis_dir) / 't1'
        t2_dir = Path(self.vis_dir) / 't2'
        cd_dir = Path(self.vis_dir) / 'cd'
        t1_dir.mkdir(parents=True, exist_ok=True)
        t2_dir.mkdir(parents=True, exist_ok=True)
        cd_dir.mkdir(parents=True, exist_ok=True)
        
        change_pred_np = change_pred.cpu().numpy()  # [H, W]
        change_pred_uint8 = (change_pred_np * 255).astype('uint8')  # 1->255, 0->0
        change_image = Image.fromarray(change_pred_uint8, mode='L')
        change_image.save(cd_dir / f"{image1_name}.png")
        
        seg1_pred_np = seg1_pred.cpu().numpy()  # [H, W]
        seg1_pred_uint8 = (seg1_pred_np * 255).astype('uint8')  # 1->255, 0->0
        seg1_image = Image.fromarray(seg1_pred_uint8, mode='L')
        seg1_image.save(t1_dir / f"{image1_name}.png")
        
        seg2_pred_np = seg2_pred.cpu().numpy()  # [H, W]
        seg2_pred_uint8 = (seg2_pred_np * 255).astype('uint8')  # 1->255, 0->0
        seg2_image = Image.fromarray(seg2_pred_uint8, mode='L')
        seg2_image.save(t2_dir / f"{image1_name}.png")
    
    def _forward(data_samples):
        """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices