_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_levir_cd.txt',  
    prob_thd=0.4,
    confidence_threshold=0.5,
    use_sem_seg=True,  
    use_transformer_decoder=True,  
    use_presence_score=True,  
    change_detection_method='instance_method',  
    instance_iou_threshold=0.25, 
    t12_min_instance_area=0,
    cd_min_instance_area=0, 
    enable_vis=False, 
    vis_dir='./vis_levir_cd',  
)

# dataset settings
dataset_type = 'CDDataset'
data_root = '/media/storage/zhangxu_space/OmniOVCD/dataset/LEVIR-CD/test_256'

test_pipeline = [
    dict(type='LoadCDImagesFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs', meta_keys=('img1_path', 'img2_path', 'seg_map_path', 'ori_shape',
                                          'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                          'flip_direction', 'reduce_zero_label'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(
            img1_path='A',          
            img2_path='B',        
            seg_map_path='label_cvt'),  
        pipeline=test_pipeline))

