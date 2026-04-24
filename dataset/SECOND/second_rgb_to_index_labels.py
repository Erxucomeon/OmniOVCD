"""
Convert RGB semantic labels to index labels for SECOND dataset

This script reads RGB color labels and converts them to single-channel index labels
according to the CLASS_MAPPING in evaluate_second.py

SECOND RGB Color Palette (RGB format):
    [255, 255, 255] - background -> index 0
    [128, 0, 0]     - building   -> index 5
    [0, 0, 255]     - water      -> index 1
    [0, 128, 0]     - low vegetation -> index 3
    [0, 255, 0]     - tree       -> index 4
    [128, 128, 128] - ground     -> index 2
    [255, 0, 0]     - playground -> index 6
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


# RGB color palette (RGB format)
ST_COLORMAP_RGB = [
    [255, 255, 255],  # 0: background
    [128, 0, 0],      # 1: building
    [0, 0, 255],      # 2: water
    [0, 128, 0],      # 3: low vegetation
    [0, 255, 0],      # 4: tree
    [128, 128, 128],  # 5: ground
    [255, 0, 0],      # 6: playground
]

# Index mapping according to evaluate_second.py CLASS_MAPPING
# background: 0, water: 1, ground: 2, low vegetation: 3, tree: 4, building: 5, playground: 6
RGB_TO_INDEX_MAPPING = {
    tuple([255, 255, 255]): 0,  # background -> 0
    tuple([0, 0, 255]): 1,       # water -> 1
    tuple([128, 128, 128]): 2,   # ground -> 2
    tuple([0, 128, 0]): 3,       # low vegetation -> 3
    tuple([0, 255, 0]): 4,       # tree -> 4
    tuple([128, 0, 0]): 5,       # building -> 5
    tuple([255, 0, 0]): 6,       # playground -> 6
}


def rgb_to_index_label(rgb_image):
    """
    Convert RGB semantic label to index label
    
    Args:
        rgb_image: RGB image (H, W, 3) in BGR format (from cv2.imread)
    
    Returns:
        index_label: Single-channel index label (H, W) with values 0-6
    """
    # Convert BGR to RGB
    rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    h, w = rgb_image_rgb.shape[:2]
    index_label = np.zeros((h, w), dtype=np.uint8)
    
    # Create mapping for each RGB color
    for rgb_color, index in RGB_TO_INDEX_MAPPING.items():
        # Find pixels matching this RGB color
        mask = np.all(rgb_image_rgb == rgb_color, axis=2)
        index_label[mask] = index
    
    return index_label


def convert_labels(input_dir, output_dir):
    """
    Convert all RGB label images in input_dir to index labels in output_dir
    
    Args:
        input_dir: Directory containing RGB label images
        output_dir: Directory to save index label images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = [f for f in os.listdir(input_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Use tqdm with more detailed information
    for filename in tqdm(image_files, 
                         desc=f"Converting {os.path.basename(input_dir)}", 
                         unit="image",
                         ncols=100):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read RGB label image (OpenCV reads as BGR)
        rgb_label = cv2.imread(input_path, cv2.IMREAD_COLOR)
        
        if rgb_label is None:
            print(f"\nWarning: Could not read {input_path}")
            continue
        
        # Convert to index label
        index_label = rgb_to_index_label(rgb_label)
        
        # Save as single-channel image
        cv2.imwrite(output_path, index_label)


def main():
    """Main execution"""
    # Base directory
    base_dir = Path(__file__).parent
    
    # Input directories (RGB labels)
    label1_rgb_dir = base_dir / 'test' / 'label1'
    label2_rgb_dir = base_dir / 'test' / 'label2'
    
    # Output directories (index labels)
    label1_index_dir = base_dir / 'test' / 'label1_index'
    label2_index_dir = base_dir / 'test' / 'label2_index'
    
    print("=" * 60)
    print("Converting RGB labels to index labels for SECOND dataset")
    print("=" * 60)
    
    # Convert label1
    print(f"\n[1/2] Processing label1...")
    print(f"  Input:  {label1_rgb_dir}")
    print(f"  Output: {label1_index_dir}")
    convert_labels(str(label1_rgb_dir), str(label1_index_dir))
    
    # Convert label2
    print(f"\n[2/2] Processing label2...")
    print(f"  Input:  {label2_rgb_dir}")
    print(f"  Output: {label2_index_dir}")
    convert_labels(str(label2_rgb_dir), str(label2_index_dir))
    
    print("\n" + "=" * 60)
    print("Conversion completed!")
    print("=" * 60)
    print(f"Index labels saved to:")
    print(f"  - {label1_index_dir}")
    print(f"  - {label2_index_dir}")


if __name__ == "__main__":
    main()

