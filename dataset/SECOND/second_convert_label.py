"""
Convert class labels from 0/255 format to 0/1 format

This script converts binary class labels for all classes except building:
    - 255 pixels -> 1
    - 0 pixels -> 0 (unchanged)

Target classes: water, ground, low vegetation, tree, playground
"""

import os
import cv2
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Classes to convert (excluding building)
TARGET_CLASSES = ['water', 'ground', 'low_vegetation', 'tree', 'playground' ,'building']


def convert_label(input_path, output_path):
    """
    Convert label from 0/255 to 0/1 format
    
    Args:
        input_path: Path to input label image (0/255 format)
        output_path: Path to save output label image (0/1 format)
    """
    # Read input label
    label = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if label is None:
        print(f"Warning: Could not read {input_path}")
        return False
    
    # Convert: 255 -> 1, 0 -> 0
    label_converted = (label == 255).astype(np.uint8)
    
    # Save converted label
    cv2.imwrite(output_path, label_converted)
    return True


def process_labels(input_dir, output_dir, class_name):
    """
    Process all label images in input_dir and save to output_dir
    
    Args:
        input_dir: Directory containing input labels (0/255 format)
        output_dir: Directory to save output labels (0/1 format)
        class_name: Name of the class (for progress display)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = [f for f in os.listdir(input_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {input_dir}")
        return 0
    
    success_count = 0
    for filename in tqdm(image_files, desc=f"Converting {class_name}", unit="image", ncols=100):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if convert_label(input_path, output_path):
            success_count += 1
    
    return success_count


def main():
    """Main execution"""
    # Base directory
    base_dir = Path(__file__).parent
    
    # Output base directory
    output_base_dir = base_dir / 'test'
    
    print("=" * 60)
    print("Converting class labels from 0/255 to 0/1 format")
    print("=" * 60)
    print(f"Output base: {output_base_dir}")
    print(f"Target classes: {', '.join(TARGET_CLASSES)}")
    print("=" * 60)
    
    # Process each target class
    converted_dirs = []
    total_converted = 0
    
    for class_name in TARGET_CLASSES:
        # Input directory (0/255 format)
        input_dir = base_dir / 'test' / f'{class_name}_label'
        
        # Output directory (0/1 format)
        output_dir = base_dir / 'test' / f'{class_name}_label_cvt'
        
        print(f"\nProcessing class: {class_name}")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        
        if not input_dir.exists():
            print(f"  Warning: Input directory does not exist, skipping...")
            continue
        
        # Process labels
        success_count = process_labels(str(input_dir), str(output_dir), class_name)
        total_converted += success_count
        converted_dirs.append(str(output_dir))
        print(f"  Successfully converted {success_count} images")
    
    print("\n" + "=" * 60)
    print("Conversion completed!")
    print("=" * 60)
    print(f"Total images converted: {total_converted}")
    print("Converted label directories:")
    for dir_path in converted_dirs:
        print(f"  - {dir_path}")


if __name__ == "__main__":
    main()

