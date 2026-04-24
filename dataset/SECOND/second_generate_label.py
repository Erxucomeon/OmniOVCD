"""
Generate class change labels from index labels for SECOND dataset

This script generates binary change labels (0/255) from index labels
following the same logic as evaluate_second.py:
    change_label = ((label1 == class_id) | (label2 == class_id)).astype(np.uint8) * 255
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


# Class mapping (same as evaluate_second.py)
CLASS_MAPPING = {
    'background': 0,
    'water': 1,
    'ground': 2,
    'low vegetation': 3,
    'tree': 4,
    'building': 5,
    'playground': 6,
}

# Classes to generate (excluding background and building)
TARGET_CLASSES = ['water', 'ground', 'low vegetation', 'tree', 'playground', 'building']


def generate_class_label(label1_index, label2_index, class_id):
    """
    Generate binary class change label from two index labels
    
    Args:
        label1_index: First time point index label (single channel, 0-6)
        label2_index: Second time point index label (single channel, 0-6)
        class_id: Target class ID (0-6)
    
    Returns:
        class_label: Binary label (0 or 255) where 255 indicates class presence
    """
    # Check if target class appears in either time point
    # Logic: ((label1 == class_id) | (label2 == class_id)).astype(np.uint8) * 255
    class_mask = ((label1_index == class_id) | (label2_index == class_id))
    class_label = class_mask.astype(np.uint8) * 255
    
    return class_label


def process_labels(label1_dir, label2_dir, output_base_dir, class_name, class_id):
    """
    Process all label pairs and generate class labels
    
    Args:
        label1_dir: Directory containing first time point index labels
        label2_dir: Directory containing second time point index labels
        output_base_dir: Base directory to save class labels
        class_name: Name of the class (for folder naming)
        class_id: Class ID (0-6)
    """
    # Create output directory for this class
    # Replace spaces with underscores for folder names
    folder_name = class_name.replace(' ', '_')
    output_dir = os.path.join(output_base_dir, f'{folder_name}_label')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files from label1_dir
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = [f for f in os.listdir(label1_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {label1_dir}")
        return
    
    for filename in tqdm(image_files, desc=f"Generating {class_name} labels", unit="image", ncols=100):
        label1_path = os.path.join(label1_dir, filename)
        label2_path = os.path.join(label2_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read index labels (single channel)
        label1 = cv2.imread(label1_path, cv2.IMREAD_GRAYSCALE)
        label2 = cv2.imread(label2_path, cv2.IMREAD_GRAYSCALE)
        
        if label1 is None:
            print(f"\nWarning: Could not read {label1_path}")
            continue
        if label2 is None:
            print(f"\nWarning: Could not read {label2_path}")
            continue
        
        # Validate shapes
        if label1.shape != label2.shape:
            print(f"\nWarning: Shape mismatch for {filename}: {label1.shape} vs {label2.shape}")
            continue
        
        # Generate class label
        class_label = generate_class_label(label1, label2, class_id)
        
        # Save binary label
        cv2.imwrite(output_path, class_label)
    
    return output_dir


def main():
    """Main execution"""
    # Base directory
    base_dir = Path(__file__).parent
    
    # Input directories (index labels)
    label1_index_dir = base_dir / 'test' / 'label1_index'
    label2_index_dir = base_dir / 'test' / 'label2_index'
    
    # Output base directory
    output_base_dir = base_dir / 'test'
    
    print("=" * 60)
    print("Generating class change labels for SECOND dataset")
    print("=" * 60)
    print(f"Input label1: {label1_index_dir}")
    print(f"Input label2: {label2_index_dir}")
    print(f"Output base:  {output_base_dir}")
    print(f"Target classes: {', '.join(TARGET_CLASSES)}")
    print("=" * 60)
    
    # Process each target class
    generated_dirs = []
    for class_name in TARGET_CLASSES:
        class_id = CLASS_MAPPING[class_name]
        print(f"\nProcessing class: {class_name} (ID: {class_id})")
        output_dir = process_labels(
            str(label1_index_dir), 
            str(label2_index_dir), 
            str(output_base_dir),
            class_name,
            class_id
        )
        generated_dirs.append(output_dir)
    
    print("\n" + "=" * 60)
    print("Generation completed!")
    print("=" * 60)
    print("Generated label directories:")
    for dir_path in generated_dirs:
        print(f"  - {dir_path}")


if __name__ == "__main__":
    main()

