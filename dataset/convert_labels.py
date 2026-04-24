#!/usr/bin/env python3
"""
Convert label images: pixels > 128 -> 1, pixels <= 128 -> 0
"""
import os
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

def convert_label_image(input_path, output_path):
    """Convert label image: > 128 -> 1, <= 128 -> 0"""
    # Read image
    img = Image.open(input_path)
    img_array = np.array(img)
    
    # Convert: > 128 -> 1, <= 128 -> 0
    converted = (img_array > 128).astype(np.uint8)
    
    # Save converted image
    converted_img = Image.fromarray(converted)
    converted_img.save(output_path)
    print(f"Converted: {os.path.basename(input_path)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
                       default='./dataset/S2Looking/test/label',)
    parser.add_argument('--output', type=str,
                       default='./dataset/S2Looking/test/label_cvt',)
    
    args = parser.parse_args()

    # Paths
    label_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files
    png_files = sorted(label_dir.glob("*.png"))
    
    print(f"Found {len(png_files)} PNG files to convert")
    print(f"Input directory: {label_dir}")
    print(f"Output directory: {output_dir}")
    
    # Convert each image
    for png_file in png_files:
        output_path = output_dir / png_file.name
        convert_label_image(png_file, output_path)
    
    print(f"\nConversion complete! Converted {len(png_files)} images.")

if __name__ == "__main__":
    main()

