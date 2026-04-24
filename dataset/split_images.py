#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Divide the 1024x1024 images in the test folder into 16 equal parts, each 256x256.
Naming rule: The original file name followed by 1, 2, ..., 16, in the order from left to right and from top to bottom.
"""

import os
from pathlib import Path
from PIL import Image
import argparse


def split_image(image_path, output_dir, base_name):
    img = Image.open(image_path)
    width, height = img.size
    
    if width != 1024 or height != 1024:
        print(f"Warning: {image_path} is not 1024x1024, got {width}x{height}, skipping...")
        return
    
    split_size = 256
    grid_size = 4  
    
    ext = Path(image_path).suffix
    
    idx = 1
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * split_size
            top = row * split_size
            right = left + split_size
            bottom = top + split_size
            
            crop = img.crop((left, top, right, bottom))
            
            output_name = f"{base_name}_{idx}{ext}"
            output_path = os.path.join(output_dir, output_name)
            crop.save(output_path)
            print(f"Saved: {output_path}")
            idx += 1


def process_folder(input_folder, output_folder, folder_name):
    input_path = Path(input_folder) / folder_name
    output_path = Path(output_folder) / folder_name
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix in image_extensions and f.is_file()]
    
    print(f"\nProcessing folder: {folder_name} ({len(image_files)} images)")
    
    for img_file in sorted(image_files):
        base_name = img_file.stem  
        split_image(str(img_file), str(output_path), base_name)


def main():
    parser = argparse.ArgumentParser(description='Split 1024x1024 images into 16 equal parts (256x256 each, 4x4 grid)')
    parser.add_argument('--input', type=str, 
                       default='./dataset/LEVIR-CD/test',
                       help='Input test folder path')
    parser.add_argument('--output', type=str,
                       default='./dataset/LEVIR-CD/test_256',
                       help='Output folder path')
    
    args = parser.parse_args()
    
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    
    if not input_folder.exists():
        print(f"Error: Input folder {input_folder} does not exist!")
        return
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    subfolders = ['A', 'B', 'label']
    
    for subfolder in subfolders:
        subfolder_path = input_folder / subfolder
        if subfolder_path.exists():
            process_folder(input_folder, output_folder, subfolder)
        else:
            print(f"Warning: Folder {subfolder} does not exist, skipping...")
    
    print(f"\nDone! All images have been split and saved to {output_folder}")


if __name__ == '__main__':
    main()

