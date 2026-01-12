#!/usr/bin/env python3
"""
Create a balanced subset of a YOLO dataset with N images per class.
Useful for quick experimentation and debugging training pipelines.
"""

import argparse
import os
import shutil
import yaml
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_data_yaml(source_path: Path) -> dict:
    """Load data.yaml from source dataset."""
    yaml_path = source_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_class_from_label(label_path: Path) -> set:
    """Extract class IDs from a label file."""
    classes = set()
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes.add(int(parts[0]))
    return classes


def collect_images_by_class(source_path: Path, split: str) -> dict:
    """
    Collect all images grouped by the classes they contain.
    Returns: {class_id: [list of image paths]}
    """
    images_dir = source_path / "images" / split
    labels_dir = source_path / "labels" / split
    
    if not images_dir.exists():
        print(f"Warning: {images_dir} does not exist, skipping split '{split}'")
        return {}
    
    class_to_images = defaultdict(list)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(images_dir.glob(ext))
    
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        classes = get_class_from_label(label_path)
        
        for cls_id in classes:
            class_to_images[cls_id].append(img_path)
    
    return class_to_images


def select_balanced_images(class_to_images: dict, images_per_class: int, seed: int = 42) -> set:
    """
    Select up to N images per class, returning a set of unique image paths.
    Uses stratified sampling to ensure diversity.
    """
    random.seed(seed)
    selected_images = set()
    
    for cls_id, images in class_to_images.items():
        # Shuffle and take up to N images
        shuffled = images.copy()
        random.shuffle(shuffled)
        
        count = 0
        for img_path in shuffled:
            if count >= images_per_class:
                break
            if img_path not in selected_images:
                selected_images.add(img_path)
                count += 1
    
    return selected_images


def copy_dataset_subset(
    source_path: Path,
    output_path: Path,
    selected_images: set,
    split: str
):
    """Copy selected images and their labels to output directory."""
    # Create directories
    (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    source_labels = source_path / "labels" / split
    
    copied = 0
    for img_path in tqdm(selected_images, desc=f"Copying {split}"):
        # Copy image
        dst_img = output_path / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)
        
        # Copy label
        label_name = img_path.stem + ".txt"
        src_label = source_labels / label_name
        if src_label.exists():
            dst_label = output_path / "labels" / split / label_name
            shutil.copy2(src_label, dst_label)
        
        copied += 1
    
    return copied


def create_data_yaml(output_path: Path, source_yaml: dict):
    """Create data.yaml for the new subset dataset."""
    new_yaml = source_yaml.copy()
    new_yaml['path'] = str(output_path.absolute())
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(new_yaml, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description='Create a balanced subset of a YOLO dataset with N images per class'
    )
    parser.add_argument(
        '--source', 
        type=str, 
        required=True,
        help='Source dataset directory (must contain data.yaml)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Output directory for the subset dataset'
    )
    parser.add_argument(
        '--images-per-class', 
        type=int, 
        default=50,
        help='Maximum number of images per class (default: 50)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to process (default: train val test)'
    )
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    output_path = Path(args.output)
    
    # Validate source
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")
    
    # Load source data.yaml
    print(f"Loading dataset from: {source_path}")
    source_yaml = load_data_yaml(source_path)
    print(f"  - Classes: {source_yaml.get('nc', 'unknown')}")
    print(f"  - Class names: {source_yaml.get('names', [])[:5]}... (first 5)")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    print(f"Target: {args.images_per_class} images per class")
    
    # Process each split
    total_images = 0
    for split in args.splits:
        print(f"\n--- Processing '{split}' split ---")
        
        # Collect images by class
        class_to_images = collect_images_by_class(source_path, split)
        
        if not class_to_images:
            print(f"  No images found for split '{split}', skipping")
            continue
        
        print(f"  Found {len(class_to_images)} classes with images")
        
        # Select balanced subset
        selected = select_balanced_images(
            class_to_images, 
            args.images_per_class, 
            args.seed
        )
        print(f"  Selected {len(selected)} unique images")
        
        # Copy files
        copied = copy_dataset_subset(source_path, output_path, selected, split)
        total_images += copied
        print(f"  Copied {copied} images")
    
    # Create data.yaml
    yaml_path = create_data_yaml(output_path, source_yaml)
    print(f"\n✓ Created {yaml_path}")
    print(f"✓ Total images in subset: {total_images}")
    
    # Print usage hint
    print(f"\n📌 Usage:")
    print(f"  python train_fast.py --data {yaml_path}")
    print(f"  python train.py --data {yaml_path}")


if __name__ == '__main__':
    main()
