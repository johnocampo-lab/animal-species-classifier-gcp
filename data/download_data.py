"""
data/download_data.py
Downloads an animal species image dataset (e.g. STL-10 animal subset or Oxford-IIIT Pets)
and performs a strict, reproducible stratified split into train, val, and test partitions
following ML Best Practices.
"""

import argparse
import os
import shutil
import numpy as np
from PIL import Image

# 6 animal classes from STL-10 (classes 1, 2, 4, 5, 6, 7 in STL-10)
ANIMAL_CLASSES = ["bird", "cat", "deer", "dog", "frog", "horse"]

def create_sample_dataset(output_dir: str, num_samples_per_class: int = 40):
    """Generates a quick dummy dataset for testing the pipeline end-to-end without waiting for downloads."""
    print("Generating demo sample dataset for instant end-to-end pipeline validation...")
    for split in ["train", "val", "test"]:
        split_count = int(num_samples_per_class * (0.7 if split == "train" else 0.15))
        for class_name in ANIMAL_CLASSES:
            class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(split_count):
                # Distinct base color per class with some noise
                color_seed = abs(hash(class_name)) % 255
                img_array = np.full((128, 128, 3), (color_seed, 150, 200 - color_seed), dtype=np.uint8)
                noise = np.random.randint(0, 30, (128, 128, 3), dtype=np.uint8)
                img = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
                img.save(os.path.join(class_dir, f"{class_name}_{i:04d}.jpg"))
    print(f"Sample dataset created successfully in: {output_dir}")

def download_and_split_stl10(raw_dir: str, output_dir: str, seed: int = 42):
    """
    Downloads STL-10 and filters for animal classes, splitting strictly into
    train (70%), val (15%), and test (15%).
    """
    from sklearn.model_selection import train_test_split
    import torchvision.datasets as datasets

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading STL-10 dataset (lightweight color images)...")
    dataset = datasets.STL10(root=raw_dir, split="train", download=True)
    test_dataset = datasets.STL10(root=raw_dir, split="test", download=True)
    
    # STL-10 classes: airplane(0), bird(1), car(2), cat(3), deer(4), dog(5), frog(6), horse(7), ship(8), truck(9)
    # Target animal classes mapping
    stl10_classes = ["airplane", "bird", "car", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    target_indices = {stl10_classes.index(cls): cls for cls in ANIMAL_CLASSES if cls in stl10_classes}

    all_images = []
    all_labels = []

    for ds in [dataset, test_dataset]:
        for img, label in zip(ds.data, ds.labels):
            if label in target_indices:
                # img in STL-10 is (3, 96, 96) numpy array
                img_pil = Image.fromarray(np.transpose(img, (1, 2, 0)))
                all_images.append(img_pil)
                all_labels.append(target_indices[label])

    print(f"Total animal images collected: {len(all_images)} across {len(set(all_labels))} classes.")

    # Strict ML Featurization Split: 70% Train, 15% Validation, 15% Test
    idx = np.arange(len(all_images))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, all_labels, test_size=0.30, random_state=seed, stratify=all_labels
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    splits = {
        "train": (train_idx, y_train),
        "val": (val_idx, y_val),
        "test": (test_idx, y_test)
    }

    for split_name, (indices, labels) in splits.items():
        print(f"Saving {split_name} split ({len(indices)} images)...")
        for i, (img_idx, label) in enumerate(zip(indices, labels)):
            class_folder = os.path.join(output_dir, split_name, label)
            os.makedirs(class_folder, exist_ok=True)
            img = all_images[img_idx]
            img.save(os.path.join(class_folder, f"{label}_{i:05d}.jpg"))

    print(f"Data successfully prepared at {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download and split animal dataset")
    parser.add_argument("--demo-sample", action="store_true", help="Generate small demo sample dataset immediately")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save train/val/test splits")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory for raw downloads")
    args = parser.parse_args()

    if args.demo_sample:
        create_sample_dataset(args.output_dir)
    else:
        try:
            download_and_split_stl10(args.raw_dir, args.output_dir)
        except Exception as e:
            print(f"Encountered error downloading STL-10: {e}")
            print("Falling back to demo sample dataset...")
            create_sample_dataset(args.output_dir)

if __name__ == "__main__":
    main()
