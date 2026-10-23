"""
data/download_data.py
Prepares animal species data for training.

Default: CIFAR-10 animal subset (real photos via torchvision).
Use --demo-sample only for fast CI/smoke tests (synthetic tiles, not real animals).
"""

import argparse
import os
import shutil

import numpy as np
from PIL import Image

from data.animal_data import ANIMAL_CLASSES, cifar10_animal_indices, write_cifar10_meta


def _reset_split_dirs(output_dir: str) -> None:
    for split in ("train", "val", "test"):
        split_dir = os.path.join(output_dir, split)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)


def create_sample_dataset(output_dir: str, num_samples_per_class: int = 40):
    """Synthetic colored tiles — pipeline smoke test only, not real animals."""
    print("WARNING: Generating synthetic demo data (not real animal photos).")
    _reset_split_dirs(output_dir)
    meta = os.path.join(output_dir, "dataset_meta.json")
    if os.path.isfile(meta):
        os.remove(meta)
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_count = int(num_samples_per_class * (0.7 if split == "train" else 0.15))
        for class_name in ANIMAL_CLASSES:
            class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(split_count):
                color_seed = abs(hash(class_name)) % 255
                img_array = np.full((128, 128, 3), (color_seed, 150, 200 - color_seed), dtype=np.uint8)
                noise = np.random.randint(0, 30, (128, 128, 3), dtype=np.uint8)
                img = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
                img.save(os.path.join(class_dir, f"{class_name}_{i:04d}.jpg"))
    print(f"Sample dataset created in: {output_dir}")


def prepare_cifar10(raw_dir: str, output_dir: str, seed: int = 42) -> None:
    """Download CIFAR-10 and register torchvision-backed splits (no JPEG export)."""
    import torchvision.datasets as datasets

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    _reset_split_dirs(output_dir)

    batches_dir = os.path.join(raw_dir, "cifar-10-batches-py")
    if not os.path.isdir(batches_dir):
        print("Downloading CIFAR-10 (real animal photos; loaded via torchvision)...")
        datasets.CIFAR10(root=raw_dir, train=True, download=True)
        datasets.CIFAR10(root=raw_dir, train=False, download=True)
    else:
        print(f"Using existing CIFAR-10 at {batches_dir}")

    train_i, val_i, test_i = cifar10_animal_indices(raw_dir, seed)
    print(
        f"CIFAR-10 animals — train: {len(train_i)}, val: {len(val_i)}, "
        f"test (official fold): {len(test_i)}"
    )
    write_cifar10_meta(output_dir, raw_dir, seed)
    print(f"Registered dataset at {output_dir}/dataset_meta.json")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare animal image dataset")
    parser.add_argument(
        "--demo-sample",
        action="store_true",
        help="Synthetic colored tiles only (smoke test; not real animals)",
    )
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.demo_sample:
        create_sample_dataset(args.output_dir)
        return

    prepare_cifar10(args.raw_dir, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
