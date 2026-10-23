"""
data/download_data.py
Prepares animal species data for training.

Default: CIFAR-10 animal subset (real photos via torchvision).
Optional: STL-10 on disk as JPEG folders, or --demo-sample for synthetic smoke tests only.
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


def _save_split(split_name: str, images, labels, output_dir: str) -> None:
    print(f"Saving {split_name} split ({len(images)} images)...")
    for i, (img, label) in enumerate(zip(images, labels)):
        class_folder = os.path.join(output_dir, split_name, label)
        os.makedirs(class_folder, exist_ok=True)
        img.save(os.path.join(class_folder, f"{label}_{i:05d}.jpg"), quality=95)


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


def download_and_split_stl10(raw_dir: str, output_dir: str, seed: int = 42) -> None:
    from sklearn.model_selection import train_test_split
    import torchvision.datasets as datasets

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    _reset_split_dirs(output_dir)
    meta = os.path.join(output_dir, "dataset_meta.json")
    if os.path.isfile(meta):
        os.remove(meta)

    print("Downloading STL-10 dataset...")
    dataset = datasets.STL10(root=raw_dir, split="train", download=True)
    test_dataset = datasets.STL10(root=raw_dir, split="test", download=True)

    stl10_classes = [
        "airplane", "bird", "car", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]
    target_indices = {stl10_classes.index(cls): cls for cls in ANIMAL_CLASSES if cls in stl10_classes}

    all_images = []
    all_labels = []

    for ds in [dataset, test_dataset]:
        for img, label in zip(ds.data, ds.labels):
            if label in target_indices:
                img_pil = Image.fromarray(np.transpose(img, (1, 2, 0)))
                all_images.append(img_pil)
                all_labels.append(target_indices[label])

    print(f"Total STL-10 animal images: {len(all_images)}.")

    idx = np.arange(len(all_images))
    train_idx, temp_idx, _, y_temp = train_test_split(
        idx, all_labels, test_size=0.30, random_state=seed, stratify=all_labels
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        imgs = [all_images[i] for i in indices]
        lbls = [all_labels[i] for i in indices]
        _save_split(split_name, imgs, lbls, output_dir)

    print(f"Data prepared at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and split animal image dataset")
    parser.add_argument(
        "--demo-sample",
        action="store_true",
        help="Synthetic colored tiles only (smoke test; not real animals)",
    )
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "stl10"),
        default="cifar10",
        help="Real photo dataset (default: cifar10)",
    )
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.demo_sample:
        create_sample_dataset(args.output_dir)
        return

    if args.dataset == "cifar10":
        prepare_cifar10(args.raw_dir, args.output_dir, seed=args.seed)
    else:
        download_and_split_stl10(args.raw_dir, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
