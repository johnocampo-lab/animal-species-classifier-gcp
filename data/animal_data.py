"""Shared CIFAR-10 animal subset loading and split logic."""

import json
import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

ANIMAL_CLASSES = ["bird", "cat", "deer", "dog", "frog", "horse"]

CIFAR10_ANIMAL_LABELS = {
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
}

CIFAR10_TO_ANIMAL_INDEX = {cifar_idx: i for i, cifar_idx in enumerate(sorted(CIFAR10_ANIMAL_LABELS))}


class _RemapCifarLabels(Dataset):
    def __init__(self, subset: Subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        img, label = self.subset[index]
        return img, CIFAR10_TO_ANIMAL_INDEX[int(label)]

META_FILENAME = "dataset_meta.json"


def cifar10_animal_indices(
    raw_dir: str, seed: int = 42
) -> Tuple[list[int], list[int], list[int]]:
    """
    Returns (train_indices, val_indices, test_indices) into torchvision CIFAR-10 subsets.
    Train/val indices reference CIFAR train=True; test indices reference train=False.
    """
    train_set = datasets.CIFAR10(root=raw_dir, train=True, download=False)
    test_set = datasets.CIFAR10(root=raw_dir, train=False, download=False)

    train_pool_idx, train_pool_labels = [], []
    for i, label in enumerate(train_set.targets):
        if label in CIFAR10_ANIMAL_LABELS:
            train_pool_idx.append(i)
            train_pool_labels.append(CIFAR10_ANIMAL_LABELS[label])

    idx = np.arange(len(train_pool_idx))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=train_pool_labels,
    )
    train_indices = [train_pool_idx[i] for i in train_idx]
    val_indices = [train_pool_idx[i] for i in val_idx]

    test_indices = [i for i, label in enumerate(test_set.targets) if label in CIFAR10_ANIMAL_LABELS]

    return train_indices, val_indices, test_indices


def write_cifar10_meta(output_dir: str, raw_dir: str, seed: int = 42) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, META_FILENAME)
    with open(meta_path, "w") as f:
        json.dump(
            {"source": "cifar10_animals", "raw_dir": raw_dir, "seed": seed},
            f,
            indent=2,
        )


def meta_path(data_dir: str) -> str:
    return os.path.join(data_dir, META_FILENAME)


def is_cifar10_meta(data_dir: str) -> bool:
    path = meta_path(data_dir)
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        meta = json.load(f)
    return meta.get("source") == "cifar10_animals"


def load_cifar10_meta(data_dir: str) -> dict:
    with open(meta_path(data_dir)) as f:
        return json.load(f)


def get_cifar10_dataloaders(
    raw_dir: str,
    seed: int,
    batch_size: int,
    train_transform,
    eval_transform,
    num_workers: int = 2,
):
    print("Using CIFAR-10 animal photos (torchvision; test = official CIFAR-10 test fold).")
    datasets.CIFAR10(root=raw_dir, train=True, download=True)
    datasets.CIFAR10(root=raw_dir, train=False, download=True)

    train_indices, val_indices, test_indices = cifar10_animal_indices(raw_dir, seed)

    train_base = datasets.CIFAR10(root=raw_dir, train=True, transform=train_transform)
    test_base = datasets.CIFAR10(root=raw_dir, train=False, transform=eval_transform)
    val_base = datasets.CIFAR10(root=raw_dir, train=True, transform=eval_transform)

    train_ds = _RemapCifarLabels(Subset(train_base, train_indices))
    val_ds = _RemapCifarLabels(Subset(val_base, val_indices))
    test_ds = _RemapCifarLabels(Subset(test_base, test_indices))

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, list(ANIMAL_CLASSES)
