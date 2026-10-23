"""
data/dataset.py
PyTorch Dataset loaders and strict transform pipelines.
Enforces ML Best Practices:
1. Training augmentations (RandomResizedCrop, Flip, Rotation) ONLY on train set.
2. Validation & Test splits use deterministic resizing and normalization (No leakage).
"""

import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Standard ImageNet normalization parameters
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def get_transforms(img_size: int = 224):
    """
    Returns separate transforms for training and validation/testing
    to prevent data leakage and enforce strict featurization ordering.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        NORMALIZE
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        NORMALIZE
    ])

    return train_transform, eval_transform

def get_dataloaders(data_dir: str, batch_size: int = 32, img_size: int = 224, num_workers: int = 2):
    """
    Creates DataLoaders for train, val, and test splits.
    """
    train_tf, eval_tf = get_transforms(img_size)

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"Expected 'train', 'val', and 'test' subdirectories in {data_dir}. "
            "Please run 'python data/download_data.py' first."
        )

    train_dataset = datasets.ImageFolder(train_path, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_path, transform=eval_tf)
    test_dataset = datasets.ImageFolder(test_path, transform=eval_tf)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    class_names = train_dataset.classes
    return loaders, class_names
