import json
from unittest.mock import MagicMock, patch

import numpy as np

from data.animal_data import (
    ANIMAL_CLASSES,
    CIFAR10_TO_ANIMAL_INDEX,
    _RemapCifarLabels,
    cifar10_animal_indices,
    is_cifar10_meta,
    write_cifar10_meta,
)


def test_cifar_label_remap_covers_six_classes():
    assert len(CIFAR10_TO_ANIMAL_INDEX) == 6
    assert set(CIFAR10_TO_ANIMAL_INDEX.values()) == set(range(6))


class _FakeSubset:
    def __len__(self):
        return 3

    def __getitem__(self, i):
        return None, [2, 3, 7][i]


def test_remap_cifar_labels():
    ds = _RemapCifarLabels(_FakeSubset())
    assert ds[0][1] == CIFAR10_TO_ANIMAL_INDEX[2]
    assert ds[2][1] == CIFAR10_TO_ANIMAL_INDEX[7]


def test_dataset_meta_roundtrip(tmp_path):
    write_cifar10_meta(str(tmp_path), "data/raw", seed=99)
    assert is_cifar10_meta(str(tmp_path))
    with open(tmp_path / "dataset_meta.json") as f:
        meta = json.load(f)
    assert meta["source"] == "cifar10_animals"
    assert meta["seed"] == 99


@patch("data.animal_data.datasets.CIFAR10")
def test_cifar_split_sizes(mock_cifar):
    train_targets = []
    for label in range(10):
        train_targets.extend([label] * 5000)
    test_targets = []
    for label in range(10):
        test_targets.extend([label] * 1000)

    train_ds = MagicMock()
    train_ds.targets = train_targets
    test_ds = MagicMock()
    test_ds.targets = test_targets
    mock_cifar.side_effect = [train_ds, test_ds]

    train_i, val_i, test_i = cifar10_animal_indices("/fake", seed=42)
    assert len(train_i) + len(val_i) == 30000
    assert len(test_i) == 6000
    assert len(train_i) > len(val_i)
    assert len(ANIMAL_CLASSES) == 6
