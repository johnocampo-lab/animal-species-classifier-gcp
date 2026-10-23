import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root():
    return ROOT


@pytest.fixture
def tmp_processed(tmp_path):
    return tmp_path / "processed"


@pytest.fixture
def demo_imagefolder(tmp_processed):
    from data.download_data import create_sample_dataset

    create_sample_dataset(str(tmp_processed))
    return tmp_processed
