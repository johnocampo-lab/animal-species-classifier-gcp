import os
from unittest.mock import patch

import pytest

from data.dataset import get_dataloaders
from data.download_data import create_sample_dataset
from training import train as train_mod


def _dataloaders_no_workers(*args, **kwargs):
    kwargs["num_workers"] = 0
    return get_dataloaders(*args, **kwargs)


@pytest.mark.integration
def test_train_and_evaluate_smoke(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    eval_dir = tmp_path / "eval"
    create_sample_dataset(str(data_dir))
    with patch("training.train.get_dataloaders", side_effect=_dataloaders_no_workers):
        train_mod.train(
            data_dir=str(data_dir),
            epochs=1,
            batch_size=8,
            lr=0.001,
            output_dir=str(model_dir),
        )
    assert os.path.isfile(model_dir / "model.pt")
    assert os.path.isfile(model_dir / "classes.json")
    assert os.path.isfile(model_dir / "model.onnx")

    from training import evaluate as eval_mod

    with patch("training.evaluate.get_dataloaders", side_effect=_dataloaders_no_workers):
        eval_mod.evaluate(data_dir=str(data_dir), model_dir=str(model_dir), output_dir=str(eval_dir))
    assert os.path.isfile(eval_dir / "metrics.json")
