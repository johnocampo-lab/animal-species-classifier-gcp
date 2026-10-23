import os

import torch

from training.model import export_to_onnx, get_model


def test_onnx_export_smoke(tmp_path):
    model = get_model(num_classes=6, pretrained=False)
    out = tmp_path / "m" / "model.onnx"
    export_to_onnx(model, str(out))
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 1000
