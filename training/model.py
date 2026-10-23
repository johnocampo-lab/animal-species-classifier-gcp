"""
training/model.py
Defines the transfer learning architecture using MobileNetV3-Small.
Optimized for high accuracy and ultra-low CPU inference latency on Cloud Run.
Includes an export routine to ONNX for minimal container image footprint (<150MB).
"""

import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def get_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Builds a MobileNetV3-Small model adapted for animal species classification.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    # Replace final classifier layer
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

def export_to_onnx(model: nn.Module, output_path: str, img_size: int = 224):
    """
    Exports PyTorch model to ONNX format.
    Allows serving with onnxruntime in production without PyTorch dependency.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, requires_grad=False)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX model exported to: {output_path} (Size: ~{os.path.getsize(output_path) / (1024*1024):.2f} MB)")
