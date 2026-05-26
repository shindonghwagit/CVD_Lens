"""
Phase 2 mask predictor.

Architecture:
    MobileNetV2 encoder + U-Net decoder + scSE decoder attention

Input:
    (B, 8, 256, 256)
    - original RGB: 3 channels
    - CVD simulation RGB: 3 channels
    - difference map: 1 channel
    - CVD type channel: 1 channel

Output:
    (B, 1, 256, 256), soft correction mask M in [0, 1]
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_mask_model() -> nn.Module:
    return smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=8,
        classes=1,
        activation="sigmoid",
        decoder_attention_type="scse",
    )


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


if __name__ == "__main__":
    model = build_mask_model()
    params = count_parameters(model)
    print(f"total parameters:     {params['total']:,}")
    print(f"trainable parameters: {params['trainable']:,}")

    dummy = torch.randn(2, 8, 256, 256)
    with torch.no_grad():
        out = model(dummy)
    print(f"input shape:  {dummy.shape}")
    print(f"output shape: {out.shape}")
    print(f"output range: [{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == (2, 1, 256, 256)
    assert 0.0 <= out.min() and out.max() <= 1.0
    print("model check passed")
