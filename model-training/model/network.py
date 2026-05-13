"""
Phase 2 - 모델 정의

MobileNetV2 Encoder + U-Net Decoder
- encoder: MobileNetV2 (ImageNet pretrained)
- decoder: U-Net (skip connection 5단계)
- input:  (B, 4, 256, 256)  — RGB 3ch + CVD type 1ch
- output: (B, 3, 256, 256)  — 보정된 RGB [0, 1]
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model() -> nn.Module:
    """
    CVD 보정 모델 생성.

    smp.Unet 한 줄로 얻는 것:
    - MobileNetV2 encoder (ImageNet pretrained, Inverted Residual Block)
    - U-Net decoder (skip connection 자동 구성, 5단계)
    - ~2.5M 파라미터
    """
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=4,       # RGB 3ch + CVD type 1ch
        classes=3,           # 출력 RGB 3ch
        activation="sigmoid" # 출력 [0, 1] 범위
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """파라미터 수 집계"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


if __name__ == "__main__":
    model = build_model()
    params = count_parameters(model)
    print(f"총 파라미터:     {params['total']:,}")
    print(f"학습 파라미터:   {params['trainable']:,}")
    print(f"고정 파라미터:   {params['frozen']:,}")

    # 입출력 shape 확인
    dummy = torch.randn(2, 4, 256, 256)
    with torch.no_grad():
        out = model(dummy)
    print(f"\n입력 shape: {dummy.shape}")
    print(f"출력 shape: {out.shape}")
    print(f"출력 범위:  [{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == (2, 3, 256, 256)
    assert 0.0 <= out.min() and out.max() <= 1.0
    print("\n모델 검증 통과")
