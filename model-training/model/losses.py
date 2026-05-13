"""
Phase 2 - 복합 손실 함수

L1 + SSIM + Perceptual (VGG16)

각 Loss의 역할:
- L1:         픽셀 단위 색상 정확도 (MSE보다 sharp한 결과)
- SSIM:       텍스처, 엣지 등 구조 보존
- Perceptual: 고차원 시각적 자연스러움 (VGG16 feature 기반)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn


class CVDCorrectionLoss(nn.Module):
    """
    복합 손실 함수: L1 + SSIM + Perceptual

    Args:
        l1_w:   L1 Loss 가중치 (기본 1.0)
        ssim_w: SSIM Loss 가중치 (기본 0.5)
        perc_w: Perceptual Loss 가중치 (기본 0.1)
    """

    def __init__(self, l1_w: float = 1.0, ssim_w: float = 0.5, perc_w: float = 0.1):
        super().__init__()
        self.l1_w   = l1_w
        self.ssim_w = ssim_w
        self.perc_w = perc_w

        # VGG16 feature extractor (block3_conv3 = features[:16])
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features[:16]))
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.l1_loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred:   (B, 3, H, W) 모델 출력 [0, 1]
            target: (B, 3, H, W) 정답 이미지 [0, 1]

        Returns:
            total_loss: 스칼라
            components: 각 loss 값 딕셔너리 (로깅용)
        """
        # 1. L1 Loss
        l1 = self.l1_loss(pred, target)

        # 2. SSIM Loss (1 - SSIM, 낮을수록 좋음)
        ssim_val = ssim_fn(pred, target, data_range=1.0)
        ssim_loss = 1.0 - ssim_val

        # 3. Perceptual Loss (VGG feature L1)
        feat_pred   = self.feature_extractor(pred)
        feat_target = self.feature_extractor(target)
        perc = self.l1_loss(feat_pred, feat_target)

        total = self.l1_w * l1 + self.ssim_w * ssim_loss + self.perc_w * perc

        components = {
            "loss_l1":   l1.item(),
            "loss_ssim": ssim_loss.item(),
            "loss_perc": perc.item(),
            "ssim":      ssim_val.item(),
        }
        return total, components


if __name__ == "__main__":
    criterion = CVDCorrectionLoss()

    pred   = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)

    loss, components = criterion(pred, target)
    print(f"total loss: {loss.item():.4f}")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")

    # 완전히 동일한 입력이면 L1=0, SSIM=1
    loss_same, comp_same = criterion(target, target)
    assert comp_same["loss_l1"] < 1e-6, "동일 입력에서 L1이 0이 아님"
    assert comp_same["ssim"] > 0.999,   "동일 입력에서 SSIM이 1이 아님"
    print("\n동일 입력 검증 통과")
