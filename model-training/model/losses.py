"""
복합 손실 함수: L1 + SSIM + Perceptual + SimLoss

SimLoss: 모델 출력을 CVD 시뮬레이션한 결과가 원본 이미지와 유사하도록 강제.
→ Brettel이 최적화하지 않는 "CVD 환자 시점의 지각 품질"을 직접 학습.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn


# ── 미분 가능한 CVD 시뮬레이션 (Brettel 행렬, torch 버전) ─────────────────

_RGB_TO_LMS_NP = np.array([
    [17.8824,  43.5161,  4.11935],
    [ 3.45565, 27.1554,  3.86714],
    [ 0.02996,  0.18431, 1.46720],
], dtype=np.float32)

_LMS_TO_RGB_NP = np.linalg.inv(_RGB_TO_LMS_NP).astype(np.float32)

_CVD_MATS_NP = {
    0.0: np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    0.5: np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]], dtype=np.float32),
    1.0: np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]], dtype=np.float32),
}

_CVD_KEYS = [0.0, 0.5, 1.0]


def _simulate_cvd_batch(rgb: torch.Tensor, cvd_vals: torch.Tensor) -> torch.Tensor:
    """
    미분 가능한 CVD 시뮬레이션.
    rgb:      (B, 3, H, W) [0, 1]
    cvd_vals: (B,)  — 0.0=p, 0.5=d, 1.0=t
    returns:  (B, 3, H, W) simulated
    """
    device = rgb.device
    B, _, H, W = rgb.shape
    result = torch.zeros_like(rgb)

    rgb2lms = torch.tensor(_RGB_TO_LMS_NP, device=device)
    lms2rgb = torch.tensor(_LMS_TO_RGB_NP, device=device)

    for i in range(B):
        val = cvd_vals[i].item()
        key = min(_CVD_KEYS, key=lambda k: abs(k - val))
        cvd_mat = torch.tensor(_CVD_MATS_NP[key], device=device)

        px = rgb[i].reshape(3, -1)          # (3, H*W)
        lms = rgb2lms @ px
        sim_lms = cvd_mat @ lms
        sim_rgb = lms2rgb @ sim_lms
        result[i] = sim_rgb.reshape(3, H, W)

    return result.clamp(0, 1)


# ── 메인 손실 클래스 ────────────────────────────────────────────────────────

class CVDCorrectionLoss(nn.Module):
    """
    Args:
        l1_w:   L1 Loss 가중치
        ssim_w: SSIM Loss 가중치
        perc_w: Perceptual Loss 가중치 (VGG16)
        sim_w:  SimLoss 가중치 — CVD 시뮬 후 원본 유사도
    """

    def __init__(
        self,
        l1_w:   float = 1.0,
        ssim_w: float = 0.5,
        perc_w: float = 0.1,
        sim_w:  float = 0.5,
    ):
        super().__init__()
        self.l1_w   = l1_w
        self.ssim_w = ssim_w
        self.perc_w = perc_w
        self.sim_w  = sim_w

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features[:16]))
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        pred:    torch.Tensor,
        target:  torch.Tensor,
        input_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        pred:    (B, 3, H, W) 모델 출력 [0, 1]
        target:  (B, 3, H, W) Brettel 정답 [0, 1]
        input_t: (B, 4, H, W) 원본 입력 — [:, :3]=원본RGB, [:, 3]=CVD값
                 제공 시 SimLoss 활성화
        """
        # 1. L1
        l1 = self.l1_loss(pred, target)

        # 2. SSIM
        ssim_val  = ssim_fn(pred, target, data_range=1.0)
        ssim_loss = 1.0 - ssim_val

        # 3. Perceptual
        feat_pred   = self.feature_extractor(pred)
        feat_target = self.feature_extractor(target)
        perc = self.l1_loss(feat_pred, feat_target)

        total = self.l1_w * l1 + self.ssim_w * ssim_loss + self.perc_w * perc

        components = {
            "loss_l1":   l1.item(),
            "loss_ssim": ssim_loss.item(),
            "loss_perc": perc.item(),
            "ssim":      ssim_val.item(),
            "loss_sim":  0.0,
        }

        # 4. SimLoss (선택적)
        if input_t is not None and self.sim_w > 0:
            original_rgb = input_t[:, :3]                  # (B, 3, H, W)
            cvd_vals     = input_t[:, 3, 0, 0]             # (B,)
            sim_pred     = _simulate_cvd_batch(pred, cvd_vals)

            sim_l1       = self.l1_loss(sim_pred, original_rgb)
            sim_ssim_val = ssim_fn(sim_pred, original_rgb, data_range=1.0)
            sim_loss     = sim_l1 + (1.0 - sim_ssim_val)

            total += self.sim_w * sim_loss
            components["loss_sim"] = sim_loss.item()

        return total, components
