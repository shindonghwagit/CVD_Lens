"""
Spatial-weighted multi-objective loss for selective CVD correction (β v1).

Core idea: use CVD confusion strength as a spatial loss weight. Confusion
regions get pushed toward visibility; non-confusion regions are pinned to
the original. This trains "selective correction" directly into the loss
rather than relying on a learned mask.

L_total = α·L_preserve + β·L_visibility + η·L_structure + γ·L_color

    w            = normalized Lab delta-E(original, CVD_sim(original))
                   then Gaussian-blurred for soft boundaries
    L_preserve   = mean((1 - w) · |out - orig|)          # 비혼동 영역 보존
    L_visibility = mean(w · |CVD_sim(out) - orig|)       # 혼동 영역 보정
    L_structure  = 1 - SSIM(out, orig)                   # 전체 구조 보존
    L_color      = |mean(out) - mean(orig)|
                 + |std(out)  - std(orig)|               # 색 분포 보존
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn


# ── 미분 가능한 CVD 시뮬레이션 (Brettel 행렬) ───────────────────────────────

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


def simulate_cvd_batch(rgb: torch.Tensor, cvd_vals: torch.Tensor) -> torch.Tensor:
    """
    미분 가능한 CVD 시뮬레이션.
    rgb:      (B, 3, H, W) [0, 1]
    cvd_vals: (B,)  — 0.0=p, 0.5=d, 1.0=t
    returns:  (B, 3, H, W) simulated, clipped to [0, 1]
    """
    device = rgb.device
    B, _, H, W = rgb.shape
    result = torch.zeros_like(rgb)

    rgb2lms = torch.tensor(_RGB_TO_LMS_NP, device=device, dtype=rgb.dtype)
    lms2rgb = torch.tensor(_LMS_TO_RGB_NP, device=device, dtype=rgb.dtype)

    for i in range(B):
        val = cvd_vals[i].item()
        key = min(_CVD_KEYS, key=lambda k: abs(k - val))
        cvd_mat = torch.tensor(_CVD_MATS_NP[key], device=device, dtype=rgb.dtype)

        px = rgb[i].reshape(3, -1)          # (3, H*W)
        lms = rgb2lms @ px
        sim_lms = cvd_mat @ lms
        sim_rgb = lms2rgb @ sim_lms
        result[i] = sim_rgb.reshape(3, H, W)

    return result.clamp(0, 1)


# ── RGB → Lab (torch native, D65) ─────────────────────────────────────────
# w 계산용 — 기울기 불필요. fp32로 강제 (pow 연산이 fp16에서 불안정).

def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) RGB [0, 1] → (B, 3, H, W) CIELAB. D65 illuminant."""
    # sRGB → linear RGB
    mask = rgb > 0.04045
    linear = torch.where(mask, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)

    # linear RGB → XYZ (D65)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], device=rgb.device, dtype=rgb.dtype)
    xyz = torch.einsum('ij,bjhw->bihw', M, linear)

    # XYZ → Lab (D65 white)
    white = torch.tensor([0.95047, 1.0, 1.08883],
                         device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    xyz_n = xyz / white
    delta = 6.0 / 29.0
    f = torch.where(
        xyz_n > delta ** 3,
        xyz_n.clamp(min=1e-8).pow(1.0 / 3.0),
        xyz_n / (3 * delta ** 2) + 4.0 / 29.0,
    )

    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], dim=1)


# ── Gaussian blur (separable approximation via 2D conv) ───────────────────

def _gaussian_kernel_2d(sigma: float, ksize: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    return (g1d.view(-1, 1) * g1d.view(1, -1)).view(1, 1, ksize, ksize)


def _gaussian_blur(x: torch.Tensor, sigma: float = 3.0, ksize: int = 11) -> torch.Tensor:
    """(B, 1, H, W) → blurred."""
    k = _gaussian_kernel_2d(sigma, ksize, x.device, x.dtype)
    pad = ksize // 2
    return F.conv2d(F.pad(x, [pad, pad, pad, pad], mode='reflect'), k)


# ── Confusion weight (spatial loss weighting) ─────────────────────────────

@torch.no_grad()
def compute_confusion_weight(
    orig: torch.Tensor,
    sim_orig: torch.Tensor,
    low: float = 5.0,
    high: float = 25.0,
    sigma: float = 3.0,
    ksize: int = 11,
) -> torch.Tensor:
    """
    Spatial confusion weight w ∈ [0, 1] from Lab delta-E.
    High where CVD distorts color a lot; low elsewhere. No gradient.

    orig, sim_orig: (B, 3, H, W) RGB [0, 1]
    returns:        (B, 1, H, W) soft weight, same dtype as orig
    """
    # fp32 for Lab pow ops
    orig32 = orig.float()
    sim32 = sim_orig.float()
    lab_o = _rgb_to_lab(orig32)
    lab_s = _rgb_to_lab(sim32)
    delta_e = torch.sqrt(((lab_o - lab_s) ** 2).sum(dim=1, keepdim=True) + 1e-6)
    w = ((delta_e - low) / (high - low)).clamp(0.0, 1.0)
    w = _gaussian_blur(w, sigma=sigma, ksize=ksize).clamp(0.0, 1.0)
    return w.to(orig.dtype)


# ── 메인 손실 클래스 ──────────────────────────────────────────────────────

class CVDCorrectionLoss(nn.Module):
    """
    Spatial-weighted multi-objective loss.

    Args (β v1.1 defaults — v1.0 was 1.0/0.7/0.5/0.1, identity collapse):
        preserve_w:   α — 비혼동 영역 원본 보존        (default 0.5)
        visibility_w: β — 혼동 영역 보정              (default 1.5)
        structure_w:  η — 전체 구조 보존 (SSIM)         (default 0.3)
        color_w:      γ — 색 분포 보존 (mean/std)       (default 0.1)
    """

    def __init__(
        self,
        preserve_w:   float = 0.5,
        visibility_w: float = 1.5,
        structure_w:  float = 0.3,
        color_w:      float = 0.1,
    ):
        super().__init__()
        self.preserve_w = preserve_w
        self.visibility_w = visibility_w
        self.structure_w = structure_w
        self.color_w = color_w

    def forward(
        self,
        pred: torch.Tensor,
        orig: torch.Tensor,
        cvd_val: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        pred:    (B, 3, H, W) 모델 출력 [0, 1]
        orig:    (B, 3, H, W) 원본 RGB [0, 1]
        cvd_val: (B,)         CVD 타입 (0.0=p, 0.5=d, 1.0=t)
        """
        # CVD simulation (both orig and pred)
        sim_orig = simulate_cvd_batch(orig, cvd_val)
        sim_pred = simulate_cvd_batch(pred, cvd_val)

        # Spatial weight w (no grad)
        w = compute_confusion_weight(orig, sim_orig)        # (B, 1, H, W)

        # 1) Preserve: 비혼동 영역에서 원본 유지
        L_preserve = ((1.0 - w) * (pred - orig).abs()).mean()

        # 2) Visibility: 혼동 영역에서 CVD-사용자 시야가 원본과 가깝게
        L_visibility = (w * (sim_pred - orig).abs()).mean()

        # 3) Structure: 전체 구조 무너짐 방지 (SSIM)
        L_structure = 1.0 - ssim_fn(pred, orig, data_range=1.0)

        # 4) Color moment: 전체 색 분포 보존 (per-image, per-channel mean+std)
        mean_pred = pred.mean(dim=[2, 3])
        mean_orig = orig.mean(dim=[2, 3])
        std_pred = pred.std(dim=[2, 3])
        std_orig = orig.std(dim=[2, 3])
        L_color = (mean_pred - mean_orig).abs().mean() \
                + (std_pred - std_orig).abs().mean()

        total = (
            self.preserve_w   * L_preserve
            + self.visibility_w * L_visibility
            + self.structure_w  * L_structure
            + self.color_w      * L_color
        )

        components = {
            "loss_preserve":   L_preserve.item(),
            "loss_visibility": L_visibility.item(),
            "loss_structure":  L_structure.item(),
            "loss_color":      L_color.item(),
            "w_mean":          w.mean().item(),
        }
        return total, components


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    pred = torch.rand(B, 3, H, W, requires_grad=True)
    orig = torch.rand(B, 3, H, W)
    cvd_val = torch.tensor([0.0, 0.5])

    criterion = CVDCorrectionLoss()
    loss, comps = criterion(pred, orig, cvd_val)
    print(f"total: {loss.item():.4f}")
    for k, v in comps.items():
        print(f"  {k}: {v:.4f}")
    loss.backward()
    print(f"grad finite: {torch.isfinite(pred.grad).all().item()}")
    print("backward OK")
