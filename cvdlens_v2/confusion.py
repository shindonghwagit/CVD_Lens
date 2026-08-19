"""
Type-adaptive confusion weight w ∈ [0, 1].

Computed from Lab delta-E between original and CVD-simulated original,
with thresholds tuned per CVD type (P/D simulation produces smaller
delta-E than T, so lower thresholds for P/D).

Tritan (t) is computed with the BRETTEL simulator + retuned thresholds (12,30):
Machado-t under-moves real desaturated blue so its w collapses on sky/sea blue;
Brettel moves it properly (reports/tritan_w_diagnosis/). t-ONLY — p/d unchanged.

Fixed, differentiable, no learnable parameters.
"""

import torch
import torch.nn.functional as F

from .color import rgb_to_lab, delta_e_lab
from .simulation import simulate


_THRESHOLDS = {
    "p": (2.0, 12.0),
    "d": (2.0, 12.0),
    "t": (5.0, 25.0),      # legacy Machado-t thresholds (production t uses _T_BRETTEL)
}

# Tritan-only: Brettel simulator + retuned thresholds. See module docstring and
# reports/tritan_w_diagnosis/ (H1 confirmed: Machado-t under-moves real blue).
_T_BRETTEL = (12.0, 30.0)


def _gaussian_kernel_2d(sigma: float, ksize: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    return (g1d.view(-1, 1) * g1d.view(1, -1)).view(1, 1, ksize, ksize)


def _blur(x: torch.Tensor, sigma: float, ksize: int) -> torch.Tensor:
    k = _gaussian_kernel_2d(sigma, ksize, x.device, x.dtype)
    pad = ksize // 2
    return F.conv2d(F.pad(x, [pad] * 4, mode='reflect'), k)


@torch.no_grad()
def compute_confusion_weight(
    rgb_linear: torch.Tensor,
    cvd_type: str,
    severity: float = 1.0,
    method: str = "machado",
    sigma: float = 3.0,
    ksize: int = 11,
) -> torch.Tensor:
    """
    Spatial confusion weight.

    Args:
        rgb_linear: (B, 3, H, W) linear RGB [0, 1]
        cvd_type:   'p' | 'd' | 't'
    Returns:
        w: (B, 1, H, W) soft mask in [0, 1]
    """
    orig32 = rgb_linear.float()
    if cvd_type == "t":
        # tritan-only branch: Brettel simulator (severity ignored, fixed 1.0) +
        # retuned thresholds. Machado-t under-moves real desaturated blue.
        low, high = _T_BRETTEL
        sim32 = simulate(orig32, "t", 1.0, "brettel")
    else:
        # protan / deutan: unchanged Machado path (no code-path change).
        low, high = _THRESHOLDS[cvd_type]
        sim32 = simulate(orig32, cvd_type, severity, method)
    dE = delta_e_lab(rgb_to_lab(orig32), rgb_to_lab(sim32))         # (B, 1, H, W)
    w = ((dE - low) / (high - low)).clamp(0.0, 1.0)
    w = _blur(w, sigma, ksize).clamp(0.0, 1.0)
    return w.to(rgb_linear.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    rgb = torch.rand(2, 3, 128, 128)
    for t in ["p", "d", "t"]:
        w = compute_confusion_weight(rgb, t)
        print(f"[{t}] w_mean={w.mean():.4f}  w_max={w.max():.4f}  shape={tuple(w.shape)}")
