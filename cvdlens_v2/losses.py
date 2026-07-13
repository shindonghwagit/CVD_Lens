"""
CVDLens v2 loss architecture.

    L = λc·L_contrast + λg·L_global + λn·L_natural

- L_contrast: multi-scale Lab gradient matching (main). The CVD-viewed
  output should carry the same local contrast structure as the normal
  view of the original.
- L_global: pairwise ΔE preservation on random pixel pairs. Non-local
  color relationships preserved under CVD view.
- L_natural: LPIPS(out, orig) + L1(sim(out), sim(orig)). Natural under
  both normal and CVD views. Small weight — never dominant.

Explicitly ABSENT (removed by design):
- |sim(out) - orig|: identity is optimal → removed
- SSIM(out, orig): identity is optimal → removed
- mean/std color moment matching: identity is optimal → removed
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import rgb_to_lab, delta_e_lab, srgb_to_linear
from .simulation import simulate


# ── Multi-scale gradient matching (L_contrast) ───────────────────────────

def _grad_xy(x: torch.Tensor) -> tuple:
    """Central-difference gradients along H, W. Returns (dy, dx) each (B,C,H,W)."""
    dy = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 0, 1))
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    return dy, dx


def _huber(x: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    abs_x = x.abs()
    quad = torch.minimum(abs_x, torch.tensor(delta, device=x.device, dtype=x.dtype))
    lin = abs_x - quad
    return 0.5 * quad ** 2 / delta + lin


def _downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return x
    return F.avg_pool2d(x, kernel_size=factor, stride=factor)


def _contrast_magnitude(lab: torch.Tensor) -> torch.Tensor:
    """
    C(x) = sqrt(sum_{ch, dir} grad^2). Total local contrast summed over
    all Lab channels and both spatial directions.
    Returns (B, 1, H, W).
    """
    dy, dx = _grad_xy(lab)               # each (B, 3, H, W)
    sq = dy ** 2 + dx ** 2               # (B, 3, H, W)
    return torch.sqrt(sq.sum(dim=1, keepdim=True) + 1e-6)


def contrast_loss(
    lab_orig: torch.Tensor,
    lab_sim_out: torch.Tensor,
    w: torch.Tensor,
    scales: tuple = (1, 2, 4, 8),
) -> torch.Tensor:
    """
    Multi-scale ONE-SIDED contrast magnitude matching.

        C(x) = sqrt( sum_{ch, dir} grad^2 )
        L_c  = mean( m * relu( C(orig) - C(sim(out)) )^2 )

    Why not per-channel gradient matching:
        Per-channel matching penalizes the *very* re-encoding we want,
        e.g. a lost a*-channel red/green edge encoded via b* (blue/yellow).
        Only the total magnitude across channels matters for the CVD viewer.

    Why one-sided (relu):
        We only care that CVD-view contrast is not BELOW the normal-view
        contrast. Over-recovery is naturally bounded by L_natural + physical
        clamp, and penalizing it here would push toward identity.
    """
    total = 0.0
    for s in scales:
        lo = _downsample(lab_orig, s)
        la = _downsample(lab_sim_out, s)
        m = _downsample(w, s)
        C_o = _contrast_magnitude(lo)
        C_a = _contrast_magnitude(la)
        deficit = torch.relu(C_o - C_a)                    # (B, 1, H, W)
        total = total + (m * deficit.pow(2)).mean()
    return total / len(scales)


# ── Pairwise ΔE matching (L_global) ──────────────────────────────────────

def pairwise_de_loss(
    lab_orig: torch.Tensor,
    lab_sim_out: torch.Tensor,
    w: torch.Tensor,
    k_pairs: int = 2048,
    huber_delta: float = 3.0,
    confusion_pair_boost: float = 2.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample K random pixel pairs per image. For each pair (i, j),
    match ΔE_Lab(sim(out)_i, sim(out)_j) to ΔE_Lab(orig_i, orig_j).

    Weighted 2x for pairs where orig ΔE is high but sim(orig) ΔE is low
    (the classic confusion pairs).

    Args:
        generator: optional torch.Generator for reproducible pair sampling.
                   Pass a fixed-seed generator for monotonic-loss debugging
                   (Phase 0). Training uses None (fresh random each call).
    """
    B, _, H, W = lab_orig.shape
    device = lab_orig.device
    N = H * W

    total = 0.0
    for b in range(B):
        lab_o = lab_orig[b].reshape(3, -1).T          # (N, 3)
        lab_a = lab_sim_out[b].reshape(3, -1).T
        w_b = w[b].reshape(-1)                        # (N,)

        i = torch.randint(0, N, (k_pairs,), device=device, generator=generator)
        j = torch.randint(0, N, (k_pairs,), device=device, generator=generator)

        de_orig = torch.linalg.norm(lab_o[i] - lab_o[j], dim=1)      # (K,)
        de_pred = torch.linalg.norm(lab_a[i] - lab_a[j], dim=1)

        residual = _huber(de_pred - de_orig, huber_delta)             # (K,)

        # Boost confusion pairs: high orig ΔE, high w on at least one endpoint
        pair_w = torch.maximum(w_b[i], w_b[j])                        # (K,)
        boost = 1.0 + (confusion_pair_boost - 1.0) * (
            pair_w * (de_orig > 10.0).float()
        )
        total = total + (boost * residual).mean()

    return total / B


# ── Naturalness (L_natural) ──────────────────────────────────────────────

class NaturalnessLoss(nn.Module):
    """
    LPIPS(out, orig) + 0.1 * L1(sim(out), sim(orig)).

    LPIPS is loss-only (VGG-based). Never exported to ONNX.
    """

    def __init__(self, sim_l1_w: float = 0.1, use_lpips: bool = True):
        super().__init__()
        self.sim_l1_w = sim_l1_w
        self.lpips = None
        if use_lpips:
            try:
                import lpips
                self.lpips = lpips.LPIPS(net='vgg', verbose=False)
                for p in self.lpips.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"[naturalness] LPIPS unavailable ({e}); falling back to L1.")
                self.lpips = None

    def forward(
        self,
        out_srgb: torch.Tensor,      # (B,3,H,W) sRGB [0,1]
        orig_srgb: torch.Tensor,     # (B,3,H,W) sRGB [0,1]
        sim_out_lin: torch.Tensor,   # (B,3,H,W) linear RGB [0,1]
        sim_orig_lin: torch.Tensor,  # (B,3,H,W) linear RGB [0,1]
    ) -> torch.Tensor:
        if self.lpips is not None:
            # LPIPS expects [-1, 1]
            l_nat = self.lpips(out_srgb * 2 - 1, orig_srgb * 2 - 1).mean()
        else:
            l_nat = (out_srgb - orig_srgb).abs().mean()

        l_sim_l1 = (sim_out_lin - sim_orig_lin).abs().mean()
        return l_nat + self.sim_l1_w * l_sim_l1


# ── Composite ────────────────────────────────────────────────────────────

class CVDLossV2(nn.Module):
    def __init__(
        self,
        lambda_c: float = 1.0,
        lambda_g: float = 0.15,
        lambda_n: float = 0.15,
        use_lpips: bool = True,
        k_pairs: int = 2048,
    ):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g
        self.lambda_n = lambda_n
        self.k_pairs = k_pairs
        self.naturalness = NaturalnessLoss(use_lpips=use_lpips)

    def forward(
        self,
        out_linear: torch.Tensor,     # (B,3,H,W) linear RGB [0,1]
        orig_linear: torch.Tensor,    # (B,3,H,W) linear RGB [0,1]
        out_srgb: torch.Tensor,       # (B,3,H,W) sRGB [0,1] (for LPIPS)
        orig_srgb: torch.Tensor,      # (B,3,H,W) sRGB [0,1]
        w: torch.Tensor,              # (B,1,H,W)
        cvd_type: str,
        severity: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        sim_orig = simulate(orig_linear, cvd_type, severity)
        sim_out = simulate(out_linear, cvd_type, severity)

        lab_orig = rgb_to_lab(orig_linear.float())
        lab_sim_out = rgb_to_lab(sim_out.float())

        L_c = contrast_loss(lab_orig, lab_sim_out, w)
        L_g = pairwise_de_loss(lab_orig, lab_sim_out, w, k_pairs=self.k_pairs,
                               generator=generator)
        L_n = self.naturalness(out_srgb, orig_srgb, sim_out, sim_orig)

        total = self.lambda_c * L_c + self.lambda_g * L_g + self.lambda_n * L_n

        return total, {
            "L_contrast": L_c.item(),
            "L_global": L_g.item(),
            "L_natural": L_n.item(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 1, 128, 128
    orig_srgb = torch.rand(B, 3, H, W)
    orig_lin = srgb_to_linear(orig_srgb)
    # Fake "output" = original for identity check
    out_srgb = orig_srgb.clone()
    out_lin = orig_lin.clone()
    from cvdlens_v2.confusion import compute_confusion_weight
    w = compute_confusion_weight(orig_lin, "p")

    loss = CVDLossV2(use_lpips=False)
    total, comps = loss(out_lin, orig_lin, out_srgb, orig_srgb, w, "p")
    print(f"[IDENTITY] total={total.item():.4f}")
    for k, v in comps.items():
        print(f"  {k}: {v:.4f}")

    # Now try a slight noise output — should differ
    out_srgb_noisy = (orig_srgb + 0.05 * torch.randn_like(orig_srgb)).clamp(0, 1)
    out_lin_noisy = srgb_to_linear(out_srgb_noisy)
    total_n, comps_n = loss(out_lin_noisy, orig_lin, out_srgb_noisy, orig_srgb, w, "p")
    print(f"[NOISY]    total={total_n.item():.4f}")
    for k, v in comps_n.items():
        print(f"  {k}: {v:.4f}")
