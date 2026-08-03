"""
Phase 2 Step 3 — evaluation metrics (two axes + Phase 1 secondary logs).

Axis 1 — CRR (contrast recovery):
    ratio_w = confusion-weighted contrast ratio of sim(method(img)) vs sim(img).
    Reuses the Phase 1 definition verbatim (validate_loss._cvd_contrast_ratio_w).
    > 1 means contrast in confusion regions grew on the CVD view.

Axis 2 — NP (naturalness / original damage):
    a. |Δ| = mean |method(img) − img|  in sRGB.
    b. LPIPS(method(img), img)  — perceptual distance, VGG backbone (same net
       the training naturalness loss used).

Secondary (Phase 1 logging set, diagnostic only):
    SI, SI_uniform, corr_guide via train._si_decomposition, computed on
    delta = method_out_linear − orig_linear for BOTH methods so CVDLens and
    daltonize are measured on identical terms.

All metrics take the SAME confusion weight w (a function of the original only)
and the SAME simulator (machado, severity 1.0), so the two methods are
apples-to-apples.
"""
from __future__ import annotations
import warnings

import torch

from cvdlens_v2.simulation import simulate
from cvdlens_v2.validate_loss import _cvd_contrast_ratio_w
from cvdlens_v2.train import _si_decomposition


@torch.no_grad()
def crr_ratio_w(out_linear, orig_linear, w, cvd_type: str, severity: float = 1.0) -> float:
    """Axis 1 — confusion-weighted contrast recovery on the CVD view."""
    sim_out = simulate(out_linear, cvd_type, severity)
    sim_orig = simulate(orig_linear, cvd_type, severity)
    return _cvd_contrast_ratio_w(sim_out, sim_orig, w, blur_sigma=1.0)


@torch.no_grad()
def np_delta(out_srgb, orig_srgb) -> float:
    """Axis 2a — mean absolute sRGB change (original-damage magnitude)."""
    return (out_srgb - orig_srgb).abs().mean().item()


_LPIPS = None


def get_lpips():
    """Lazy singleton VGG-LPIPS (weights download + load once)."""
    global _LPIPS
    if _LPIPS is None:
        warnings.filterwarnings("ignore")
        import lpips
        m = lpips.LPIPS(net="vgg", verbose=False)
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        _LPIPS = m
    return _LPIPS


@torch.no_grad()
def np_lpips(out_srgb, orig_srgb) -> float:
    """Axis 2b — LPIPS perceptual distance. Inputs [0,1] → lpips wants [-1,1]."""
    m = get_lpips()
    return m(out_srgb * 2 - 1, orig_srgb * 2 - 1).item()


@torch.no_grad()
def secondary_logs(out_linear, orig_linear, w) -> dict:
    """Phase 1 secondary logging set on delta = out_linear − orig_linear."""
    delta = out_linear - orig_linear
    d = _si_decomposition(delta, w, orig_linear)
    return {"SI": d["SI"], "SI_uniform": d["SI_uniform"], "corr_guide": d["corr_guide"]}
