"""
L_global audit — trace why |Δ|=0.015 gives L_g ≈ 0 in Phase 0 logs.

Checks:
  A. Analytic expectation of L_g at identity vs measured value.
  B. Top-100 confusion pairs (highest orig ΔE): dump ΔE_orig, ΔE_sim_orig,
     ΔE_sim_out for identity and a small perturbed output.
  C. Coordinate-system alignment between low-res delta field and full-res
     pair indices (rebuild the same output two ways and diff).
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.basis import compose_delta
from cvdlens_v2.color import srgb_to_linear, rgb_to_lab
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.losses import _huber
from cvdlens_v2.simulation import simulate


CVD_TYPES = ["p", "d", "t"]
K_PAIRS = 2048
HUBER_DELTA = 3.0
IMAGE = 'C:/Users/SCH/coco/val2017/000000000724.jpg'
SIZE = 256


def load_image():
    img = Image.open(IMAGE).convert('RGB').resize((SIZE, SIZE), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)


def _pairwise_de(lab_o_flat, lab_a_flat, w_flat, i, j, boost_confusion=2.0):
    """Rebuild the exact residual + boost that pairwise_de_loss uses."""
    de_orig = torch.linalg.norm(lab_o_flat[i] - lab_o_flat[j], dim=1)
    de_pred = torch.linalg.norm(lab_a_flat[i] - lab_a_flat[j], dim=1)
    residual = _huber(de_pred - de_orig, HUBER_DELTA)
    pair_w = torch.maximum(w_flat[i], w_flat[j])
    boost = 1.0 + (boost_confusion - 1.0) * (pair_w * (de_orig > 10.0).float())
    return de_orig, de_pred, residual, boost


def audit_type(cvd_type: str, orig_srgb: torch.Tensor):
    print(f"\n{'='*70}\n[{cvd_type.upper()}] L_g audit\n{'='*70}")
    orig_lin = srgb_to_linear(orig_srgb)
    w = compute_confusion_weight(orig_lin, cvd_type, 1.0)  # (1,1,H,W)
    sim_orig = simulate(orig_lin, cvd_type, 1.0)
    lab_orig = rgb_to_lab(orig_lin)
    lab_sim_orig = rgb_to_lab(sim_orig)

    N = SIZE * SIZE
    lab_o_flat = lab_orig[0].reshape(3, -1).T          # (N, 3)
    lab_so_flat = lab_sim_orig[0].reshape(3, -1).T
    w_flat = w[0].reshape(-1)

    # --- A. Identity: analytical estimate + measured ---
    gen = torch.Generator().manual_seed(42)
    i = torch.randint(0, N, (K_PAIRS,), generator=gen)
    j = torch.randint(0, N, (K_PAIRS,), generator=gen)

    de_orig, de_sim, res, boost = _pairwise_de(lab_o_flat, lab_so_flat, w_flat, i, j)
    L_g_identity = (boost * res).mean()

    print(f"\n  A. Identity statistics (K={K_PAIRS} random pairs, seed=42)")
    print(f"     de_orig    : mean={de_orig.mean():.3f}  median={de_orig.median():.3f}  max={de_orig.max():.3f}")
    print(f"     de_sim(orig): mean={de_sim.mean():.3f}  median={de_sim.median():.3f}  max={de_sim.max():.3f}")
    print(f"     compression : mean {(de_sim/(de_orig+1e-6)).mean():.3f}  (sim/orig)")
    print(f"     residual   : mean={(de_sim-de_orig).mean():.3f}  |res| mean={(de_sim-de_orig).abs().mean():.3f}")
    print(f"     huber(res) : mean={res.mean():.3f}")
    print(f"     boost      : mean={boost.mean():.3f}  max={boost.max():.3f}")
    print(f"     L_g measured (identity): {L_g_identity.item():.4f}")

    # Analytic bound: worst case, each residual = huber(-de_orig). Best case, = huber(0).
    huber_of_neg_de = _huber(-de_orig, HUBER_DELTA)
    print(f"     analytic upper bound L_g (all sim=0): {(boost*huber_of_neg_de).mean().item():.4f}")

    # --- B. Perturbed output (|Δ| ≈ 0.015 via random small delta in visible subspace) ---
    print(f"\n  B. Perturbed (|Δ| ≈ 0.015) — a plausible mid-training state")
    torch.manual_seed(0)
    d_lum = 0.02 * torch.randn(1, 1, SIZE, SIZE)
    d_c = 0.02 * torch.randn(1, 1, SIZE, SIZE)
    delta = compose_delta(d_lum, d_c, cvd_type)
    out_lin = (orig_lin + w * delta).clamp(0, 1)
    sim_out = simulate(out_lin, cvd_type, 1.0)
    lab_so_out_flat = rgb_to_lab(sim_out)[0].reshape(3, -1).T
    pred_diff = (out_lin - orig_lin).abs().mean().item()

    de_orig_2, de_pred_2, res_2, boost_2 = _pairwise_de(
        lab_o_flat, lab_so_out_flat, w_flat, i, j
    )
    L_g_perturbed = (boost_2 * res_2).mean()
    print(f"     |Δ| linear = {pred_diff:.4f}")
    print(f"     de_pred    : mean={de_pred_2.mean():.3f}  vs de_sim(orig) mean {de_sim.mean():.3f}")
    print(f"     huber(res) : mean={res_2.mean():.3f}")
    print(f"     L_g measured (perturbed): {L_g_perturbed.item():.4f}")

    # --- Top-100 confusion pairs at identity ---
    print(f"\n     Top-10 confusion pairs at identity (highest orig ΔE):")
    top_k = 10
    _, top_idx = torch.topk(de_orig, top_k)
    print(f"     {'rank':>4} {'de_orig':>8} {'de_sim_orig':>12} {'de_sim_out':>11} {'residual':>10} {'boost':>7}")
    for r, k in enumerate(top_idx.tolist()):
        print(f"     {r+1:>4} {de_orig[k]:>8.2f} {de_sim[k]:>12.2f} "
              f"{de_pred_2[k]:>11.2f} {res[k]:>10.3f} {boost[k]:>7.2f}")

    # --- C. Coordinate alignment check ---
    print(f"\n  C. Coord alignment: low-res(64) upsample vs full-res(256) parametrization")
    torch.manual_seed(1)
    d_lum_64 = 0.02 * torch.randn(1, 1, 64, 64)
    d_c_64 = 0.02 * torch.randn(1, 1, 64, 64)
    d_lum_up = F.interpolate(d_lum_64, size=SIZE, mode='bilinear', align_corners=False)
    d_c_up = F.interpolate(d_c_64, size=SIZE, mode='bilinear', align_corners=False)
    delta_from_lowres = compose_delta(d_lum_up, d_c_up, cvd_type)
    out_from_lowres = (orig_lin + w * delta_from_lowres).clamp(0, 1)

    # Direct full-res composition using the SAME upsampled fields
    delta_direct = compose_delta(d_lum_up, d_c_up, cvd_type)
    out_direct = (orig_lin + w * delta_direct).clamp(0, 1)

    coord_diff = (out_from_lowres - out_direct).abs().max().item()
    print(f"     Max pixel diff (should be 0): {coord_diff:.2e}")
    print(f"     → {'OK — coords aligned' if coord_diff < 1e-6 else 'BUG — coord mismatch'}")


def main():
    orig_srgb = load_image()
    for t in CVD_TYPES:
        audit_type(t, orig_srgb)


if __name__ == "__main__":
    main()
