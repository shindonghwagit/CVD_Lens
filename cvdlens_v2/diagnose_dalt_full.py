"""
Three-part daltonize diagnosis:
    1. Visual side-by-side (final judge): sim(orig) vs sim(daltonize) — does
       the STOP sign visually stand out more after daltonize?
    2. Clipping stats: fraction of daltonize output clamped to [0, 1], per
       channel, restricted to confusion regions. If >10-20%, that's why
       C(sim(dalt)) collapsed.
    3. Implementation sanity: daltonize must run on LINEAR RGB (not sRGB).
       Also print err-shift matrix signs and unit trace.

Then, if clipping is heavy, re-measure C(sim(dalt)) using a soft-compression
variant (tanh gamut mapping) to see if the loss verdict changes.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.color import srgb_to_linear, linear_to_srgb, rgb_to_lab
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.losses import _contrast_magnitude
from cvdlens_v2.simulation import simulate, daltonize, _ERR2MOD


IMAGE = 'C:/Users/SCH/coco/val2017/000000000724.jpg'
SIZE = 256
OUT_DIR = Path('C:/Users/SCH/graduation_project/outputs/v2_phase0/dalt_diag')


def load_image_srgb() -> torch.Tensor:
    img = Image.open(IMAGE).convert('RGB').resize((SIZE, SIZE), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def _soft_clamp_gamut(rgb: torch.Tensor, k: float = 8.0) -> torch.Tensor:
    """
    Smooth gamut mapping: instead of hard clamp to [0,1], compress smoothly
    so out-of-gamut colors preserve *relative* channel differences.
    Uses per-pixel scale = min(1, 1/max_channel_abs). This preserves local
    contrast much better than hard clamp when channels overshoot.
    """
    # If max channel per pixel > 1, scale whole pixel down. Below 1 → identity.
    max_c = rgb.amax(dim=1, keepdim=True).clamp(min=1e-6)   # (B, 1, H, W)
    scale = 1.0 / max_c.clamp(min=1.0)
    rgb = rgb * scale
    # Now no channel > 1. Handle negatives: soft-plus toward 0.
    return torch.where(rgb < 0, rgb * torch.exp(rgb * k), rgb)


def daltonize_soft(rgb_linear: torch.Tensor, cvd_type: str,
                   severity: float = 1.0, method: str = "machado") -> torch.Tensor:
    """Same as daltonize() but soft-gamut-mapped instead of hard-clamped."""
    sim = simulate(rgb_linear, cvd_type, severity, method)
    err = rgb_linear - sim
    shift_mat = torch.from_numpy(_ERR2MOD).to(rgb_linear.device, rgb_linear.dtype)
    err_shifted = torch.einsum('ij,bjhw->bihw', shift_mat, err)
    return _soft_clamp_gamut(rgb_linear + err_shifted)


def diagnose(cvd_type: str, orig_srgb: torch.Tensor):
    print(f"\n{'='*72}\n[{cvd_type.upper()}] Full daltonize diagnosis\n{'='*72}")

    # --- 3. Implementation sanity ---
    print("\n  Part 3 — Implementation sanity")
    orig_lin = srgb_to_linear(orig_srgb)
    print(f"    Input to daltonize is LINEAR (max diff from sRGB: "
          f"{(orig_lin - orig_srgb).abs().max().item():.4f}, must be nonzero)")
    print(f"    _ERR2MOD matrix:\n{_ERR2MOD}")
    print(f"    Row sums (should be R:0, G:1.7, B:1.7 for red-error diffusion): "
          f"{_ERR2MOD.sum(axis=1)}")

    # --- 2. Clipping stats ---
    print("\n  Part 2 — Clipping stats (before clamp)")
    w = compute_confusion_weight(orig_lin, cvd_type, 1.0)
    sim = simulate(orig_lin, cvd_type, 1.0)
    err = orig_lin - sim
    shift = torch.from_numpy(_ERR2MOD).to(orig_lin.dtype)
    err_shifted = torch.einsum('ij,bjhw->bihw', shift, err)
    dalt_pre_clamp = orig_lin + err_shifted             # (B, 3, H, W), unclamped

    conf_mask = (w > 0.5)                                # (B, 1, H, W)
    n_conf = int(conf_mask.sum().item())
    ch_names = ["R", "G", "B"]
    print(f"    Confusion pixels: {n_conf} / {SIZE*SIZE}")
    for ch, name in enumerate(ch_names):
        vals = dalt_pre_clamp[0, ch]
        conf = conf_mask[0, 0]
        n_over = int(((vals > 1.0) & conf).sum().item())
        n_under = int(((vals < 0.0) & conf).sum().item())
        n_over_all = int((vals > 1.0).sum().item())
        n_under_all = int((vals < 0.0).sum().item())
        max_v = vals.max().item()
        min_v = vals.min().item()
        print(f"    {name}: max={max_v:.3f}  min={min_v:.3f}  "
              f"clipped_high(all)={n_over_all}  clipped_low(all)={n_under_all}  "
              f"in-confusion clipped_high={n_over} ({n_over/n_conf*100:.1f}%)  "
              f"clipped_low={n_under} ({n_under/n_conf*100:.1f}%)")

    # --- Compare hard clamp vs soft gamut mapping ---
    print("\n    C(sim(dalt)) comparison — hard clamp vs soft gamut mapping:")
    dalt_hard = daltonize(orig_lin, cvd_type, 1.0)             # hard clamp
    dalt_soft = daltonize_soft(orig_lin, cvd_type, 1.0)        # soft
    sim_dalt_hard = simulate(dalt_hard, cvd_type, 1.0)
    sim_dalt_soft = simulate(dalt_soft, cvd_type, 1.0)
    lab_orig = rgb_to_lab(orig_lin)
    lab_sim_orig = rgb_to_lab(sim)
    lab_sim_hard = rgb_to_lab(sim_dalt_hard)
    lab_sim_soft = rgb_to_lab(sim_dalt_soft)
    C_o = _contrast_magnitude(lab_orig)
    C_so = _contrast_magnitude(lab_sim_orig)
    C_sh = _contrast_magnitude(lab_sim_hard)
    C_ss = _contrast_magnitude(lab_sim_soft)
    conf_f = conf_mask.float()
    for name, C in [("C(sim(orig))", C_so), ("C(sim(dalt_hard))", C_sh),
                    ("C(sim(dalt_soft))", C_ss)]:
        avg = (C * conf_f).sum().item() / (conf_f.sum().item() + 1e-8)
        deficit = (torch.relu(C_o - C) * conf_f).mean().item()
        print(f"    {name:20s}  in-conf avg = {avg:.3f}  "
              f"w-weighted deficit = {deficit:.4f}")

    # --- 1. Visual side-by-side (the final judge) ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    axes[0, 0].imshow(_to_np(orig_srgb));                        axes[0, 0].set_title('Original')
    axes[0, 1].imshow(_to_np(linear_to_srgb(sim)));              axes[0, 1].set_title(f'sim(orig) — CVD view')
    axes[0, 2].imshow(w[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Confusion w')

    axes[1, 0].imshow(_to_np(linear_to_srgb(dalt_hard)));         axes[1, 0].set_title('daltonize (hard clamp)')
    axes[1, 1].imshow(_to_np(linear_to_srgb(sim_dalt_hard)));     axes[1, 1].set_title('sim(dalt_hard) — CVD view')
    axes[1, 2].imshow(_to_np(linear_to_srgb(sim_dalt_soft)));     axes[1, 2].set_title('sim(dalt_soft) — CVD view')
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{cvd_type.upper()} — visual side-by-side: does daltonize enhance the STOP sign?',
                 fontsize=13, fontweight='bold')
    out_path = OUT_DIR / f'dalt_visual_{cvd_type}.png'
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved visual: {out_path}")


def main():
    orig_srgb = load_image_srgb()
    for t in ["p", "d", "t"]:
        diagnose(t, orig_srgb)


if __name__ == "__main__":
    main()
