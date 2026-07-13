"""
Visual comparison of the two loss-landscape valleys uncovered by ray_scan:

    P/D: narrow color-axis valley at (dir=-b_C, β≈0.02) vs
         broad luminance-axis basin at (dir=-b_L, β≈0.10).

Saves one figure per type showing:
    row 1: orig | out(-b_C,0.02) | out(-b_L,0.10)
    row 2: sim(orig) | sim(out(-b_C,0.02)) | sim(out(-b_L,0.10))
    row 3: |out-orig| heatmaps

Purpose: check whether the sharp -b_C minimum represents a real perceptual
gain or is a metric hack that the CVD-viewer cannot see.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.basis import compose_delta
from cvdlens_v2.color import srgb_to_linear, linear_to_srgb
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.simulation import simulate


CVD_NAMES = {"p": "Protanopia", "d": "Deuteranopia", "t": "Tritanopia"}


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


@torch.no_grad()
def make_out(orig_lin, w, cvd_type, direction, beta):
    B, _, H, W = orig_lin.shape
    zeros = torch.zeros(B, 1, H, W, device=orig_lin.device, dtype=orig_lin.dtype)
    val = torch.full_like(zeros, beta)
    if direction == "-b_L":
        delta = compose_delta(-val, zeros, cvd_type)
    elif direction == "-b_C":
        delta = compose_delta(zeros, -val, cvd_type)
    else:
        raise ValueError(direction)
    return (orig_lin + w * delta).clamp(0, 1)


def visualize(cvd_type: str, orig_srgb, out_path: Path):
    orig_lin = srgb_to_linear(orig_srgb)
    w = compute_confusion_weight(orig_lin, cvd_type, 1.0)
    sim_orig = simulate(orig_lin, cvd_type, 1.0)

    out_bC = make_out(orig_lin, w, cvd_type, "-b_C", 0.02)
    out_bL = make_out(orig_lin, w, cvd_type, "-b_L", 0.10)
    sim_bC = simulate(out_bC, cvd_type, 1.0)
    sim_bL = simulate(out_bL, cvd_type, 1.0)

    d_bC = (out_bC - orig_lin).abs().mean(1, keepdim=True)
    d_bL = (out_bL - orig_lin).abs().mean(1, keepdim=True)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), facecolor='white')
    imgs = [
        (linear_to_srgb(orig_lin), 'Original', axes[0, 0]),
        (linear_to_srgb(out_bC),   f'out(-b_C, β=0.02)  (narrow valley)', axes[0, 1]),
        (linear_to_srgb(out_bL),   f'out(-b_L, β=0.10)  (broad basin)',   axes[0, 2]),
        (linear_to_srgb(sim_orig), f'sim(orig) — {CVD_NAMES[cvd_type]}',  axes[1, 0]),
        (linear_to_srgb(sim_bC),   'sim(out(-b_C,0.02))',                 axes[1, 1]),
        (linear_to_srgb(sim_bL),   'sim(out(-b_L,0.10))',                 axes[1, 2]),
    ]
    for tens, title, ax in imgs:
        ax.imshow(_to_np(tens))
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    # Row 3: absolute delta heatmaps + one metric card
    vmax_common = max(d_bC.max().item(), d_bL.max().item(), 1e-6)
    for tens, title, ax in [(d_bC, 'Δ(-b_C,0.02)', axes[2, 0]),
                             (d_bL, 'Δ(-b_L,0.10)', axes[2, 1])]:
        ax.imshow(tens[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=vmax_common)
        ax.set_title(title + f"  |Δ|_mean={tens.mean().item():.4f}",
                     fontsize=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
    axes[2, 2].axis('off')
    metric = (
        f"CVD type: {CVD_NAMES[cvd_type]}\n\n"
        f"-b_C @ β=0.02\n"
        f"  |Δ|linear = {(out_bC-orig_lin).abs().mean().item():.4f}\n"
        f"  |sim_out-sim_orig|_L1 = {(sim_bC-sim_orig).abs().mean().item():.4f}\n\n"
        f"-b_L @ β=0.10\n"
        f"  |Δ|linear = {(out_bL-orig_lin).abs().mean().item():.4f}\n"
        f"  |sim_out-sim_orig|_L1 = {(sim_bL-sim_orig).abs().mean().item():.4f}\n\n"
        f"If the CVD-view of the -b_C column looks\n"
        f"visually IDENTICAL to sim(orig), the deep\n"
        f"L_c minimum there is a metric hack.\n"
    )
    axes[2, 2].text(0.02, 0.98, metric, family='monospace', fontsize=10,
                    verticalalignment='top')

    fig.suptitle(f"Loss-landscape valleys — {CVD_NAMES[cvd_type]}",
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str,
                   default="C:/Users/SCH/coco/val2017/000000000724.jpg")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--types", type=str, default="p,d")
    p.add_argument("--out-dir", type=str,
                   default="C:/Users/SCH/graduation_project/outputs/ray_scan")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    orig_srgb = load_image(args.image, args.size)

    for t in [x.strip() for x in args.types.split(",") if x.strip()]:
        out_path = out_dir / f"narrow_valley_{t}.png"
        visualize(t, orig_srgb, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
