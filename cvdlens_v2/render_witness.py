"""
Render 5-panel witness PNGs (orig / sim(orig) / w / out / sim(out))
from a trained CVDCorrectionNet checkpoint. Used for visual speckle
judgment when SI metric is suspected to be an artifact.

Usage:
    py -m cvdlens_v2.render_witness \\
        --ckpt C:/Users/SCH/Downloads/model_latest.zip \\
        --stems 000000000724,000000000632,000000001584 \\
        --out-dir outputs/v2_phase1/witness
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.model import CVDCorrectionNet
from cvdlens_v2.color import srgb_to_linear, linear_to_srgb
from cvdlens_v2.simulation import simulate

CVD_NAMES = {"p": "Protanopia", "d": "Deuteranopia", "t": "Tritanopia"}


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t[0].detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def load_ckpt(path: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = CVDCorrectionNet(pretrained_backbone=False).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck.get("step", "?")


def load_image(path: str, size: int = 256, device="cpu"):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


@torch.no_grad()
def render(net, orig_srgb, cvd_type, step_tag, out_path):
    r = net(orig_srgb, cvd_type=cvd_type, severity=1.0)
    orig_lin = srgb_to_linear(orig_srgb)
    sim_orig = simulate(orig_lin, cvd_type, 1.0)
    sim_out = simulate(r["out_linear"], cvd_type, 1.0)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), facecolor="white")

    axes[0].imshow(_to_np(orig_srgb))
    axes[0].set_title("Original", fontweight="bold")

    axes[1].imshow(_to_np(linear_to_srgb(sim_orig)))
    axes[1].set_title(f"CVD Sim (input) — {CVD_NAMES[cvd_type]}",
                      fontweight="bold")

    w = r["w"][0, 0].cpu().numpy()
    axes[2].imshow(w, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title(f"Confusion w  (mean={w.mean():.3f})",
                      fontweight="bold")

    axes[3].imshow(_to_np(r["out_srgb"]))
    delta_mag = (r["out_srgb"] - orig_srgb).abs().mean().item()
    axes[3].set_title(f"Optimized Output  |Δ|={delta_mag:.4f}",
                      fontweight="bold")

    axes[4].imshow(_to_np(linear_to_srgb(sim_out)))
    axes[4].set_title("CVD Sim (output)", fontweight="bold")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Phase 1 — step {step_tag} — {CVD_NAMES[cvd_type]}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved {out_path}  |Δ|={delta_mag:.4f}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    net, step = load_ckpt(args.ckpt, device)
    print(f"[ckpt] step={step}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in [s.strip() for s in args.stems.split(",") if s.strip()]:
        img_path = f"{args.val_dir}/{stem}.jpg"
        print(f"\n[{stem}]")
        orig = load_image(img_path, size=args.size, device=device)
        for t in ["p", "d", "t"]:
            out_path = out_dir / f"witness_{stem}_{t}_step{step}.png"
            render(net, orig, t, step, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-dir", default="C:/Users/SCH/coco/val2017")
    p.add_argument("--stems", default="000000000724,000000000632,000000001584")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--out-dir",
                   default="C:/Users/SCH/graduation_project/outputs/v2_phase1/witness")
    main(p.parse_args())
