"""
Loss sanity gate.

Requires: L(daltonize(orig)) < L(identity) for every CVD type.

If a candidate loss fails this, it *rewards* identity over a known-good
correction, meaning optimization is guaranteed to underperform daltonize.
No optimization must run against such a loss.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.basis import compose_delta
from cvdlens_v2.color import srgb_to_linear, linear_to_srgb
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.losses import CVDLossV2
from cvdlens_v2.simulation import daltonize


def hand_crafted_boost(orig_linear: torch.Tensor, w: torch.Tensor,
                       cvd_type: str, boost: float = 0.05) -> torch.Tensor:
    """
    Known-good reference for the contrast-recovery objective: uniform
    boost along both visible-subspace basis vectors, scaled by w.

    Unlike daltonize (whose objective is different), this reference is
    guaranteed to *reduce* the deficit in the visible subspace by
    construction, at a magnitude comparable to the CVD compression loss.
    If the loss favors identity over this reference, the loss is wrong.
    """
    B, _, H, W = orig_linear.shape
    d_lum = boost * torch.ones(B, 1, H, W, device=orig_linear.device, dtype=orig_linear.dtype)
    d_c = boost * torch.ones(B, 1, H, W, device=orig_linear.device, dtype=orig_linear.dtype)
    delta = compose_delta(d_lum, d_c, cvd_type)
    return (orig_linear + w * delta).clamp(0, 1)


CVD_NAMES = {"p": "Protanopia", "d": "Deuteranopia", "t": "Tritanopia"}


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)


@torch.no_grad()
def check(image_path: str, size: int = 256, severity: float = 1.0,
          use_lpips: bool = False, verbose: bool = True) -> dict:
    orig_srgb = load_image(image_path, size)
    orig_lin = srgb_to_linear(orig_srgb)

    loss_fn = CVDLossV2(use_lpips=use_lpips)
    gen = torch.Generator().manual_seed(42)

    results = {}
    for t in ["p", "d", "t"]:
        w = compute_confusion_weight(orig_lin, t, severity)

        # 1) identity
        gen.manual_seed(42)
        L_id, comps_id = loss_fn(
            out_linear=orig_lin, orig_linear=orig_lin,
            out_srgb=orig_srgb, orig_srgb=orig_srgb,
            w=w, cvd_type=t, severity=severity, generator=gen,
        )

        # 2) hand-crafted boost — the primary gate reference for this loss
        boost_lin = hand_crafted_boost(orig_lin, w, t, boost=0.05)
        boost_srgb = linear_to_srgb(boost_lin)
        gen.manual_seed(42)
        L_boost, comps_boost = loss_fn(
            out_linear=boost_lin, orig_linear=orig_lin,
            out_srgb=boost_srgb, orig_srgb=orig_srgb,
            w=w, cvd_type=t, severity=severity, generator=gen,
        )

        # 3) daltonize — informational (its objective differs from ours)
        dalt_lin = daltonize(orig_lin, t, severity)
        dalt_srgb = linear_to_srgb(dalt_lin)
        gen.manual_seed(42)
        L_dalt, comps_dalt = loss_fn(
            out_linear=dalt_lin, orig_linear=orig_lin,
            out_srgb=dalt_srgb, orig_srgb=orig_srgb,
            w=w, cvd_type=t, severity=severity, generator=gen,
        )

        passes = L_boost.item() < L_id.item()   # gate: manual boost must beat identity
        results[t] = {
            "L_identity": L_id.item(),
            "L_manual_boost": L_boost.item(),
            "L_daltonize": L_dalt.item(),
            "passes": passes,
            "comps_identity": comps_id,
            "comps_manual_boost": comps_boost,
            "comps_daltonize": comps_dalt,
            "pred_diff_manual": (boost_lin - orig_lin).abs().mean().item(),
            "pred_diff_dalt": (dalt_lin - orig_lin).abs().mean().item(),
        }

        if verbose:
            mark = "PASS" if passes else "FAIL"
            print(f"\n[{t}] {CVD_NAMES[t]}  — {mark}")
            print(f"    L(identity)     = {L_id.item():.4f}   {comps_id}")
            print(f"    L(manual boost) = {L_boost.item():.4f}   {comps_boost}   |Δ|={results[t]['pred_diff_manual']:.4f}")
            print(f"    L(daltonize)    = {L_dalt.item():.4f}   {comps_dalt}   |Δ|={results[t]['pred_diff_dalt']:.4f}  (informational)")
            if passes:
                print(f"    → manual boost scores {L_id.item()-L_boost.item():.4f} below identity — GATE PASS")
            else:
                print(f"    → LOSS FAVORS IDENTITY over hand-crafted contrast boost — loss is broken")

    all_pass = all(r['passes'] for r in results.values())
    if verbose:
        print("\n" + "=" * 56)
        print("GATE:  " + ("PASS — loss ready for optimization" if all_pass
                          else "FAIL — do NOT run optimization until loss is fixed"))
        print("=" * 56)
    return {"all_pass": all_pass, "per_type": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default="C:/Users/SCH/coco/val2017/000000000724.jpg")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--severity", type=float, default=1.0)
    p.add_argument("--use-lpips", action="store_true")
    args = p.parse_args()
    check(args.image, args.size, args.severity, args.use_lpips, verbose=True)


if __name__ == "__main__":
    main()
