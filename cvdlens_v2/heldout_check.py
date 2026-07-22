"""
Held-out spot check (overfit defence).

Selects 8 COCO val2017 images NOT in the 10-image validation bank — 4 from the
top confusion-mass tier and 4 from the middle tier — then measures model_best
(step 9000) on each: per-type ratio_w, |Δ|, and confusion mass w̄. Reports the
type-averaged ratio_w and compares against the bank figures.

Confusion mass = mean over {p,d,t} of w.mean()  (type-agnostic ranking scalar,
same quantity annotated next to the bank stems in validate_multi.py).

Usage:
    py -m cvdlens_v2.heldout_check --ckpt outputs/phase3_kaggle/v2_phase1/model_best.pt
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import torch

from cvdlens_v2.model import CVDCorrectionNet, CVD_TYPES
from cvdlens_v2.color import srgb_to_linear
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.simulation import simulate
from cvdlens_v2.validate_loss import _cvd_contrast_ratio_w
from cvdlens_v2.validate_multi import IMAGES as BANK
from PIL import Image
from torchvision import transforms

BANK_RATIO_W = {"p": 1.394, "d": 1.363, "t": 1.367}   # step-9000 bank type-means
TOL = 0.10


def load_srgb(path, size, device):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


@torch.no_grad()
def confusion_mass(orig_lin) -> float:
    return sum(compute_confusion_weight(orig_lin, t).mean().item()
               for t in CVD_TYPES) / len(CVD_TYPES)


def main(a):
    device = "cpu"
    random.seed(a.seed)

    bank = set(BANK)
    all_jpg = sorted(Path(a.val_dir).glob("*.jpg"))
    pool = [p for p in all_jpg if p.stem not in bank]
    cand = random.sample(pool, min(a.pool, len(pool)))
    print(f"pool={len(pool)} candidates, sampling {len(cand)} for ranking (seed={a.seed})")

    # rank candidates by confusion mass
    ranked = []
    for i, p in enumerate(cand):
        orig = load_srgb(p, a.size, device)
        ranked.append((confusion_mass(srgb_to_linear(orig)), p.stem))
    ranked.sort(reverse=True)

    n = len(ranked)
    top4 = ranked[:4]
    mid_start = n // 2 - 2
    mid4 = ranked[mid_start:mid_start + 4]
    selected = [("top", cm, st) for cm, st in top4] + \
               [("mid", cm, st) for cm, st in mid4]

    print("\nSELECTED held-out stems (not in bank):")
    for tier, cm, st in selected:
        print(f"  [{tier}] {st}  w̄={cm:.3f}")

    # evaluate model_best
    net = CVDCorrectionNet(pretrained_backbone=False).to(device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ck["state_dict"]); net.eval()
    print(f"\n[ckpt] step={ck.get('step')}  evaluating {len(selected)} images...\n")

    rows = []
    by_type = {t: [] for t in CVD_TYPES}
    with torch.no_grad():
        for tier, cm, st in selected:
            orig = load_srgb(f"{a.val_dir}/{st}.jpg", a.size, device)
            orig_lin = srgb_to_linear(orig)
            for t in CVD_TYPES:
                r = net(orig, cvd_type=t, severity=1.0)
                dmag = (r["out_srgb"] - orig).abs().mean().item()
                rw = _cvd_contrast_ratio_w(
                    simulate(r["out_linear"], t, 1.0),
                    simulate(orig_lin, t, 1.0), r["w"], blur_sigma=1.0)
                by_type[t].append(rw)
                rows.append({"stem": st, "tier": tier, "type": t,
                             "w_mean": cm, "ratio_w": rw, "delta": dmag,
                             "is_do_nothing": dmag < 0.005})
                print(f"  {st} [{tier}] {t}: ratio_w={rw:.3f}  |Δ|={dmag:.4f}"
                      + ("  (do-nothing)" if dmag < 0.005 else ""))

    type_mean = {t: sum(v) / len(v) for t, v in by_type.items()}
    within = {t: abs(type_mean[t] - BANK_RATIO_W[t]) <= TOL for t in CVD_TYPES}
    consistent = all(within.values())

    print("\n── held-out type-mean ratio_w vs bank (±0.10) ──")
    for t in CVD_TYPES:
        print(f"  {t}: heldout={type_mean[t]:.3f}  bank={BANK_RATIO_W[t]:.3f}  "
              f"Δ={type_mean[t]-BANK_RATIO_W[t]:+.3f}  "
              f"{'within' if within[t] else 'OUTSIDE'}")
    print(f"\nVERDICT: {'HELD-OUT CONSISTENT' if consistent else 'OUT OF BAND — report raw, judgment withheld'}")

    out = {
        "ckpt_step": ck.get("step"),
        "seed": a.seed, "pool_sampled": len(cand), "size": a.size,
        "selected": [{"tier": ti, "stem": st, "w_mean": cm}
                     for ti, cm, st in selected],
        "bank_ratio_w": BANK_RATIO_W, "tolerance": TOL,
        "heldout_type_mean_ratio_w": type_mean,
        "within_band": within, "consistent": consistent,
        "rows": rows,
    }
    dst = Path(a.ckpt).parent / "heldout_check.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/phase3_kaggle/v2_phase1/model_best.pt")
    p.add_argument("--val-dir", default="C:/Users/SCH/coco/val2017")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--pool", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260722)
    main(p.parse_args())
