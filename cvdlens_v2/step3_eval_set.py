"""
Phase 2 Step 3 — evaluation-set builder (stratified 60-image sample).

Ranks a fixed, seeded candidate pool from COCO val2017 by confusion mass
(mean over {p,d,t} of w.mean() — the same type-agnostic scalar used by
heldout_check / validate_multi) and takes the top-20 / mid-20 / bottom-20
tiers, for 60 images spanning the confusion-mass distribution.

Disjointness (selection-bias guard): the pool excludes both the Phase 1
validation bank (10 stems, validate_multi.IMAGES) and the held-out spot-check
set (8 stems, heldout_check.json). So no image the model was tuned on, or that
model selection ever looked at, can leak into the eval set.

Writes outputs/v2_phase3/eval_set.json.

Usage:
    py -m cvdlens_v2.step3_eval_set
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from cvdlens_v2.model import CVD_TYPES
from cvdlens_v2.color import srgb_to_linear
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.validate_multi import IMAGES as BANK

# 8 held-out spot-check stems. Read from heldout_check.json when present so the
# exclusion never drifts from what heldout_check actually selected; the literal
# is a fallback (values as of step-9000 selection, seed 20260722).
_HELDOUT_JSON = Path("outputs/phase3_kaggle/v2_phase1/heldout_check.json")
_HELDOUT_FALLBACK = [
    "000000243344", "000000379441", "000000561465", "000000221872",
    "000000218362", "000000437110", "000000451879", "000000166918",
]


def load_heldout_stems() -> list[str]:
    if _HELDOUT_JSON.exists():
        d = json.loads(_HELDOUT_JSON.read_text())
        return [s["stem"] for s in d["selected"]]
    return list(_HELDOUT_FALLBACK)


def load_srgb(path, size, device):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


@torch.no_grad()
def confusion_mass(orig_lin) -> float:
    return sum(compute_confusion_weight(orig_lin, t, 1.0).mean().item()
               for t in CVD_TYPES) / len(CVD_TYPES)


def build(a):
    device = "cpu"
    random.seed(a.seed)

    heldout = load_heldout_stems()
    exclude = set(BANK) | set(heldout)
    all_jpg = sorted(Path(a.val_dir).glob("*.jpg"))
    pool_paths = [p for p in all_jpg if p.stem not in exclude]
    cand = random.sample(pool_paths, min(a.pool, len(pool_paths)))
    print(f"corpus={len(all_jpg)}  excluded={len(exclude)}  "
          f"pool={len(pool_paths)}  sampling {len(cand)} (seed={a.seed})")

    ranked = []
    for i, p in enumerate(cand):
        orig = load_srgb(p, a.size, device)
        ranked.append((confusion_mass(srgb_to_linear(orig)), p.stem))
        if (i + 1) % 50 == 0:
            print(f"  ranked {i + 1}/{len(cand)}")
    ranked.sort(reverse=True)

    n = len(ranked)
    k = a.per_tier
    assert n >= 3 * k, f"pool too small ({n}) for 3×{k} disjoint tiers"
    top = ranked[:k]
    bottom = ranked[-k:]
    mid_start = n // 2 - k // 2
    mid = ranked[mid_start:mid_start + k]

    stems = ([{"stem": s, "tier": "top", "w_mean": cm} for cm, s in top]
             + [{"stem": s, "tier": "mid", "w_mean": cm} for cm, s in mid]
             + [{"stem": s, "tier": "low", "w_mean": cm} for cm, s in bottom])

    sset = {x["stem"] for x in stems}
    assert len(sset) == len(stems), "duplicate stems across tiers (pool too small)"
    assert not (sset & exclude), "eval set overlaps bank/heldout"

    out = {
        "seed": a.seed, "pool_sampled": len(cand), "per_tier": k, "size": a.size,
        "val_dir": a.val_dir,
        "excluded_bank": sorted(BANK),
        "excluded_heldout": sorted(heldout),
        "tier_w_ranges": {
            "top": [top[-1][0], top[0][0]],
            "mid": [mid[-1][0], mid[0][0]],
            "low": [bottom[-1][0], bottom[0][0]],
        },
        "stems": stems,
    }
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}  ({len(stems)} stems: top/mid/low = {k}/{k}/{k})")
    for tier in ("top", "mid", "low"):
        r = out["tier_w_ranges"][tier]
        print(f"  {tier}: w̄ ∈ [{r[0]:.3f}, {r[1]:.3f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val-dir", default="C:/Users/SCH/coco/val2017")
    p.add_argument("--out", default="outputs/v2_phase3/eval_set.json")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--pool", type=int, default=300)
    p.add_argument("--per-tier", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260803)
    build(p.parse_args())
