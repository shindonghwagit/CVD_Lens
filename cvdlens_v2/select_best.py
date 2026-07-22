"""
Phase 1 best-checkpoint selector.

The LAST step is not necessarily the best: p/d mean ratio_w peak mid-run and
regress afterward while t keeps climbing (the "724 regression" pattern). We
therefore pick, among the gate-passing checkpoints, the step that MAXIMISES
the sum of the three type-mean ratio_w values — the balanced optimum.

Hard gate (same as train.validate):
    per-type mean ratio_w ≥ {p:1.10, d:1.13, t:1.27}   (mean over 9 active imgs)
    do-nothing anchor (LOW_W_STEM) |Δ| < 0.005

Reads the per-step val_step*.json files (schema-agnostic: only needs
`rows[*].{type, ratio_w, is_low_w, delta}`), selects the winner, copies
model_step{best:06d}.pt → model_best.pt, and writes selection_rationale.json.

Usage:
    py -m cvdlens_v2.select_best --dir outputs/phase3_kaggle/v2_phase1
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
from pathlib import Path

RATIO_W_MIN_BY_TYPE = {"p": 1.10, "d": 1.13, "t": 1.27}
DELTA_MAX_LOW_W = 0.005
CVD_TYPES = ("p", "d", "t")


def gate_for(val: dict) -> dict:
    """Recompute the type-mean hard gate from a val JSON's rows."""
    by = {t: [] for t in CVD_TYPES}
    low_w_delta = None
    for r in val["rows"]:
        if r.get("is_low_w"):
            low_w_delta = r["delta"]
        else:
            by[r["type"]].append(r["ratio_w"])
    type_mean = {t: (sum(v) / len(v) if v else float("nan")) for t, v in by.items()}
    type_pass = {t: type_mean[t] >= RATIO_W_MIN_BY_TYPE[t] for t in CVD_TYPES}
    low_w_pass = (low_w_delta is not None and low_w_delta < DELTA_MAX_LOW_W)
    return {
        "type_mean_ratio_w": type_mean,
        "type_pass": type_pass,
        "low_w_delta": low_w_delta,
        "low_w_pass": low_w_pass,
        "all_pass": all(type_pass.values()) and low_w_pass,
        "score_sum": sum(type_mean.values()),
    }


def main(args):
    d = Path(args.dir)
    val_files = sorted(d.glob("val_step*.json"))
    assert val_files, f"no val_step*.json under {d}"

    table = []
    for f in val_files:
        step = int(re.search(r"(\d+)", f.name).group(1))
        g = gate_for(json.loads(f.read_text()))
        g["step"] = step
        table.append(g)

    passing = [g for g in table if g["all_pass"]]
    assert passing, "no checkpoint passes the hard gate"
    best = max(passing, key=lambda g: g["score_sum"])
    step = best["step"]

    src = d / f"model_step{step:06d}.pt"
    assert src.exists(), f"checkpoint missing: {src}"
    dst = d / "model_best.pt"
    shutil.copyfile(src, dst)

    rationale = {
        "selected_step": step,
        "selected_checkpoint": src.name,
        "copied_to": dst.name,
        "criterion": "max sum of per-type mean ratio_w among gate-passing steps",
        "hard_gate": {
            "type_thresholds": RATIO_W_MIN_BY_TYPE,
            "low_w_delta_max": DELTA_MAX_LOW_W,
        },
        "selected_metrics": {
            "type_mean_ratio_w": best["type_mean_ratio_w"],
            "score_sum": best["score_sum"],
            "low_w_delta": best["low_w_delta"],
        },
        "why_not_last_step": (
            "p/d mean ratio_w peak mid-run then regress while t keeps rising; "
            "the last step (20000) trades away d headroom. Selected step is the "
            "balanced maximum."
        ),
        "per_step": [
            {
                "step": g["step"],
                "p": round(g["type_mean_ratio_w"]["p"], 4),
                "d": round(g["type_mean_ratio_w"]["d"], 4),
                "t": round(g["type_mean_ratio_w"]["t"], 4),
                "sum": round(g["score_sum"], 4),
                "gate_pass": g["all_pass"],
            }
            for g in table
        ],
    }
    (d / "selection_rationale.json").write_text(json.dumps(rationale, indent=2))

    print(f"BEST step {step}  "
          f"p={best['type_mean_ratio_w']['p']:.3f} "
          f"d={best['type_mean_ratio_w']['d']:.3f} "
          f"t={best['type_mean_ratio_w']['t']:.3f}  "
          f"sum={best['score_sum']:.3f}")
    print(f"copied {src.name} -> {dst.name}")
    print(f"wrote  selection_rationale.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/phase3_kaggle/v2_phase1")
    main(p.parse_args())
