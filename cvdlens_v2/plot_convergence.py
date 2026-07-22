"""
Phase 1 convergence curve from per-step val_step*.json.

Top panel : per-type mean ratio_w (P/D/T) over training, with the 3 recalibrated
            thresholds (P 1.10 / D 1.13 / T 1.27) as dashed lines and a vertical
            marker at the selected best step.
Bottom    : per-type mean |Δ| over the 9 CVD-active images.

Also prints an "effective convergence" diagnosis: the first step past which the
summed type-mean ratio_w stays within `--band` of its running-max plateau.

Usage:
    py -m cvdlens_v2.plot_convergence --dir outputs/phase3_kaggle/v2_phase1 \
        --best 9000 --out outputs/phase3_kaggle/v2_phase1/convergence_curve.png
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESH = {"p": 1.10, "d": 1.13, "t": 1.27}
COLORS = {"p": "#d62728", "d": "#2ca02c", "t": "#1f77b4"}
LABELS = {"p": "Protan", "d": "Deutan", "t": "Tritan"}
LOW_W = "000000001761"


def series(d: Path):
    steps, rw, dl = [], {"p": [], "d": [], "t": []}, {"p": [], "d": [], "t": []}
    for f in sorted(d.glob("val_step*.json")):
        m = re.search(r"val_step(\d+)\.json", f.name)
        if not m:
            continue
        step = int(m.group(1))
        v = json.loads(f.read_text())
        by_rw = {"p": [], "d": [], "t": []}
        by_dl = {"p": [], "d": [], "t": []}
        for r in v["rows"]:
            if r.get("is_low_w"):
                continue
            by_rw[r["type"]].append(r["ratio_w"])
            by_dl[r["type"]].append(r["delta"])
        steps.append(step)
        for t in THRESH:
            rw[t].append(sum(by_rw[t]) / len(by_rw[t]))
            dl[t].append(sum(by_dl[t]) / len(by_dl[t]))
    return steps, rw, dl


def effective_convergence(steps, rw, band):
    """First step from which summed type-mean ratio_w stays within `band` of the
    eventual running-max, i.e. no later point exceeds it by more than `band`."""
    total = [rw["p"][i] + rw["d"][i] + rw["t"][i] for i in range(len(steps))]
    final_plateau = max(total)
    for i, s in enumerate(steps):
        if all(total[j] <= total[i] + band for j in range(i, len(steps))):
            return s, total[i], final_plateau
    return steps[-1], total[-1], final_plateau


def main(a):
    d = Path(a.dir)
    steps, rw, dl = series(d)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}, facecolor="white")

    for t in ("p", "d", "t"):
        ax1.plot(steps, rw[t], "-o", ms=3, color=COLORS[t],
                 label=f"{LABELS[t]} mean ratio_w")
        ax1.axhline(THRESH[t], ls="--", lw=1, color=COLORS[t], alpha=0.6)
        ax1.text(steps[-1], THRESH[t], f"  {LABELS[t]} thr {THRESH[t]:.2f}",
                 va="center", ha="left", fontsize=8, color=COLORS[t])
    ax1.axvline(a.best, color="k", ls="-", lw=1.2, alpha=0.7)
    ymax = max(max(rw[t]) for t in rw)
    ax1.annotate("selected (best)", xy=(a.best, ymax), xytext=(a.best + 800, ymax),
                 fontsize=9, fontweight="bold", va="top")
    ax1.set_ylabel("type-mean ratio_w")
    ax1.set_title("Phase 1 convergence — type-mean ratio_w (9 CVD-active images)",
                  fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    for t in ("p", "d", "t"):
        ax2.plot(steps, dl[t], "-o", ms=3, color=COLORS[t],
                 label=f"{LABELS[t]} mean |Δ|")
    ax2.axvline(a.best, color="k", ls="-", lw=1.2, alpha=0.7)
    ax2.set_ylabel("type-mean |Δ|")
    ax2.set_xlabel("training step")
    ax2.set_title("type-mean |Δ| (identity movement)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(a.out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()

    eff_step, eff_val, plateau = effective_convergence(steps, rw, a.band)
    still_rising = eff_step >= steps[-1]
    print(f"saved {a.out}")
    print(f"summed type-mean ratio_w: first={rw['p'][0]+rw['d'][0]+rw['t'][0]:.3f} "
          f"plateau_max={plateau:.3f}")
    print(f"effective convergence (within band={a.band}): step {eff_step} "
          f"(sum={eff_val:.3f})")
    print("STILL RISING AT END" if still_rising else "PLATEAUED before end")
    # per-type peak step
    for t in ("p", "d", "t"):
        pk = steps[max(range(len(steps)), key=lambda i: rw[t][i])]
        print(f"  {LABELS[t]}: peak at step {pk}  (final {rw[t][-1]:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/phase3_kaggle/v2_phase1")
    p.add_argument("--best", type=int, default=9000)
    p.add_argument("--band", type=float, default=0.03,
                   help="ratio_w-sum tolerance for 'effective convergence'")
    p.add_argument("--out",
                   default="outputs/phase3_kaggle/v2_phase1/convergence_curve.png")
    main(p.parse_args())
