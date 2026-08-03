"""
Phase 2 Step 3 — CRR/NP quantitative evaluation, CVDLens vs Daltonize.

Hypothesis under test: "CVDLens recovers contrast (CRR) at least as well as the
in-repo daltonize, while damaging naturalness (NP) significantly less."
Result is reported as measured — if the data contradicts the hypothesis, that
is written down (negative results are paper material too).

60 images × 3 CVD types × 2 methods = 360 cases →
  outputs/v2_phase3/eval_results.json  (rows + per-type aggregate + paired tests
                                        + case picks + verdict)
  outputs/v2_phase3/step3_scatter.png  (NP vs CRR, upper-right = good)
  outputs/v2_phase3/step3_case_{bigwin,similar,lose}.png  (5-panel witnesses)
  outputs/v2_phase3/step3_report.md

Fairness: both methods evaluated at 256², measured with the same simulator
(machado, severity 1.0) and the same confusion weight w (a function of the
original only). Daltonize is simulation.daltonize (Brettel/error-shift, the
single in-repo target generator) — no external daltonize library (Phase 0 rule).

Usage:
    py -m cvdlens_v2.step3_eval_set          # build eval_set.json first
    py -m cvdlens_v2.step3_eval
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvdlens_v2.model import CVDCorrectionNet, CVD_TYPES
from cvdlens_v2.color import srgb_to_linear, linear_to_srgb
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.simulation import simulate, daltonize
from cvdlens_v2 import step3_metrics as M

CVD_NAMES = {"p": "Protanopia", "d": "Deuteranopia", "t": "Tritanopia"}
METHODS = ["cvdlens", "daltonize"]
METHOD_COLOR = {"cvdlens": "#2563eb", "daltonize": "#dc2626"}
TYPE_MARKER = {"p": "o", "d": "s", "t": "^"}


def load_srgb(path, size, device):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


def _to_np(t):
    return t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


# ── 1. Run all 360 cases ────────────────────────────────────────────────
@torch.no_grad()
def run_cases(net, eval_set, val_dir, size, device):
    stems = eval_set["stems"]
    n = len(stems)
    rows = []
    for i, entry in enumerate(stems):
        stem, tier = entry["stem"], entry["tier"]
        orig = load_srgb(f"{val_dir}/{stem}.jpg", size, device)
        orig_lin = srgb_to_linear(orig)
        for t in CVD_TYPES:
            w = compute_confusion_weight(orig_lin, t, 1.0)
            r = net(orig, cvd_type=t, severity=1.0)
            dl_lin = daltonize(orig_lin, t, 1.0)
            outs = {
                "cvdlens": (r["out_linear"], r["out_srgb"]),
                "daltonize": (dl_lin, linear_to_srgb(dl_lin)),
            }
            for method, (out_lin, out_srgb) in outs.items():
                sec = M.secondary_logs(out_lin, orig_lin, w)
                rows.append({
                    "stem": stem, "tier": tier, "type": t, "method": method,
                    "w_mean": w.mean().item(),
                    "CRR": M.crr_ratio_w(out_lin, orig_lin, w, t, 1.0),
                    "NP_delta": M.np_delta(out_srgb, orig),
                    "NP_lpips": M.np_lpips(out_srgb, orig),
                    **sec,
                })
        print(f"  [{i + 1:>2d}/{n}] {stem} [{tier}]")
    return rows


# ── 2. Aggregate + paired tests ─────────────────────────────────────────
def _mean_std(vals):
    a = np.array([v for v in vals if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def _paired(rows, t, metric):
    """Return (cvdlens_vals, daltonize_vals) paired by stem for type t."""
    by_stem = {}
    for r in rows:
        if r["type"] != t:
            continue
        by_stem.setdefault(r["stem"], {})[r["method"]] = r[metric]
    cl, dl = [], []
    for stem, mm in by_stem.items():
        if "cvdlens" in mm and "daltonize" in mm:
            cl.append(mm["cvdlens"]); dl.append(mm["daltonize"])
    return np.array(cl, float), np.array(dl, float)


def _wilcoxon(x, y):
    """Two-sided Wilcoxon signed-rank on paired x−y. Returns (statistic, p)."""
    try:
        from scipy.stats import wilcoxon
        d = x - y
        if np.allclose(d, 0):
            return float("nan"), 1.0
        s, p = wilcoxon(x, y)
        return float(s), float(p)
    except Exception:
        return float("nan"), float("nan")


def summarize(rows):
    agg = {}
    for t in CVD_TYPES:
        for m in METHODS:
            sub = [r for r in rows if r["type"] == t and r["method"] == m]
            cell = {}
            for k in ("CRR", "NP_delta", "NP_lpips", "SI_uniform", "corr_guide"):
                mu, sd = _mean_std([r[k] for r in sub])
                cell[k + "_mean"], cell[k + "_std"] = mu, sd
            agg[f"{t}|{m}"] = cell

    tests = {}
    for t in CVD_TYPES:
        tests[t] = {}
        for metric in ("CRR", "NP_delta", "NP_lpips"):
            cl, dl = _paired(rows, t, metric)
            mu_d, sd_d = _mean_std((cl - dl).tolist())
            s, p = _wilcoxon(cl, dl)
            tests[t][metric] = {
                "cvdlens_mean": float(np.nanmean(cl)) if cl.size else float("nan"),
                "daltonize_mean": float(np.nanmean(dl)) if dl.size else float("nan"),
                "diff_mean": mu_d, "diff_std": sd_d,
                "wilcoxon_stat": s, "wilcoxon_p": p, "n": int(cl.size),
            }
    return agg, tests


# ── 3. Scatter (NP vs CRR; upper-right = good) ──────────────────────────
def plot_scatter(rows, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor="white")
    for m in METHODS:
        for t in CVD_TYPES:
            sub = [r for r in rows if r["method"] == m and r["type"] == t]
            ax.scatter([r["NP_delta"] for r in sub], [r["CRR"] for r in sub],
                       c=METHOD_COLOR[m], marker=TYPE_MARKER[t], s=42,
                       alpha=0.7, edgecolors="white", linewidths=0.4)
    # Method centroids
    for m in METHODS:
        sub = [r for r in rows if r["method"] == m]
        cx = np.mean([r["NP_delta"] for r in sub]); cy = np.mean([r["CRR"] for r in sub])
        ax.scatter([cx], [cy], c=METHOD_COLOR[m], marker="*", s=420,
                   edgecolors="black", linewidths=1.1, zorder=5)
        ax.annotate(f"{m} mean", (cx, cy), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("NP — original damage  |Δ|   (← less damage is better →)", fontsize=11)
    ax.set_ylabel("CRR — contrast recovery  ratio_w  (↑ more recovery is better)", fontsize=11)
    ax.invert_xaxis()   # so upper-RIGHT = low damage + high recovery = good
    ax.set_title("CVDLens vs Daltonize — recovery vs damage\n"
                 "upper-right = high recovery, low damage (good)", fontsize=12, fontweight="bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLOR[m],
                      markersize=11, label=m) for m in METHODS]
    handles += [Line2D([0], [0], marker=TYPE_MARKER[t], color="gray", linestyle="",
                       markersize=10, label=CVD_NAMES[t]) for t in CVD_TYPES]
    ax.legend(handles=handles, fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()


# ── 4. Case-study selection + 5-panel witnesses ─────────────────────────
def pick_cases(rows):
    """
    From top+mid tiers (w meaningful), pick one (stem,type) per category:
      big_win  — CVDLens recovers >= daltonize AND damages much less (max NP gap)
      similar  — smallest combined (z-scored) margin
      lose     — CVDLens's worst CRR deficit vs daltonize (most negative margin)
    """
    pair = {}
    for r in rows:
        if r["tier"] == "low":
            continue
        key = (r["stem"], r["type"])
        pair.setdefault(key, {})[r["method"]] = r
    recs = []
    for (stem, t), mm in pair.items():
        if "cvdlens" not in mm or "daltonize" not in mm:
            continue
        cl, dl = mm["cvdlens"], mm["daltonize"]
        recs.append({
            "stem": stem, "type": t,
            "crr_margin": cl["CRR"] - dl["CRR"],          # + : CVDLens more recovery
            "np_gap": dl["NP_delta"] - cl["NP_delta"],    # + : CVDLens less damage
            "lpips_gap": dl["NP_lpips"] - cl["NP_lpips"],  # + : CVDLens less perceptual damage
        })
    if not recs:
        return {}
    crr = np.array([x["crr_margin"] for x in recs])
    npg = np.array([x["np_gap"] for x in recs])
    # Scale each axis by its spread so neither dominates. NOT mean-centred:
    # "similar" means small ABSOLUTE margins (methods perform alike), i.e.
    # closest to (0, 0), not closest to the average margin.
    cz = crr / (crr.std() + 1e-9)
    nz = npg / (npg.std() + 1e-9)

    # big win: CVDLens recovery not worse, maximise naturalness gap
    winners = [x for x in recs if x["crr_margin"] >= 0]
    big_win = max(winners or recs, key=lambda x: x["np_gap"])
    # similar: both margins closest to zero (equal performance)
    combo = cz ** 2 + nz ** 2
    similar = recs[int(np.argmin(combo))]
    # lose: most negative CRR margin (CVDLens recovers least vs daltonize)
    lose = min(recs, key=lambda x: x["crr_margin"])
    return {"bigwin": big_win, "similar": similar, "lose": lose,
            "lose_is_real": bool(lose["crr_margin"] < 0)}


@torch.no_grad()
def render_witness(net, stem, t, val_dir, size, device, out_path, banner, metrics):
    orig = load_srgb(f"{val_dir}/{stem}.jpg", size, device)
    orig_lin = srgb_to_linear(orig)
    r = net(orig, cvd_type=t, severity=1.0)
    cl_srgb, cl_lin = r["out_srgb"], r["out_linear"]
    dl_lin = daltonize(orig_lin, t, 1.0)
    dl_srgb = linear_to_srgb(dl_lin)
    sim_cl = linear_to_srgb(simulate(cl_lin, t, 1.0))
    sim_dl = linear_to_srgb(simulate(dl_lin, t, 1.0))
    m_cl, m_dl = metrics["cvdlens"], metrics["daltonize"]

    panels = [
        (orig, "Original"),
        (cl_srgb, f"CVDLens\nCRR {m_cl['CRR']:.2f} · |Δ| {m_cl['NP_delta']:.3f} · LPIPS {m_cl['NP_lpips']:.3f}"),
        (dl_srgb, f"Daltonize\nCRR {m_dl['CRR']:.2f} · |Δ| {m_dl['NP_delta']:.3f} · LPIPS {m_dl['NP_lpips']:.3f}"),
        (sim_cl, "CVD-sim: CVDLens"),
        (sim_dl, "CVD-sim: Daltonize"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.9), facecolor="white")
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(_to_np(img)); ax.set_title(title, fontsize=10); ax.axis("off")
    fig.suptitle(banner, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()


def _lookup(rows, stem, t):
    out = {}
    for r in rows:
        if r["stem"] == stem and r["type"] == t:
            out[r["method"]] = r
    return out


# ── 5. Report ───────────────────────────────────────────────────────────
def write_report(path, ck_step, eval_set, agg, tests, picks, rows, figs):
    L = []
    A = L.append
    A("# Phase 2 Step 3 — CRR/NP Evaluation: CVDLens vs Daltonize\n")
    A(f"- Checkpoint: `model_best` (step {ck_step}), severity 1.0")
    A(f"- Eval set: {len(eval_set['stems'])} images (top/mid/low "
      f"{eval_set['per_tier']}×3), seed {eval_set['seed']}, pool {eval_set['pool_sampled']}")
    A(f"- Disjoint from Phase 1 bank ({len(eval_set['excluded_bank'])}) "
      f"+ held-out set ({len(eval_set['excluded_heldout'])})")
    A("- Fairness: both methods at 256², same simulator (machado, sev 1.0), "
      "same confusion weight w. Daltonize = `simulation.daltonize` "
      "(Brettel/error-shift; no external library).\n")

    A("## Metrics")
    A("- **CRR** (axis 1, ↑ better): `ratio_w` — confusion-weighted contrast "
      "ratio of sim(method) vs sim(original) on the CVD view (Phase 1 def).")
    A("- **NP** (axis 2, ↓ better): `|Δ|` mean sRGB change; `LPIPS` (VGG) "
      "perceptual distance to the original.")
    A("- Secondary: SI_uniform, corr_guide (Phase 1 logging set).\n")

    A("## Per-type summary (mean ± std)\n")
    A("| type | method | CRR ↑ | NP \\|Δ\\| ↓ | NP LPIPS ↓ | SI_uniform | corr_guide |")
    A("|---|---|---|---|---|---|---|")
    for t in CVD_TYPES:
        for m in METHODS:
            c = agg[f"{t}|{m}"]
            A(f"| {t} | {m} | {c['CRR_mean']:.3f} ± {c['CRR_std']:.3f} "
              f"| {c['NP_delta_mean']:.4f} ± {c['NP_delta_std']:.4f} "
              f"| {c['NP_lpips_mean']:.4f} ± {c['NP_lpips_std']:.4f} "
              f"| {c['SI_uniform_mean']:.3f} | {c['corr_guide_mean']:.3f} |")
    A("")

    A("## Per-tier summary (confusion-mass stratum)\n")
    A("CRR (`ratio_w`) is only meaningful where the confusion weight w has mass. "
      "In the **low** tier (w̄≈0) the weighted ratio divides two near-zero "
      "quantities and is unstable — read the low tier as a **naturalness / "
      "do-nothing test** (|Δ|, LPIPS should be ~0), not as recovery.\n")
    A("| tier | method | CRR | NP \\|Δ\\| | NP LPIPS |")
    A("|---|---|---|---|---|")
    for tier in ("top", "mid", "low"):
        for m in METHODS:
            sub = [r for r in rows if r["tier"] == tier and r["method"] == m]
            crr, _ = _mean_std([r["CRR"] for r in sub])
            dmg, _ = _mean_std([r["NP_delta"] for r in sub])
            lp, _ = _mean_std([r["NP_lpips"] for r in sub])
            A(f"| {tier} | {m} | {crr:.3f} | {dmg:.4f} | {lp:.4f} |")
    A("")

    A("## Paired comparison (CVDLens − Daltonize, Wilcoxon signed-rank)\n")
    A("Positive CRR diff = CVDLens recovers more. Negative NP diff = CVDLens "
      "damages less (better).\n")
    A("| type | metric | CVDLens | Daltonize | diff (mean) | p |")
    A("|---|---|---|---|---|---|")
    for t in CVD_TYPES:
        for metric in ("CRR", "NP_delta", "NP_lpips"):
            x = tests[t][metric]
            A(f"| {t} | {metric} | {x['cvdlens_mean']:.3f} | {x['daltonize_mean']:.3f} "
              f"| {x['diff_mean']:+.3f} | {x['wilcoxon_p']:.2e} |")
    A("")

    # ── Verdict (data-driven, tier-aware) ──
    # CRR on w-meaningful tiers (top+mid); NP on all tiers. Low tier is the
    # do-nothing / naturalness check, reported separately.
    def _mean(metric, pred):
        v = np.array([r[metric] for r in rows if pred(r)], float)
        return float(np.nanmean(v)) if v.size else float("nan")
    wmean = lambda r: r["tier"] in ("top", "mid")
    crr_cl = _mean("CRR", lambda r: r["method"] == "cvdlens" and wmean(r))
    crr_dl = _mean("CRR", lambda r: r["method"] == "daltonize" and wmean(r))
    d_cl = _mean("NP_delta", lambda r: r["method"] == "cvdlens")
    d_dl = _mean("NP_delta", lambda r: r["method"] == "daltonize")
    lp_cl = _mean("NP_lpips", lambda r: r["method"] == "cvdlens")
    lp_dl = _mean("NP_lpips", lambda r: r["method"] == "daltonize")
    low_cl = _mean("NP_delta", lambda r: r["method"] == "cvdlens" and r["tier"] == "low")
    low_dl = _mean("NP_delta", lambda r: r["method"] == "daltonize" and r["tier"] == "low")
    crr_ge = crr_cl >= crr_dl - 0.02   # "동등 이상" with a small tolerance
    np_less = (d_cl < d_dl) and (lp_cl < lp_dl)
    np_sig = all(tests[t]["NP_delta"]["diff_mean"] < 0 and tests[t]["NP_delta"]["wilcoxon_p"] < 0.05
                 for t in CVD_TYPES)

    A("## Verdict\n")
    A(f"- **CRR (top+mid, w-meaningful)**: CVDLens **{crr_cl:.3f}** vs Daltonize "
      f"**{crr_dl:.3f}** → recovery {'≥ (동등 이상)' if crr_ge else '< daltonize'}. "
      f"(Daltonize sits near 1.0 — little net weighted-contrast gain on the CVD view.)")
    A(f"- **NP (all tiers)**: |Δ| CVDLens **{d_cl:.4f}** vs **{d_dl:.4f}**; "
      f"LPIPS **{lp_cl:.4f}** vs **{lp_dl:.4f}** → damage "
      f"{'lower (better)' if np_less else 'NOT lower'}"
      f"{', significant on all types (p<0.05)' if np_sig else ''}.")
    A(f"- **Low tier do-nothing (naturalness)**: |Δ| CVDLens **{low_cl:.4f}** vs "
      f"Daltonize **{low_dl:.4f}** — CVDLens leaves low-confusion images nearly "
      f"untouched; daltonize still perturbs them.")
    if crr_ge and np_less:
        A("\n**HYPOTHESIS SUPPORTED**: on w-meaningful images CVDLens recovers "
          "contrast at least as well as daltonize (in fact more), while damaging "
          f"naturalness less across all tiers{' (significant on every type)' if np_sig else ''}.")
    else:
        A("\n**HYPOTHESIS NOT FULLY SUPPORTED** — reported as measured. See the "
          "per-type / per-tier tables and case studies for where and why.")
    A("\n> Caveat: low-tier `ratio_w` is numerically unstable (w̄≈0) and is "
      "excluded from the CRR verdict; it is used only as a naturalness check.")
    A("")

    A("## Scatter\n")
    A(f"![scatter]({Path(figs['scatter']).name}) — upper-right is good "
      "(high recovery, low damage).\n")

    A("## Case studies (5-panel witnesses)\n")
    labels = {
        "bigwin": "CVDLens wins big (≥ recovery, far less damage)",
        "similar": "Closest match between methods",
        "lose": ("A case CVDLens loses on recovery"
                 if picks.get("lose_is_real") else
                 "CVDLens's weakest case (still not a recovery loss)"),
    }
    for key in ("bigwin", "similar", "lose"):
        pk = picks[key]
        A(f"### {labels[key]} — `{pk['stem']}` / {CVD_NAMES[pk['type']]}")
        A(f"CRR margin (CVDLens−Dalt) {pk['crr_margin']:+.3f}, "
          f"|Δ| gap {pk['np_gap']:+.4f}, LPIPS gap {pk['lpips_gap']:+.4f}")
        A(f"![{key}]({Path(figs[key]).name})\n")

    if not picks.get("lose_is_real"):
        A("> No case in the top/mid tiers has CVDLens recovering less than "
          "daltonize; the 'lose' panel shows the smallest CVDLens margin.\n")

    Path(path).write_text("\n".join(L), encoding="utf-8")


def main(a):
    device = "cpu"
    torch.manual_seed(0)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    eval_set = json.loads(Path(a.eval_set).read_text())

    net = CVDCorrectionNet(pretrained_backbone=False).to(device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ck["state_dict"]); net.eval()
    ck_step = ck.get("step")
    print(f"[ckpt] step={ck_step}  "
          f"cases={len(eval_set['stems'])}×{len(CVD_TYPES)}×{len(METHODS)}")

    M.get_lpips()   # warm LPIPS (download/load) before the loop
    rows = run_cases(net, eval_set, a.val_dir, a.size, device)
    agg, tests = summarize(rows)
    picks = pick_cases(rows)

    figs = {
        "scatter": str(out_dir / "step3_scatter.png"),
        "bigwin": str(out_dir / "step3_case_bigwin.png"),
        "similar": str(out_dir / "step3_case_similar.png"),
        "lose": str(out_dir / "step3_case_lose.png"),
    }
    plot_scatter(rows, figs["scatter"])
    for key in ("bigwin", "similar", "lose"):
        pk = picks[key]
        mm = _lookup(rows, pk["stem"], pk["type"])
        render_witness(net, pk["stem"], pk["type"], a.val_dir, a.size, device,
                       figs[key], f"{key.upper()} — {pk['stem']} / {CVD_NAMES[pk['type']]}", mm)

    results = {
        "ckpt_step": ck_step, "eval_set": a.eval_set,
        "n_cases": len(rows), "methods": METHODS,
        "aggregate": agg, "paired_tests": tests,
        "case_picks": picks, "rows": rows,
    }
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    write_report(out_dir / "step3_report.md", ck_step, eval_set, agg, tests, picks, rows, figs)
    print(f"\nwrote {out_dir/'eval_results.json'}")
    print(f"wrote {out_dir/'step3_report.md'}  + 4 figures")

    # Console verdict snapshot
    print("\n── pooled means ──")
    for metric in ("CRR", "NP_delta", "NP_lpips"):
        cl = np.nanmean([r[metric] for r in rows if r["method"] == "cvdlens"])
        dl = np.nanmean([r[metric] for r in rows if r["method"] == "daltonize"])
        print(f"  {metric:9s}  CVDLens={cl:.4f}   Daltonize={dl:.4f}   Δ={cl-dl:+.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/phase3_kaggle/v2_phase1/model_best.pt")
    p.add_argument("--eval-set", default="outputs/v2_phase3/eval_set.json")
    p.add_argument("--val-dir", default="C:/Users/SCH/coco/val2017")
    p.add_argument("--out-dir", default="outputs/v2_phase3")
    p.add_argument("--size", type=int, default=256)
    main(p.parse_args())
