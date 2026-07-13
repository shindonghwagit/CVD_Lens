"""
Ray scan of loss landscape along visible-subspace basis directions.

For each CVD type and each direction d in {+b_L, -b_L, +b_C, -b_C}:
    out(β) = clamp(orig + w * β * d, 0, 1)
    for β in linspace(0, 0.3, 16)

Records L_c, L_g, L_natural, total, and C(sim(out(β))) blurred with σ=1.
Auto-emits verdict A/B/C per type based on judgment rules:
    A  some direction has L_c(β) monotonically ↓ AND C(sim) ↑
    B  every direction: L_c(β) flat or ↑  → L_c still identity-favoring
    C  L_c ↓ but total ↑  → L_natural or L_g is cancelling
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.basis import compose_delta, get_basis
from cvdlens_v2.color import srgb_to_linear, linear_to_srgb, rgb_to_lab
from cvdlens_v2.confusion import compute_confusion_weight, _gaussian_kernel_2d, _blur
from cvdlens_v2.losses import CVDLossV2, _contrast_magnitude, contrast_loss
from cvdlens_v2.simulation import simulate


CVD_NAMES = {"p": "Protanopia", "d": "Deuteranopia", "t": "Tritanopia"}
DIRECTIONS = ["+b_L", "-b_L", "+b_C", "-b_C"]
DIR_COLOR = {"+b_L": "#d62728", "-b_L": "#1f77b4",
             "+b_C": "#2ca02c", "-b_C": "#ff7f0e"}
DIR_STYLE = {"+b_L": "-", "-b_L": "-", "+b_C": "--", "-b_C": "--"}


def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)


def _delta_for_direction(direction: str, beta: float, B: int, H: int, W: int,
                          cvd_type: str, device, dtype) -> torch.Tensor:
    """Return (B,3,H,W) delta in visible subspace for named direction."""
    zeros = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    val = torch.full_like(zeros, beta)
    if direction == "+b_L":
        return compose_delta(+val, zeros, cvd_type)
    if direction == "-b_L":
        return compose_delta(-val, zeros, cvd_type)
    if direction == "+b_C":
        return compose_delta(zeros, +val, cvd_type)
    if direction == "-b_C":
        return compose_delta(zeros, -val, cvd_type)
    raise ValueError(direction)


def _blurred_C(sim_lin: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Contrast magnitude of Lab of sim, after blurring RGB by σ (channel-wise)."""
    C = sim_lin.shape[1]
    k = _gaussian_kernel_2d(sigma, 5, sim_lin.device, sim_lin.dtype)
    k = k.expand(C, 1, 5, 5).contiguous()
    pad = 2
    sim_blur = F.conv2d(F.pad(sim_lin, [pad] * 4, mode='reflect'), k, groups=C)
    lab = rgb_to_lab(sim_blur)
    return _contrast_magnitude(lab)


def _scale_deficit_breakdown(orig_lin: torch.Tensor, out_lin: torch.Tensor,
                              w: torch.Tensor, cvd_type: str,
                              scales=(1, 2, 4, 8)) -> dict:
    """Report L_c contribution per scale."""
    sim_out = simulate(out_lin, cvd_type, 1.0)
    lab_o = rgb_to_lab(orig_lin.float())
    lab_a = rgb_to_lab(sim_out.float())
    per_scale = {}
    for s in scales:
        lo = F.avg_pool2d(lab_o, s, s) if s > 1 else lab_o
        la = F.avg_pool2d(lab_a, s, s) if s > 1 else lab_a
        m = F.avg_pool2d(w, s, s) if s > 1 else w
        deficit = torch.relu(_contrast_magnitude(lo) - _contrast_magnitude(la))
        per_scale[s] = (m * deficit.pow(2)).mean().item()
    return per_scale


@torch.no_grad()
def scan_type(orig_srgb: torch.Tensor, cvd_type: str, severity: float,
              betas: np.ndarray, loss_fn: CVDLossV2, seed: int = 42) -> dict:
    device = orig_srgb.device
    orig_lin = srgb_to_linear(orig_srgb)
    B, _, H, W = orig_lin.shape
    w = compute_confusion_weight(orig_lin, cvd_type, severity)

    sim_orig = simulate(orig_lin, cvd_type, severity)
    C_sim_orig_mean = _blurred_C(sim_orig, sigma=1.0).mean().item()

    gen = torch.Generator(device=device).manual_seed(seed)
    results = {d: {"beta": [], "L_c": [], "L_g": [], "L_n": [],
                    "total": [], "C_sim_mean": [],
                    "pred_diff": []} for d in DIRECTIONS}

    for d in DIRECTIONS:
        for b in betas:
            delta = _delta_for_direction(d, float(b), B, H, W, cvd_type,
                                          device, orig_lin.dtype)
            out_lin = (orig_lin + w * delta).clamp(0.0, 1.0)
            out_srgb = linear_to_srgb(out_lin)

            gen.manual_seed(seed)
            total, comps = loss_fn(
                out_linear=out_lin, orig_linear=orig_lin,
                out_srgb=out_srgb, orig_srgb=orig_srgb,
                w=w, cvd_type=cvd_type, severity=severity, generator=gen,
            )
            sim_out = simulate(out_lin, cvd_type, severity)
            C_sim_mean = _blurred_C(sim_out, sigma=1.0).mean().item()
            pred_diff = (out_lin - orig_lin).abs().mean().item()

            results[d]["beta"].append(float(b))
            results[d]["L_c"].append(comps["L_contrast"])
            results[d]["L_g"].append(comps["L_global"])
            results[d]["L_n"].append(comps["L_natural"])
            results[d]["total"].append(total.item())
            results[d]["C_sim_mean"].append(C_sim_mean)
            results[d]["pred_diff"].append(pred_diff)

    results["_C_sim_orig_mean"] = C_sim_orig_mean

    # Diagnostics at β=0 (identity)
    # Per-scale deficit breakdown
    zero_out = orig_lin.clone()
    results["_scale_deficit_identity"] = _scale_deficit_breakdown(
        orig_lin, zero_out, w, cvd_type)

    # Finite-difference gradient of L_c wrt β at β=0 for each direction
    grads = {}
    dbeta = float(betas[1] - betas[0]) if len(betas) > 1 else 0.02
    for d in DIRECTIONS:
        L_c0 = results[d]["L_c"][0]
        L_c1 = results[d]["L_c"][1]
        grads[d] = (L_c1 - L_c0) / dbeta
    results["_grad_Lc_at_0"] = grads

    return results


def verdict_for_type(cvd_type: str, res: dict) -> dict:
    """
    Apply A/B/C rule per type.

    A: at least one direction has L_c(β) monotonically ↓ over first half
       AND C(sim) increasing in that direction AND total(β) also ↓.
    B: every direction has L_c(β) non-decreasing (grad ≥ ~0 at 0).
    C: some direction has L_c ↓ but total ↑ over the same range.
    """
    ver = {"cvd_type": cvd_type, "per_dir": {}}
    best_dir_A = None
    best_drop_A = 0.0
    saw_lc_drop = False
    saw_total_up_with_lc_drop = False
    all_flat_or_up = True

    for d in DIRECTIONS:
        Lc = np.array(res[d]["L_c"])
        Lg = np.array(res[d]["L_g"])
        Ln = np.array(res[d]["L_n"])
        total = np.array(res[d]["total"])
        Csim = np.array(res[d]["C_sim_mean"])
        Lc_drop = Lc[0] - Lc.min()
        Lc_argmin = int(np.argmin(Lc))
        total_drop = total[0] - total.min()
        total_argmin = int(np.argmin(total))
        Csim_gain = Csim.max() - Csim[0]
        Csim_argmax = int(np.argmax(Csim))

        # Grad sign at 0
        grad0 = res["_grad_Lc_at_0"][d]
        if grad0 < -1e-6:
            all_flat_or_up = False
        if Lc_drop > 1e-4:
            saw_lc_drop = True
            if total_drop <= 1e-4:
                saw_total_up_with_lc_drop = True
            # Alignment: L_c min direction and C_sim max direction agree?
            aligned = (Csim_argmax > 0) and (Lc_argmin > 0)
            if aligned and Lc_drop > best_drop_A and total_drop > 1e-4:
                best_drop_A = Lc_drop
                best_dir_A = (d, Lc_argmin, res[d]["beta"][Lc_argmin])

        ver["per_dir"][d] = {
            "grad_Lc_at_0": float(grad0),
            "Lc_drop": float(Lc_drop),
            "Lc_argmin_beta": float(res[d]["beta"][Lc_argmin]),
            "total_drop": float(total_drop),
            "total_argmin_beta": float(res[d]["beta"][total_argmin]),
            "Csim_gain": float(Csim_gain),
            "Csim_argmax_beta": float(res[d]["beta"][Csim_argmax]),
        }

    if best_dir_A is not None:
        ver["verdict"] = "A"
        ver["best_direction"] = best_dir_A[0]
        ver["best_beta"] = best_dir_A[2]
        ver["reason"] = ("A: found aligned direction with L_c ↓ + total ↓ + "
                         "C(sim) ↑. Phase 0 re-run cleared.")
    elif not saw_lc_drop or all_flat_or_up:
        ver["verdict"] = "B"
        ver["reason"] = ("B: L_c did not decrease in any direction (grads ≥ 0 "
                         "at identity). L_c is still identity-favoring — "
                         "no Phase 0 re-run; L_c must be redesigned.")
    else:
        ver["verdict"] = "C"
        ver["reason"] = ("C: L_c decreased in some direction but total did "
                         "not — L_natural or L_g is cancelling the gain. "
                         "Report cancellation magnitudes below.")

    ver["scale_deficit_identity"] = res["_scale_deficit_identity"]
    ver["C_sim_orig_mean"] = res["_C_sim_orig_mean"]
    return ver


def plot_type(cvd_type: str, res: dict, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
    fig.suptitle(f"Ray Scan — {CVD_NAMES[cvd_type]}",
                 fontsize=14, fontweight='bold')

    plots = [
        ("L_c(β)",       "L_c",        axes[0, 0]),
        ("L_g(β)",       "L_g",        axes[0, 1]),
        ("L_natural(β)", "L_n",        axes[0, 2]),
        ("total(β)",     "total",      axes[1, 0]),
        ("C(sim(out(β)))  (blur σ=1, mean)", "C_sim_mean", axes[1, 1]),
        ("|Δ|(β) linear", "pred_diff", axes[1, 2]),
    ]

    for title, key, ax in plots:
        for d in DIRECTIONS:
            ax.plot(res[d]["beta"], res[d][key],
                    color=DIR_COLOR[d], linestyle=DIR_STYLE[d],
                    marker='o', markersize=3, label=d, linewidth=1.6)
        ax.set_xlabel('β')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    axes[1, 1].axhline(y=res["_C_sim_orig_mean"], color='gray',
                       linestyle=':', label='C(sim(orig))')
    axes[1, 1].legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def write_table(cvd_type: str, res: dict, table_path: Path):
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(f"# Ray scan {cvd_type} — {CVD_NAMES[cvd_type]}\n")
        f.write(f"# C(sim(orig))_mean = {res['_C_sim_orig_mean']:.4f}\n")
        f.write(f"# grad(L_c)/dβ at β=0: "
                + "  ".join(f"{d}={res['_grad_Lc_at_0'][d]:+.4f}"
                            for d in DIRECTIONS) + "\n")
        f.write(f"# scale deficit @ identity: {res['_scale_deficit_identity']}\n\n")
        for d in DIRECTIONS:
            f.write(f"\n## direction {d}\n")
            f.write(f"{'beta':>8} {'L_c':>10} {'L_g':>10} {'L_n':>10}"
                    f" {'total':>10} {'C_sim':>10} {'|Δ|':>10}\n")
            for i in range(len(res[d]["beta"])):
                f.write(f"{res[d]['beta'][i]:>8.4f}"
                        f" {res[d]['L_c'][i]:>10.4f}"
                        f" {res[d]['L_g'][i]:>10.4f}"
                        f" {res[d]['L_n'][i]:>10.4f}"
                        f" {res[d]['total'][i]:>10.4f}"
                        f" {res[d]['C_sim_mean'][i]:>10.4f}"
                        f" {res[d]['pred_diff'][i]:>10.4f}\n")


def print_verdict(ver: dict):
    t = ver["cvd_type"]
    print(f"\n{'='*74}\n[{t}] {CVD_NAMES[t]}  →  VERDICT: {ver['verdict']}")
    print(f"{'='*74}")
    print(f"  {ver['reason']}")
    print(f"  per-direction summary:")
    header = f"    {'dir':>6} {'grad@0':>10} {'Lc_drop':>10} {'β@Lc_min':>10}" \
             f" {'tot_drop':>10} {'β@tot_min':>10} {'Csim_gain':>10} {'β@Csim_max':>11}"
    print(header)
    for d in DIRECTIONS:
        p = ver["per_dir"][d]
        print(f"    {d:>6} {p['grad_Lc_at_0']:>+10.4f}"
              f" {p['Lc_drop']:>10.4f} {p['Lc_argmin_beta']:>10.4f}"
              f" {p['total_drop']:>10.4f} {p['total_argmin_beta']:>10.4f}"
              f" {p['Csim_gain']:>10.4f} {p['Csim_argmax_beta']:>11.4f}")
    if ver["verdict"] == "A":
        print(f"  best direction: {ver['best_direction']}  best β: {ver['best_beta']:.4f}")
    print(f"  scale deficit @ identity (should concentrate at fine scales):"
          f" {ver['scale_deficit_identity']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str,
                   default="C:/Users/SCH/coco/val2017/000000000724.jpg")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--severity", type=float, default=1.0)
    p.add_argument("--beta-max", type=float, default=0.3)
    p.add_argument("--n-beta", type=int, default=16)
    p.add_argument("--use-lpips", action="store_true")
    p.add_argument("--out-dir", type=str,
                   default="C:/Users/SCH/graduation_project/outputs/ray_scan")
    p.add_argument("--types", type=str, default="p,d,t",
                   help="Comma-separated subset of CVD types to scan")
    p.add_argument("--tag", type=str, default="",
                   help="Optional suffix on output filenames (e.g. '_ext')")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    orig_srgb = load_image(args.image, size=args.size).to(device)
    loss_fn = CVDLossV2(use_lpips=args.use_lpips).to(device)
    betas = np.linspace(0.0, args.beta_max, args.n_beta)

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    tag = args.tag
    all_verdicts = {}
    for t in types:
        print(f"\n>>> Ray scan {t} ({CVD_NAMES[t]})")
        res = scan_type(orig_srgb, t, args.severity, betas, loss_fn)
        plot_type(t, res, out_dir / f"ray_scan_{t}{tag}.png")
        write_table(t, res, out_dir / f"ray_scan_{t}{tag}.txt")
        ver = verdict_for_type(t, res)
        all_verdicts[t] = ver
        print_verdict(ver)

    verdict_path = out_dir / f"verdicts{tag}.json"
    with open(verdict_path, 'w', encoding='utf-8') as f:
        json.dump({t: {k: v for k, v in ver.items()
                        if k != "scale_deficit_identity"}  # for readability
                    for t, ver in all_verdicts.items()},
                    f, indent=2, default=float)
    print(f"\nSaved verdicts → {verdict_path}")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for t in types:
        ver = all_verdicts[t]
        extra = ""
        if ver["verdict"] == "A":
            extra = f"  best={ver['best_direction']} β={ver['best_beta']:.3f}"
        print(f"  [{t}] {CVD_NAMES[t]:15s}  →  {ver['verdict']}{extra}")


if __name__ == "__main__":
    main()
