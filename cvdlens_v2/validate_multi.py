"""
Multi-image Phase 0 validation.

Runs the same optimize() as validate_loss.py across a curated list of
COCO val images, with the confirmed λ_tv. Applies revised pass criteria
per image (|Δ| > 0.01, ratio_w > 1.15, loss decreased, SI < 1.5× baseline),
plus a REVERSED criterion for the tagged low-w image:
    |Δ| < 0.005  (should stay near-identity when confusion is minimal).

Usage:
    py -m cvdlens_v2.validate_multi --lambda-tv 0.03 --iters 200
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.validate_loss import (
    CVD_NAMES, load_image, optimize, plot_result,
)


# Curated 10-image list. Chosen from w_mean_scan.json to span w range.
# The last entry is the low-w "do-nothing" test.
IMAGES = [
    "000000000724",   # w=0.184  moderate — original witness
    "000000000632",   # w=0.291  moderate
    "000000000872",   # w=0.501  high
    "000000000776",   # w=0.730  very high (traffic-sign heavy)
    "000000001000",   # w=0.535  high
    "000000001296",   # w=0.497  high
    "000000001584",   # w=0.392  medium-high
    "000000002006",   # w=0.326  moderate
    "000000002261",   # w=0.627  very high
    "000000001761",   # w=0.024  LOW-W — reversed criterion (|Δ| < 0.005)
]
LOW_W_STEM = "000000001761"

DELTA_MIN = 0.01
DELTA_MAX_LOW_W = 0.005
RATIO_W_MIN = 1.15
SI_MULT = 1.5      # SI < SI_MULT × per-type baseline (from exp1c)


def load_baselines() -> dict:
    """Read SI baselines produced by aggregate_exp1c (tv=0.1 SI per type)."""
    agg_path = Path("C:/Users/SCH/graduation_project/outputs/v2_phase0/exp1c_aggregate.json")
    with open(agg_path) as f:
        agg = json.load(f)
    return agg["criteria"]["si_baselines"]


def main(args):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   λ_tv={args.lambda_tv}   iters={args.iters}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    si_base = load_baselines()

    types = ["p", "d", "t"]
    per_image = {}
    for i, stem in enumerate(IMAGES):
        img_path = f"C:/Users/SCH/coco/val2017/{stem}.jpg"
        print(f"\n[{i+1:>2d}/{len(IMAGES)}] {stem}"
              + ("  (LOW-W — reversed criterion)" if stem == LOW_W_STEM else ""))
        orig_srgb = load_image(img_path, size=args.size).to(device)
        per_type = {}
        for t in types:
            result = optimize(
                orig_srgb=orig_srgb,
                cvd_type=t,
                severity=1.0,
                n_iters=args.iters,
                lr=args.lr,
                field_size=64,
                use_lpips=False,
                log_every=args.iters,   # log only final iter to keep output terse
                init_direction="identity",
                init_beta=0.0,
                lambda_tv=args.lambda_tv,
                lambda_n=0.15,
                lambda_c=1.0,
                lambda_g=0.15,
                lambda_excess=args.lambda_excess,
            )
            final = result["history"][-1]
            si = final["speckle_index"]
            si_thr = SI_MULT * si_base[t]
            per_type[t] = {
                "delta": final["pred_diff"],
                "ratio": final["cvd_contrast_ratio"],
                "ratio_w": final["cvd_contrast_ratio_w"],
                "SI": si,
                "SI_thr": si_thr,
                "L_tv": final["L_tv"],
                "w_mean": result["w"].mean().item(),
                "initial_total": result["initial_total"],
                "final_total": result["final_total"],
            }
            # Save PNG for low-w image (special check) and for any that will fail
            # (decided after aggregation). Emit thumbnail for low-w now.
            if stem == LOW_W_STEM:
                png = out_dir / f"validate_{stem}_{t}_mi.png"
                plot_result(result, t, png)

            print(f"    {CVD_NAMES[t]:<14} |Δ|={final['pred_diff']:.4f}  "
                  f"ratio_w={final['cvd_contrast_ratio_w']:.3f}  "
                  f"SI={si:.3f} (thr={si_thr:.3f})  "
                  f"w̄={per_type[t]['w_mean']:.3f}")
        per_image[stem] = per_type

    # ── Pass/fail per image ────────────────────────────────────────
    print("\n" + "=" * 82)
    print("MULTI-IMAGE VERDICT (λ_tv = {})   |Δ| gate: > 0.01 normal, "
          "< 0.005 for low-w".format(args.lambda_tv))
    print("=" * 82)

    all_pass = True
    rows = []
    for stem in IMAGES:
        is_low = stem == LOW_W_STEM
        print(f"\n{stem}" + ("  (LOW-W)" if is_low else ""))
        img_pass = True
        for t in types:
            m = per_image[stem][t]
            id_ok = (m["delta"] < DELTA_MAX_LOW_W) if is_low \
                    else (m["delta"] > DELTA_MIN)
            # Confusion signal on low-w is by design uninteresting → skip ratio_w
            # and SI gates; only |Δ| < 0.005 matters.
            if is_low:
                rw_ok = True
                si_ok = True
            else:
                rw_ok = m["ratio_w"] > RATIO_W_MIN
                si_ok = m["SI"] < m["SI_thr"]
            l_ok = m["final_total"] < m["initial_total"]
            t_pass = id_ok and rw_ok and si_ok and l_ok
            img_pass = img_pass and t_pass
            marks = ("I" if id_ok else "i") \
                  + ("R" if rw_ok else "r") \
                  + ("S" if si_ok else "s") \
                  + ("L" if l_ok else "l")
            print(f"    {CVD_NAMES[t]:<14} |Δ|={m['delta']:.4f}  "
                  f"ratio_w={m['ratio_w']:.3f}  SI={m['SI']:.3f}  "
                  f"w̄={m['w_mean']:.3f}  [{marks}] "
                  f"{'PASS' if t_pass else 'FAIL'}")
            rows.append({
                "stem": stem, "type": t, "is_low_w": is_low,
                **m, "pass": t_pass,
                "pass_id": id_ok, "pass_rw": rw_ok,
                "pass_si": si_ok, "pass_loss": l_ok,
            })
        all_pass = all_pass and img_pass
        # Save failing-image PNGs
        if not img_pass and not is_low:
            for t in types:
                m = per_image[stem][t]
                id_ok = m["delta"] > DELTA_MIN
                rw_ok = m["ratio_w"] > RATIO_W_MIN
                si_ok = m["SI"] < m["SI_thr"]
                if not (id_ok and rw_ok and si_ok):
                    img_path = f"C:/Users/SCH/coco/val2017/{stem}.jpg"
                    orig = load_image(img_path, size=args.size).to(device)
                    r = optimize(
                        orig_srgb=orig, cvd_type=t, severity=1.0,
                        n_iters=args.iters, lr=args.lr, field_size=64,
                        use_lpips=False, log_every=args.iters,
                        init_direction="identity", init_beta=0.0,
                        lambda_tv=args.lambda_tv, lambda_n=0.15,
                        lambda_c=1.0, lambda_g=0.15,
                        lambda_excess=args.lambda_excess,
                    )
                    png = out_dir / f"validate_{stem}_{t}_mi_FAIL.png"
                    plot_result(r, t, png)

    print("\n" + ("*** MULTI-IMAGE PASS ***" if all_pass
                  else "*** MULTI-IMAGE FAIL — inspect failing images ***"))
    print(f"λ_tv = {args.lambda_tv}   images = {len(IMAGES)}   "
          f"low-w test = {LOW_W_STEM}")

    # Save aggregate JSON
    out_json = out_dir / f"multi_image_verdict_tv{str(args.lambda_tv).replace('.', '')}.json"
    with open(out_json, "w") as f:
        json.dump({
            "lambda_tv": args.lambda_tv,
            "iters": args.iters,
            "criteria": {
                "delta_min": DELTA_MIN,
                "delta_max_low_w": DELTA_MAX_LOW_W,
                "ratio_w_min": RATIO_W_MIN,
                "si_mult": SI_MULT,
                "si_baselines": si_base,
                "low_w_stem": LOW_W_STEM,
            },
            "rows": rows,
            "all_pass": all_pass,
        }, f, indent=2)
    print(f"Verdict → {out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lambda-tv", type=float, default=0.03)
    p.add_argument("--lambda-excess", type=float, default=0.0,
                   help="Asymmetric excess-contrast penalty inside L_c")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--out-dir", type=str,
                   default="C:/Users/SCH/graduation_project/outputs/v2_phase0")
    main(p.parse_args())
