"""
작업 3 — 작은 영역 delta 희석 정량화 (traffic_street).

가설: 16×16 bilateral grid 셀 내 공간 평균화 때문에, 작은 채도 빨강 객체(신호등 등)는
delta가 희석돼 거의 보정되지 않는다. 영역 크기(px)가 작을수록 평균 |delta|가 작으면
'grid 공간 평균화' 가설 확정.

방법:
  • 채도 빨강 마스크(HSV: sat>0.4, hue∈빨강대)를 connected components 로 분리.
  • 각 컴포넌트의 area(px) vs 평균 |delta| 산점도 — guided filter 적용 전/후 각각.
  • area↔|delta| 상관(로그area 기준). 후처리가 작은 영역을 얼마나 살리는지 비교.

Run: py -m cvdlens_v2.diag_dilution
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import cv2
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2 import infer_local as il
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/dilution"); OUT.mkdir(parents=True, exist_ok=True)
TARGETS = [("traffic_street", "p"), ("traffic_street", "d")]
SEV = 1.0
MIN_AREA = 12          # ignore specks smaller than this (px)


def red_mask(img_f32):
    """Saturated red: HSV sat>0.4 and hue in red band (OpenCV H in 0-180)."""
    hsv = cv2.cvtColor((img_f32 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    H, S = hsv[..., 0], hsv[..., 1] / 255.0
    red = ((H <= 12) | (H >= 168)) & (S > 0.4)
    return red.astype(np.uint8)


def components(mask, min_area=MIN_AREA):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out.append((i, area, lab == i))
    return out


def region_deltas(delta, comps):
    dmag = np.linalg.norm(delta, axis=2)
    return [(area, float(dmag[m].mean())) for _, area, m in comps]


def analyze(cat, t):
    img = il.load_rgb(f"outputs/daily_test/{cat}.jpg")
    off = il.correct(img, t, SEV, use_guided=False)
    on = il.correct(img, t, SEV, use_guided=True)
    mask = red_mask(img)
    comps = components(mask)
    tag = f"{cat}_{t}"
    if not comps:
        print(f"[{tag}] no red components ≥{MIN_AREA}px"); return None

    d_off = region_deltas(off["delta_pre"], comps)     # before guided
    d_on = region_deltas(on["delta_post"], comps)      # after guided
    areas = np.array([a for a, _ in d_off], float)
    mag_off = np.array([m for _, m in d_off])
    mag_on = np.array([m for _, m in d_on])

    la = np.log10(areas)
    corr_off = float(np.corrcoef(la, mag_off)[0, 1]) if len(areas) > 2 else float("nan")
    corr_on = float(np.corrcoef(la, mag_on)[0, 1]) if len(areas) > 2 else float("nan")

    # small vs large split at median area — how much does guided lift small regions?
    med = np.median(areas)
    sm = areas < med; lg = ~sm
    small_off, small_on = float(mag_off[sm].mean()), float(mag_on[sm].mean())
    large_off, large_on = float(mag_off[lg].mean()), float(mag_on[lg].mean())

    rec = dict(tag=tag, cat=cat, type=t, n_regions=len(comps),
               area_min=int(areas.min()), area_max=int(areas.max()), area_median=float(med),
               corr_logarea_vs_delta_off=round(corr_off, 3),
               corr_logarea_vs_delta_on=round(corr_on, 3),
               small_region_mean_delta_off=round(small_off, 5),
               small_region_mean_delta_on=round(small_on, 5),
               small_region_guided_lift=round((small_on / (small_off + 1e-9) - 1) * 100, 1),
               large_region_mean_delta_off=round(large_off, 5),
               large_region_mean_delta_on=round(large_on, 5),
               dilution_confirmed=bool(corr_off > 0.2))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].imshow(img); ys, xs = np.where(mask > 0)
    ax[0].scatter(xs, ys, s=0.2, c="cyan", alpha=0.3)
    ax[0].set_title(f"{tag}  red mask ({len(comps)} regions ≥{MIN_AREA}px)"); ax[0].axis("off")
    ax[1].scatter(areas, mag_off, s=18, c="crimson", label=f"guided OFF (r={corr_off:+.2f})")
    ax[1].scatter(areas, mag_on, s=18, c="seagreen", marker="^", label=f"guided ON (r={corr_on:+.2f})")
    ax[1].set_xscale("log"); ax[1].set_xlabel("region area (px, log)"); ax[1].set_ylabel("mean |delta|")
    ax[1].set_title("region size vs mean |delta|  (up-right slope = 작은영역 희석)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    fig.suptitle(f"[작은영역 delta 희석] {tag}  corr(logArea,|δ|) off={corr_off:+.2f} on={corr_on:+.2f}  "
                 f"small-region guided lift {rec['small_region_guided_lift']:+.1f}%", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"{tag}_dilution.png", dpi=100, bbox_inches="tight"); plt.close(fig)

    print(f"[{tag}] n={len(comps)} corr(logArea,|δ|)off={corr_off:+.2f} on={corr_on:+.2f} "
          f"small|δ| {small_off:.4f}→{small_on:.4f} ({rec['small_region_guided_lift']:+.1f}%) "
          f"dilution={'YES' if rec['dilution_confirmed'] else 'no'}")
    return rec


def main():
    recs = [r for cat, t in TARGETS if (r := analyze(cat, t)) is not None]
    (OUT / "dilution.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
    print(f"\n[save] {OUT/'dilution.json'} ({len(recs)} cases)")
    return recs


if __name__ == "__main__":
    main()
