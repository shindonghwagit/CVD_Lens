"""
작업 3 — 타일 추론 실험 (실험 전용, 서버/배포 코드 미변경).

가설: 작은 객체가 타일 내에서 상대적으로 커지면 16×16 grid 셀 점유율이 올라가 delta 희석이
완화된다. N∈{1,2,3} 타일 스윕으로 검증.

파이프라인(로컬):
  이미지를 N×N 겹치는 타일로 분할(overlap = 타일 크기의 25%) → 각 타일 letterbox 256 →
  ONNX 추론 → content-box delta를 타일 원해상도로 업샘플 → Hann(raised-cosine) feather
  overlap-add 로 전체 delta 맵 재조립 → 기존 guided filter → 원본에 더함.
  N=1 은 단일 타일=전체 이미지 → 현행 배포 경로와 동일(Hann 정규화로 상쇄).

평가(traffic_street p/d, sev 1.0): dilution(영역크기 vs |δ|, corr) + 작은빨강 ΔE00 +
전체지표(ΔE_mean/CRR/NP) + 타일 경계 아티팩트 시각화 + N별 추론 시간.

결론: tiling_effective / tiling_marginal / tiling_ineffective.
Run: py -m cvdlens_v2.tiled_infer
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

import cv2
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2 import infer_local as il
from cvdlens_v2.daily_test import sat, crr
from cvdlens_v2.artifact_probe import ciede2000, to_lab
from cvdlens_v2.diag_dilution import red_mask, components
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/tiled_inference"); OUT.mkdir(parents=True, exist_ok=True)
TARGETS = [("traffic_street", "p"), ("traffic_street", "d")]
NS = [1, 2, 3]
SEV = 1.0
OVERLAP = 0.25


def tile_bounds(L, n, overlap=OVERLAP):
    """n tiles along a length-L axis, overlapping by `overlap` of tile size.
    Returns list of (start,end). n=1 → whole axis. Defends L<tile / tiny images."""
    if n <= 1:
        return [(0, L)]
    t = int(round(L / (1 + (1 - overlap) * (n - 1))))
    t = max(1, min(L, t))
    if t >= L:                      # image too small to tile meaningfully
        return [(0, L)]
    stride = (L - t) / (n - 1)
    out = []
    for i in range(n):
        s = int(round(i * stride))
        e = min(L, s + t)
        s = max(0, e - t)
        if not out or (s, e) != out[-1]:
            out.append((s, e))
    return out


def _hann2d(h, w):
    """Raised-cosine window, >0 everywhere (half-sample offset), peak at center."""
    wy = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(h) + 0.5) / h)
    wx = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(w) + 0.5) / w)
    return np.outer(wy, wx).astype(np.float32)


def tiled_delta(img_f32, cvd_type, severity, n):
    """Reassembled native-resolution delta from N×N letterbox-256 tile inferences.
    Returns (delta_full, infer_seconds, boundary_xs, boundary_ys)."""
    H, W = img_f32.shape[:2]
    ys = tile_bounds(H, n); xs = tile_bounds(W, n)
    accum = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W), np.float32)
    t0 = time.perf_counter()
    for (y0, y1) in ys:
        for (x0, x1) in xs:
            tile = img_f32[y0:y1, x0:x1]
            th, tw = tile.shape[:2]
            lb, (bx0, by0, bx1, by1) = il._letterbox(tile, 256)
            out = il._run_float(lb, cvd_type, severity)
            d_box = (out - lb)[by0:by1, bx0:bx1]
            d_tile = cv2.resize(d_box, (tw, th), interpolation=cv2.INTER_LINEAR)
            win = _hann2d(th, tw)
            accum[y0:y1, x0:x1] += d_tile * win[..., None]
            wsum[y0:y1, x0:x1] += win
    dt = time.perf_counter() - t0
    # wsum > 0 everywhere (Hann half-offset); tiny floor only guards exact-zero.
    delta = accum / np.maximum(wsum[..., None], 1e-12)
    bxs = sorted({b for (a, b) in xs if b < W} | {a for (a, b) in xs if a > 0})
    bys = sorted({b for (a, b) in ys if b < H} | {a for (a, b) in ys if a > 0})
    return delta, dt, bxs, bys


def tiled_correct(img_f32, cvd_type, severity, n, use_guided=True):
    delta, dt, bxs, bys = tiled_delta(img_f32, cvd_type, severity, n)
    if use_guided:
        radius = max(1, max(img_f32.shape[:2]) // il.DEFAULT_RADIUS_DIVISOR)
        from guided import guided_filter
        delta = guided_filter(img_f32, delta, radius, il.DEFAULT_EPS)
    corrected = np.clip(img_f32 + delta, 0.0, 1.0)
    return dict(corrected=corrected, delta=delta, infer_s=dt, bxs=bxs, bys=bys)


def np_delta(corr, orig):
    return float(np.abs(corr - orig).mean())


def small_red_dE(de, comps):
    if not comps:
        return None
    areas = np.array([a for _, a, _ in comps]); med = np.median(areas)
    smask = np.zeros(de.shape, bool)
    for _, a, m in comps:
        if a < med:
            smask |= m
    return float(de[smask].mean())


def region_corr(delta, comps):
    dmag = np.linalg.norm(delta, axis=2)
    areas = np.array([a for _, a, _ in comps], float)
    mags = np.array([float(dmag[m].mean()) for _, a, m in comps])
    la = np.log10(areas)
    r = float(np.corrcoef(la, mags)[0, 1]) if len(areas) > 2 else float("nan")
    return r, areas, mags


def boundary_viz(name, t, img, res_by_n):
    """Delta magnitude map per N with tile boundaries overlaid + a horizontal profile."""
    fig, ax = plt.subplots(2, len(NS), figsize=(6 * len(NS), 9))
    for j, n in enumerate(NS):
        r = res_by_n[n]
        dmag = np.linalg.norm(r["delta"], axis=2)
        a = ax[0, j]; im = a.imshow(dmag, cmap="magma"); a.set_title(f"N={n} |delta| + tile bounds")
        for bx in r["bxs"]: a.axvline(bx, color="cyan", lw=0.6, alpha=0.7)
        for by in r["bys"]: a.axhline(by, color="cyan", lw=0.6, alpha=0.7)
        a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046)
        # horizontal profile through image mid-row
        row = dmag[dmag.shape[0] // 2]
        ax[1, j].plot(row, lw=0.7)
        for bx in r["bxs"]: ax[1, j].axvline(bx, color="r", ls=":", lw=0.8)
        ax[1, j].set_title(f"N={n} mid-row |delta| (red=tile boundary)")
        ax[1, j].set_xlabel("x")
    fig.suptitle(f"{name}_{t}  tile-boundary artifact check", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"{name}_{t}_boundary.png", dpi=90, bbox_inches="tight"); plt.close(fig)


def montage(name, t, img, res_by_n):
    fig, ax = plt.subplots(1, len(NS) + 1, figsize=(6 * (len(NS) + 1), 5))
    ax[0].imshow(img); ax[0].set_title("original"); ax[0].axis("off")
    for j, n in enumerate(NS):
        ax[j + 1].imshow(res_by_n[n]["corrected"]); ax[j + 1].set_title(f"corrected N={n}"); ax[j + 1].axis("off")
    fig.suptitle(f"{name}_{t}  tiled correction (guided ON)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"{name}_{t}_montage.png", dpi=80, bbox_inches="tight"); plt.close(fig)


def analyze(name, t):
    img = il.load_rgb(f"outputs/daily_test/{name}.jpg")
    rmask = red_mask(img); comps = components(rmask)
    res_by_n, rows = {}, []
    for n in NS:
        r = tiled_correct(img, t, SEV, n, use_guided=True)
        res_by_n[n] = r
        de = ciede2000(to_lab(img), to_lab(r["corrected"]))
        rc, _, _ = region_corr(r["delta"], comps)
        rows.append(dict(name=name, type=t, N=n,
                         deE_mean=round(float(de.mean()), 2), deE_max=round(float(de.max()), 2),
                         sat_delta=round(float(sat(r["corrected"]).mean() - sat(img).mean()), 4),
                         crr=round(crr(img, r["corrected"], t), 3),
                         np_delta=round(np_delta(r["corrected"], img), 4),
                         small_red_dE=round(small_red_dE(de, comps), 2),
                         corr_logarea_delta=round(rc, 3),
                         n_tiles=int(n * n), infer_s=round(r["infer_s"], 2)))
        print(f"[{name}_{t} N={n}] CRR={rows[-1]['crr']} ΔE={rows[-1]['deE_mean']} "
              f"smallRedΔE={rows[-1]['small_red_dE']} corr={rows[-1]['corr_logarea_delta']} "
              f"NP={rows[-1]['np_delta']} t={rows[-1]['infer_s']}s")
    montage(name, t, img, res_by_n)
    boundary_viz(name, t, img, res_by_n)
    return rows


def verdict(all_rows):
    """작은빨강 ΔE 상승 + 부작용 없음 → effective; 개선<추론비용 → marginal; 개선없음 → ineffective."""
    v = {}
    for (name, t) in TARGETS:
        rs = {r["N"]: r for r in all_rows if r["name"] == name and r["type"] == t}
        base, best = rs[1], rs[max(NS)]
        d_small = best["small_red_dE"] - base["small_red_dE"]
        rel = d_small / (base["small_red_dE"] + 1e-9)
        # side effects: overall CRR not worse, NP not worse (>+10%)
        crr_ok = best["crr"] >= base["crr"] - 0.005
        np_ok = best["np_delta"] <= base["np_delta"] * 1.10
        cost = best["infer_s"] / (base["infer_s"] + 1e-9)
        if rel > 0.15 and crr_ok and np_ok:
            lab = "tiling_effective"
        elif rel > 0.05:
            lab = "tiling_marginal"
        else:
            lab = "tiling_ineffective"
        v[f"{name}_{t}"] = dict(small_red_dE_base=base["small_red_dE"], small_red_dE_bestN=best["small_red_dE"],
                                small_red_rel=round(rel, 3), crr_base=base["crr"], crr_bestN=best["crr"],
                                np_base=base["np_delta"], np_bestN=best["np_delta"],
                                infer_cost_x=round(cost, 1), verdict=lab)
    return v


def main():
    all_rows = []
    for name, t in TARGETS:
        all_rows += analyze(name, t)
    v = verdict(all_rows)
    (OUT / "tiled.json").write_text(json.dumps(dict(rows=all_rows, verdict=v), indent=2), encoding="utf-8")
    _write_md(all_rows, v)
    print("\n[verdict]")
    for k, val in v.items():
        print(f"  {k}: {val['verdict']} (smallRedΔE {val['small_red_dE_base']}→{val['small_red_dE_bestN']}, "
              f"rel {val['small_red_rel']:+.2f}, cost {val['infer_cost_x']}x)")
    print(f"[save] {OUT/'tiled.json'} + REPORT.md")


def _write_md(rows, v):
    L = ["# 타일 추론 실험 (작업 3, 실험 전용·미배포)", "",
         "N×N 겹침 타일(overlap 25%), 타일별 letterbox 256 추론 → Hann feather overlap-add →",
         "guided filter. N=1은 현행 배포 경로와 동일. 지표 정의는 기존 스크립트 그대로.", "",
         "## N별 지표 (traffic_street, sev 1.0, guided ON)", "",
         "| case | N | tiles | ΔE_mean | CRR | NP\\|Δ\\| | 작은빨강ΔE | corr(logArea,\\|δ\\|) | 추론(s) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']}_{r['type']} | {r['N']} | {r['n_tiles']} | {r['deE_mean']} | {r['crr']} | "
                 f"{r['np_delta']} | {r['small_red_dE']} | {r['corr_logarea_delta']} | {r['infer_s']} |")
    L += ["", "## 결론 (케이스별)", ""]
    for k, val in v.items():
        L.append(f"### {k} → **{val['verdict']}**")
        L.append(f"- 작은빨강 ΔE00: N=1 {val['small_red_dE_base']} → N={max(NS)} {val['small_red_dE_bestN']} "
                 f"({val['small_red_rel']:+.0%})")
        L.append(f"- 부작용 점검: CRR {val['crr_base']}→{val['crr_bestN']}, NP {val['np_base']}→{val['np_bestN']}")
        L.append(f"- 추론 비용: {val['infer_cost_x']}× (N=1 대비)")
        L.append("")
    L += ["## 판정 기준", "- **tiling_effective**: 작은빨강 ΔE 유의 상승(>15%) + CRR/NP 부작용 없음 → 배포 후보",
          "- **tiling_marginal**: 개선(>5%) 있으나 추론 비용 대비 미미",
          "- **tiling_ineffective**: 개선 없음(≤5%) → grid 해상도가 근본 원인이라는 ablation 근거", "",
          "타일 경계 아티팩트: `*_boundary.png`(재조립 delta + 경계선 + mid-row 프로파일) 참조.", ""]
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
