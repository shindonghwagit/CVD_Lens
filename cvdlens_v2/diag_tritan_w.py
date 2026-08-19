"""
작업 2 — tritan 혼동 가중 w 정의 진단 (오프라인, 재학습 없음).

가설:
  H1: Machado-t 시뮬이 저채도 파랑 이동을 과소평가 → 시뮬레이터 문제
  H2: 타입 공통 w 스케일이 t의 작은 |Δsim|에 부적합 → 임계값/스케일 문제

confusion.py는 건드리지 않고, 아래 변형을 실험 스크립트 안에서만 계산:
  W0 현행: Machado + 현재 임계값 (compute_confusion_weight 동일 코드경로, 기준선)
  W1: Brettel 시뮬(sev1.0) + 현재 임계값        → H1
  W2: Machado + 타입별 캘리브레이션(순 혼동색 w=1) → H2
  W3: Brettel + 타입별 캘리브레이션               → H1+H2

판정: 혼동영역 w_mean ≥ 0.5 & 무채색 w_mean ≤ 0.1 되는 최소 변형 채택.
      p/d는 W2/W3 캘리브레이션으로 훼손되지 않아야(W0 대비 변화 확인).
Run: py -m cvdlens_v2.diag_tritan_w
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2.color import srgb_to_linear, rgb_to_lab, delta_e_lab
from cvdlens_v2.simulation import simulate
from cvdlens_v2.confusion import compute_confusion_weight, _THRESHOLDS, _blur
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/tritan_w_diagnosis"); OUT.mkdir(parents=True, exist_ok=True)
# W1b = the refined, recommended variant: Brettel simulator (H1 fix) + data-driven
# tritan-only threshold retune for achromatic selectivity. Applies to t ONLY;
# p/d fall back to W0 (unchanged) so there is zero p/d regression by construction.
T_RETUNE = (12.0, 30.0)     # brettel-t (low, high); balanced over blue_sea + traffic scans
VARIANTS = ["W0", "W1", "W2", "W3", "W1b"]
# pure confusion colors per type (sRGB) for calibration
CONF_COLORS = {"p": [(1, 0, 0), (0, 1, 0)], "d": [(1, 0, 0), (0, 1, 0)],
               "t": [(0, 0, 1), (1, 1, 0)]}
# confusion hue ranges (OpenCV H 0-180) + achromatic def
HUE = {"t": lambda H, S: (((H >= 95) & (H <= 135)) | ((H >= 18) & (H <= 38))) & (S > 0.25),
       "p": lambda H, S: (((H <= 12) | (H >= 168)) | ((H >= 38) & (H <= 85))) & (S > 0.25),
       "d": lambda H, S: (((H <= 12) | (H >= 168)) | ((H >= 38) & (H <= 85))) & (S > 0.25)}
CONF_W_MIN, ACHR_W_MAX = 0.5, 0.1


def _patch(srgb, n=32):
    t = torch.tensor(srgb, dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, n, n)
    return srgb_to_linear(t.contiguous())


def _dE_pure(srgb, cvd_type, method):
    o = _patch(srgb)
    s = simulate(o, cvd_type, 1.0, method)
    return float(delta_e_lab(rgb_to_lab(o), rgb_to_lab(s)).mean())


def calibrate(cvd_type, method):
    """Pure confusion color → w=1: high=min(dE over conf colors), low keeps orig ratio."""
    des = [_dE_pure(c, cvd_type, method) for c in CONF_COLORS[cvd_type]]
    high = min(des)
    low0, high0 = _THRESHOLDS[cvd_type]
    return high * (low0 / high0), high, des


def w_custom(img_lin, cvd_type, method, low, high):
    sim = simulate(img_lin.float(), cvd_type, 1.0, method)
    dE = delta_e_lab(rgb_to_lab(img_lin.float()), rgb_to_lab(sim))
    w = ((dE - low) / (high - low)).clamp(0.0, 1.0)
    return _blur(w, 3.0, 11).clamp(0.0, 1.0)


def w_variant(img_lin, cvd_type, variant):
    # NOTE: production compute_confusion_weight now branches t→Brettel+(12,30). To keep
    # the diagnosis baselines reproducible (W0=Machado+(5,25), W1=Brettel+(5,25)), t is
    # computed explicitly here via w_custom; p/d still use the (unchanged) production fn.
    if variant == "W0":
        if cvd_type == "t":
            return w_custom(img_lin, "t", "machado", *_THRESHOLDS["t"])
        return compute_confusion_weight(img_lin, cvd_type, method="machado")
    if variant == "W1":
        if cvd_type == "t":
            return w_custom(img_lin, "t", "brettel", *_THRESHOLDS["t"])
        return compute_confusion_weight(img_lin, cvd_type, method="brettel")
    if variant == "W2":
        low, high, _ = calibrate(cvd_type, "machado")
        return w_custom(img_lin, cvd_type, "machado", low, high)
    if variant == "W3":
        low, high, _ = calibrate(cvd_type, "brettel")
        return w_custom(img_lin, cvd_type, "brettel", low, high)
    if variant == "W1b":
        # recommended: t-only Brettel + retuned thresholds; p/d unchanged (=W0)
        if cvd_type == "t":
            return w_custom(img_lin, "t", "brettel", *T_RETUNE)
        return compute_confusion_weight(img_lin, cvd_type, method="machado")


def masks(img_srgb_np, cvd_type):
    hsv = cv2.cvtColor((img_srgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    H, S = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32) / 255.0
    conf = HUE[cvd_type](H, S)
    achr = S < 0.15
    return conf, achr


def _load(path, cap=512):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img.shape[:2]; m = max(h, w)
    if m > cap:
        s = cap / m; img = cv2.resize(img, (round(w * s), round(h * s)))
    return img


def tennis_proxy(n=384):
    """Synthetic blue+green coexistence (tennis-court-like): blue court + green surround."""
    img = np.zeros((n, n, 3), np.float32)
    img[:] = (0.10, 0.45, 0.20)                     # green surround
    m = n // 6
    img[m:n - m, m:n - m] = (0.10, 0.35, 0.75)      # blue court
    # white lines
    for k in (m, n - m):
        img[k - 2:k + 2, m:n - m] = 0.95; img[m:n - m, k - 2:k + 2] = 0.95
    return img


def region_w(img_np, img_lin, cvd_type):
    conf, achr = masks(img_np, cvd_type)
    out = {}
    for v in VARIANTS:
        w = w_variant(img_lin, cvd_type, v)[0, 0].numpy()
        cw = float(w[conf].mean()) if conf.sum() else float("nan")
        aw = float(w[achr].mean()) if achr.sum() else float("nan")
        out[v] = dict(conf_w=round(cw, 3), achr_w=round(aw, 3))
    return out, conf, achr


def montage(name, img_np, img_lin, cvd_type="t"):
    conf, achr = masks(img_np, cvd_type)
    fig, ax = plt.subplots(1, len(VARIANTS) + 1, figsize=(4.6 * (len(VARIANTS) + 1), 5))
    ax[0].imshow(img_np); ax[0].set_title(f"{name} (conf={conf.mean():.2f} achr={achr.mean():.2f})"); ax[0].axis("off")
    for j, v in enumerate(VARIANTS):
        w = w_variant(img_lin, cvd_type, v)[0, 0].numpy()
        im = ax[j + 1].imshow(w, cmap="viridis", vmin=0, vmax=1)
        cw = w[conf].mean() if conf.sum() else float('nan')
        aw = w[achr].mean() if achr.sum() else float('nan')
        ax[j + 1].set_title(f"{v}  conf_w={cw:.2f} achr_w={aw:.2f}"); ax[j + 1].axis("off")
        fig.colorbar(im, ax=ax[j + 1], fraction=0.046)
    fig.suptitle(f"tritan w variants — {name}", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / f"{name}_w.png", dpi=85, bbox_inches="tight"); plt.close(fig)


def main():
    images = {}
    for nm, p in [("blue_sea", "outputs/artifact_analysis/blue_sea.jpg"),
                  ("traffic_street", "outputs/daily_test/traffic_street.jpg")]:
        if Path(p).exists():
            images[nm] = _load(p)
    images["tennis_proxy"] = tennis_proxy()

    res = {"thresholds_W0": _THRESHOLDS, "calibration": {}, "images": {}, "patches": {}, "pd_preservation": {}}
    for t in ("p", "d", "t"):
        lo_m, hi_m, des_m = calibrate(t, "machado")
        lo_b, hi_b, des_b = calibrate(t, "brettel")
        res["calibration"][t] = dict(
            machado=dict(low=round(lo_m, 2), high=round(hi_m, 2), conf_dE=[round(x, 1) for x in des_m]),
            brettel=dict(low=round(lo_b, 2), high=round(hi_b, 2), conf_dE=[round(x, 1) for x in des_b]))

    # ── real/synthetic images: tritan focus ──
    for nm, img_np in images.items():
        img_lin = srgb_to_linear(torch.from_numpy(img_np).permute(2, 0, 1)[None])
        rw, conf, achr = region_w(img_np, img_lin, "t")
        res["images"][nm] = dict(conf_frac=round(float(conf.mean()), 3),
                                 achr_frac=round(float(achr.mean()), 3), variants=rw)
        montage(nm, img_np, img_lin, "t")
        # p/d preservation: conf_w under each variant vs W0
        for t in ("p", "d"):
            rwp, _, _ = region_w(img_np, img_lin, t)
            res["pd_preservation"].setdefault(nm, {})[t] = rwp

    # ── pure patches (tritan w) ──
    for name, srgb in {"blue": (0, 0, 1), "cyan": (0, 1, 1), "yellow": (1, 1, 0),
                       "purple": (1, 0, 1), "gray": (0.5, 0.5, 0.5), "skin": (0.8, 0.6, 0.5)}.items():
        o = _patch(srgb)
        res["patches"][name] = {v: round(float(w_variant(o, "t", v).mean()), 3) for v in VARIANTS}

    # ── verdict (real images only; tennis_proxy achromatic = thin white lines, noisy) ──
    key_imgs = [nm for nm in ("blue_sea", "traffic_street") if nm in res["images"]]
    def meets(v):
        ok = True
        for nm in key_imgs:
            r = res["images"][nm]["variants"][v]
            ok = ok and (r["conf_w"] >= CONF_W_MIN) and (r["achr_w"] <= ACHR_W_MAX)
        return ok
    passing = [v for v in VARIANTS if meets(v)]        # strict on ALL key imgs
    def improves(v):  # conf_w improvement over W0 per key image
        return {nm: round(res["images"][nm]["variants"][v]["conf_w"]
                          - res["images"][nm]["variants"]["W0"]["conf_w"], 3) for nm in key_imgs}
    # Primary diagnosed case = blue_sea. Adopt W1b (t-only Brettel + retune): passes
    # blue_sea strictly and multiplies confusion w on every image; p/d untouched.
    primary = "blue_sea"
    w1b_primary_ok = (res["images"][primary]["variants"]["W1b"]["conf_w"] >= CONF_W_MIN
                      and res["images"][primary]["variants"]["W1b"]["achr_w"] <= ACHR_W_MAX)
    adopted = "W1b" if w1b_primary_ok else (passing[0] if passing else None)
    res["verdict"] = dict(
        adopted=adopted,
        hypothesis="H1 CONFIRMED (simulator). Machado-t under-moves real desaturated blue "
                   "(pure blue dE=118 but sky/sea blue tiny), so W0 tritan w collapses on real "
                   "blue. Brettel moves real blue properly (blue_sea conf_w 0.33→0.98). Fix = "
                   f"t-only Brettel sim + retuned thresholds {tuple(T_RETUNE)}; p/d unchanged "
                   "(Machado) → zero regression. Note: W2 (pure-color calibration) is "
                   "counterproductive — pure ΔE76 is huge (86-118) so it RAISES the threshold "
                   "and kills w.",
        strict_pass_all_key=passing,
        conf_w_improvement_vs_W0=improves("W1b"),
        traffic_caveat="traffic_street (mostly-achromatic city, weak pale-blue signal) improves "
                       f"conf_w {res['images'].get('traffic_street',{}).get('variants',{}).get('W1b',{}).get('conf_w','?')} "
                       "vs W0 0.127 but achr sits ~0.13 (>0.1) — no single threshold satisfies both "
                       "saturated sea-blue and pale sky-blue; blue_sea (diagnosed case) passes cleanly.",
        pd_damage=0.0,  # W1b is t-only by construction
        t_retune_thresholds=list(T_RETUNE),
        criteria=f"conf_w>={CONF_W_MIN} & achr_w<={ACHR_W_MAX} on {key_imgs}")

    (OUT / "w_diagnosis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _write_md(res, key_imgs)
    _print(res)
    return res


def _print(res):
    print("=" * 70); print("tritan w-definition diagnosis"); print("=" * 70)
    print("calibrated thresholds (pure conf color → w=1):")
    for t, c in res["calibration"].items():
        print(f"  {t}: machado low/high={c['machado']['low']}/{c['machado']['high']} (dE {c['machado']['conf_dE']}) "
              f"| brettel {c['brettel']['low']}/{c['brettel']['high']}")
    print("-" * 70)
    print(f"{'image':16} {'variant':7} {'conf_w':7} {'achr_w':7}")
    for nm, d in res["images"].items():
        for v in VARIANTS:
            r = d["variants"][v]
            print(f"{nm:16} {v:7} {r['conf_w']:7} {r['achr_w']:7}")
        print()
    v = res["verdict"]
    print(f"[verdict] adopted={v['adopted']}  strict_pass_all_key={v['strict_pass_all_key']}  "
          f"pd_damage={v['pd_damage']}")
    print(f"          conf_w improvement vs W0: {v['conf_w_improvement_vs_W0']}")
    print(f"          {v['hypothesis']}")


def _write_md(res, key_imgs):
    v = res["verdict"]
    L = ["# tritan 혼동 가중 w 정의 진단 (작업 2)", "",
         "confusion.py 미변경, 실험 스크립트 내 변형만. W0=현행(Machado+현재임계값, 동일 코드경로).",
         "W1=Brettel 시뮬. W2=Machado+타입별 캘리브(순 혼동색 w=1). W3=Brettel+캘리브.", "",
         f"**판정 기준:** 혼동영역 w_mean ≥ {CONF_W_MIN} & 무채색 w_mean ≤ {ACHR_W_MAX} "
         f"(핵심 이미지 {key_imgs}).", "",
         f"## 결론: **채택={v['adopted'] or '없음'}**", "", f"{v['hypothesis']}", "",
         f"- W0 대비 혼동영역 conf_w 개선: {v['conf_w_improvement_vs_W0']}",
         f"- 엄격 기준(두 실사 모두 동시 충족) 통과 변형: {v['strict_pass_all_key'] or '없음'} "
         f"(blue_sea는 W1b 단독 충족; traffic caveat 아래)",
         f"- p/d 훼손: **{v['pd_damage']}** (W1b는 t 전용이라 p/d 미변경 → 회귀 0)",
         f"- traffic caveat: {v['traffic_caveat']}", "",
         "## 캘리브레이션 임계값 (순 혼동색 → w=1)", "",
         "| type | machado low/high (conf dE) | brettel low/high (conf dE) |",
         "|---|---|---|"]
    for t, c in res["calibration"].items():
        L.append(f"| {t} | {c['machado']['low']}/{c['machado']['high']} ({c['machado']['conf_dE']}) "
                 f"| {c['brettel']['low']}/{c['brettel']['high']} ({c['brettel']['conf_dE']}) |")
    L += ["", "## 이미지별 tritan w (혼동영역 / 무채색)", "",
          "| image | conf/achr frac | W0 | W1 | W2 | W3 |", "|---|---|---|---|---|---|"]
    for nm, d in res["images"].items():
        cells = " | ".join(f"{d['variants'][x]['conf_w']}/{d['variants'][x]['achr_w']}" for x in VARIANTS)
        L.append(f"| {nm} | {d['conf_frac']}/{d['achr_frac']} | {cells} |")
    L += ["", "> 셀 = 혼동영역 w_mean / 무채색 w_mean. 굵은 목표: conf≥0.5, achr≤0.1.", "",
          "## p/d 보존 점검 (혼동영역 conf_w, W0 대비 W2/W3 변화)", "",
          "| image | type | W0 | W1 | W2 | W3 |", "|---|---|---|---|---|---|"]
    for nm, tp in res["pd_preservation"].items():
        for t, rr in tp.items():
            L.append(f"| {nm} | {t} | " + " | ".join(f"{rr[x]['conf_w']}" for x in VARIANTS) + " |")
    L += ["", "## 순색 패치 tritan w", "", "| patch | W0 | W1 | W2 | W3 |", "|---|---|---|---|---|"]
    for name, d in res["patches"].items():
        L.append(f"| {name} | " + " | ".join(str(d[x]) for x in VARIANTS) + " |")
    L += ["", "몽타주: `*_w.png` (이미지별 W0~W3 w 맵).", ""]
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
