"""
작업 4 — guided filter on/off before/after 재평가.

기존 지표 정의를 그대로 import(daily_test.sat/crr, artifact_probe.ciede2000/to_lab,
step3 NP|Δ|)해 비교 가능성 유지. 지표: ΔE00 mean/max, Δsat, CRR, NP|Δ|.
로컬 ONNX 추론(guided on/off) — 배포 API의 JPEG 왕복은 제외(raw 비교), 단 CRR<1
케이스는 JPEG q92 CRR도 병기(배포 실측 재현).

케이스: food_tomatoes, nature_autumn, traffic_street, skin_portrait + stop 표지판
        × p/d/t × s0.6/s1.0 × guided{off,on}.

특별 점검:
  (a) stop 빨간 링 ΔE 균일도(CoV=std/mean) 감소하는지
  (b) traffic 신호등(작은 빨강) ΔE 상승하는지
  (c) skin_portrait NP|Δ| 악화 안 되는지
  (d) CRR<1(food_d, traffic_p, traffic_d)가 1 이상 회복되는지

Run: py -m cvdlens_v2.reeval_guided
"""
from __future__ import annotations
import sys, io, json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvdlens_v2 import infer_local as il
from cvdlens_v2.daily_test import sat, crr
from cvdlens_v2.artifact_probe import ciede2000, to_lab
from cvdlens_v2.diag_dilution import red_mask, components
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/guided_reeval"); OUT.mkdir(parents=True, exist_ok=True)
IMGS = {  # name → path
    "food_tomatoes":  "outputs/daily_test/food_tomatoes.jpg",
    "nature_autumn":  "outputs/daily_test/nature_autumn.jpg",
    "traffic_street": "outputs/daily_test/traffic_street.jpg",
    "skin_portrait":  "outputs/daily_test/skin_portrait.jpg",
    "stop_sign":      "outputs/artifact_analysis/stop_sign.jpg",
}
TYPES = ["p", "d", "t"]
SEVS = [0.6, 1.0]
CRR_SUB1 = {("food_tomatoes", "d"), ("traffic_street", "p"), ("traffic_street", "d")}


def np_delta(corr, orig):
    return float(np.abs(corr - orig).mean())


def crr_jpeg(orig, corr, t):
    u8 = (corr * 255 + 0.5).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(u8).save(buf, format="JPEG", quality=92)
    cj = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")).astype(np.float32) / 255.0
    return crr(orig, cj, t)


def metrics(orig, corr, t):
    de = ciede2000(to_lab(orig), to_lab(corr))
    return dict(deE_mean=round(float(de.mean()), 2), deE_max=round(float(de.max()), 2),
                sat_delta=round(float(sat(corr).mean() - sat(orig).mean()), 4),
                crr=round(crr(orig, corr, t), 3), np_delta=round(np_delta(corr, orig), 4)), de


def montage(name, t, sv, orig, c_off, c_on, de_off, de_on):
    vmax = float(max(de_off.max(), de_on.max()))
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.6))
    ax[0].imshow(orig); ax[0].set_title("original"); ax[0].axis("off")
    ax[1].imshow(c_off); ax[1].set_title("corrected OFF"); ax[1].axis("off")
    ax[2].imshow(c_on); ax[2].set_title("corrected ON (guided)"); ax[2].axis("off")
    im = ax[3].imshow(de_off, cmap="inferno", vmax=vmax); ax[3].set_title(f"ΔE OFF (max {de_off.max():.1f})"); ax[3].axis("off")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    im2 = ax[4].imshow(de_on, cmap="inferno", vmax=vmax); ax[4].set_title(f"ΔE ON (max {de_on.max():.1f})"); ax[4].axis("off")
    fig.colorbar(im2, ax=ax[4], fraction=0.046)
    fig.suptitle(f"{name}_{t}_s{sv}   guided on/off", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"{name}_{t}_s{sv}.png", dpi=80, bbox_inches="tight"); plt.close(fig)


def cov_in_mask(de, mask):
    v = de[mask > 0]
    return float(v.std() / (v.mean() + 1e-9)) if v.size else float("nan")


def main():
    rows = []
    special = {"a_stop_ring_cov": {}, "b_traffic_smallred_dE": {}, "c_skin_npdelta": {}, "d_crr_recovery": {}}

    for name, path in IMGS.items():
        img = il.load_rgb(path)
        rmask = red_mask(img) if name in ("stop_sign", "traffic_street") else None
        comps = components(rmask) if rmask is not None else None
        for t in TYPES:
            for sv in SEVS:
                off = il.correct(img, t, sv, use_guided=False)
                on = il.correct(img, t, sv, use_guided=True)
                m_off, de_off = metrics(img, off["corrected"], t)
                m_on, de_on = metrics(img, on["corrected"], t)
                for tag, m in [("off", m_off), ("on", m_on)]:
                    rows.append(dict(name=name, type=t, sev=sv, guided=tag, **m))
                montage(name, t, sv, img, off["corrected"], on["corrected"], de_off, de_on)

                # ── special checks at s1.0 (the reported problem severity) ──
                if sv == 1.0:
                    if name == "stop_sign":
                        special["a_stop_ring_cov"][t] = dict(
                            off=round(cov_in_mask(de_off, rmask), 3),
                            on=round(cov_in_mask(de_on, rmask), 3))
                    if name == "traffic_street" and comps:
                        # smallest-half red regions mean ΔE
                        areas = np.array([a for _, a, _ in comps])
                        med = np.median(areas)
                        small = [m for _, a, m in comps if a < med]
                        smask = np.zeros(img.shape[:2], bool)
                        for m_ in small: smask |= m_
                        special["b_traffic_smallred_dE"][t] = dict(
                            off=round(float(de_off[smask].mean()), 2),
                            on=round(float(de_on[smask].mean()), 2))
                    if name == "skin_portrait":
                        special["c_skin_npdelta"][t] = dict(off=m_off["np_delta"], on=m_on["np_delta"])
                    if (name, t) in CRR_SUB1:
                        special["d_crr_recovery"][f"{name}_{t}"] = dict(
                            crr_off=m_off["crr"], crr_on=m_on["crr"],
                            crr_off_jpeg=round(crr_jpeg(img, off["corrected"], t), 3),
                            crr_on_jpeg=round(crr_jpeg(img, on["corrected"], t), 3))
                print(f"[{name}_{t}_s{sv}] CRR {m_off['crr']}→{m_on['crr']}  "
                      f"ΔE {m_off['deE_mean']}→{m_on['deE_mean']}  NP {m_off['np_delta']}→{m_on['np_delta']}")

    (OUT / "reeval.json").write_text(json.dumps(dict(rows=rows, special=special), indent=2), encoding="utf-8")
    _write_markdown(rows, special)
    print(f"\n[save] {OUT/'reeval.json'} + REPORT.md ({len(rows)} rows)")


def _write_markdown(rows, special):
    lines = ["# guided filter on/off 재평가 (작업 4)", "",
             "로컬 ONNX 추론, guided filter on/off. 지표 정의는 기존 스크립트 그대로",
             "(daily_test.sat/crr, artifact_probe.ciede2000/to_lab, NP|Δ|=mean|corr−orig|).",
             "CRR은 기존 정의대로 sim severity 1.0에서 평가(보정 severity와 무관).", "",
             "## 전체 지표 표 (off → on)", "",
             "| image | type | sev | ΔE_mean | ΔE_max | Δsat | CRR | NP\\|Δ\\| |",
             "|---|---|---|---|---|---|---|---|"]
    by = {}
    for r in rows:
        by.setdefault((r["name"], r["type"], r["sev"]), {})[r["guided"]] = r
    for (name, t, sv), pair in by.items():
        o, n = pair["off"], pair["on"]
        lines.append(f"| {name} | {t} | {sv} | {o['deE_mean']}→{n['deE_mean']} | "
                     f"{o['deE_max']}→{n['deE_max']} | {o['sat_delta']:+.3f}→{n['sat_delta']:+.3f} | "
                     f"{o['crr']}→{n['crr']} | {o['np_delta']}→{n['np_delta']} |")

    lines += ["", "## 특별 점검 (severity 1.0)", "",
              "### (a) stop 표지판 빨간 링 ΔE 균일도 — CoV=std/mean (낮을수록 균일)"]
    for t, v in special["a_stop_ring_cov"].items():
        arrow = "↓개선" if v["on"] < v["off"] else "↑악화"
        lines.append(f"- type {t}: CoV {v['off']} → {v['on']}  ({arrow})")
    lines += ["", "### (b) traffic 신호등(작은 빨강 영역) 평균 ΔE — 상승 기대"]
    for t, v in special["b_traffic_smallred_dE"].items():
        arrow = "↑상승" if v["on"] > v["off"] else "↓하락"
        lines.append(f"- type {t}: 작은빨강 ΔE {v['off']} → {v['on']}  ({arrow})")
    lines += ["", "### (c) skin_portrait NP|Δ| — 악화 안 돼야 함"]
    for t, v in special["c_skin_npdelta"].items():
        arrow = "악화" if v["on"] > v["off"] + 0.0005 else "유지/개선"
        lines.append(f"- type {t}: NP|Δ| {v['off']} → {v['on']}  ({arrow})")
    lines += ["", "### (d) CRR<1 케이스 회복 — raw / JPEG q92(배포 재현)"]
    for k, v in special["d_crr_recovery"].items():
        lines.append(f"- {k}: raw {v['crr_off']}→{v['crr_on']}  |  jpeg {v['crr_off_jpeg']}→{v['crr_on_jpeg']}")
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
