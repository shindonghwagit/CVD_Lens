"""
작업 2 — CRR<1 원인 진단 (direction_error vs magnitude_deficit vs both).

대상: food_tomatoes(d), traffic_street(p), traffic_street(d)  @ severity 1.0
      (몽타주에서 CRR<1로 관측된 케이스).

핵심 판정 = delta 스케일 스윕. corrected(α) = clip(img + α·delta) 의 CRR을
α∈{0,0.5,1,1.5,2}에서 계산한다(α=0 → CRR=1 by def). 모델이 보정을 delta로만
표현하므로(3장 파라미터화) α는 "보정 세기"를 그대로 스케일한다.
  • α↑ 에 CRR 단조 감소  → 더 보정할수록 나빠짐 → direction_error
  • α↑ 에 CRR 증가하고 α>1 에서 1 통과 → 방향 맞고 세기 부족 → magnitude_deficit
  • 증가하나 α=2 에서도 <1, 또는 부호 혼재 → both

보조:
  • delta의 혼동축 성분 <δ_lin, n_t> — 구조적으로 ≈0 임을 확인(왜 혼동축 부호가
    진단 정보가 못 되는지). 대신 가시 색축 <δ_lin, b_C> 부호 분포를 채도 마스크 안에서 출력.
  • sim(orig) vs sim(corr) 로컬 대비(라플라시안 에너지) 히스토그램 — 마스크 영역.

MEASURE ONLY. direction_error 판명 시 수정 금지(학습 문제), 리포트만.
Run: py -m cvdlens_v2.diag_crr
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
from cvdlens_v2 import infer_local as il
from cvdlens_v2.color import srgb_to_linear, rgb_to_lab
from cvdlens_v2.confusion import compute_confusion_weight
from cvdlens_v2.simulation import simulate
from cvdlens_v2.basis import get_confusion_dir, get_basis
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

OUT = Path("reports/crr_diag"); OUT.mkdir(parents=True, exist_ok=True)
TARGETS = [("food_tomatoes", "d"), ("traffic_street", "p"), ("traffic_street", "d")]
SEV = 1.0
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SAT_THR = 0.5


def _t(img):  # HWC f32 → (1,3,H,W)
    return torch.from_numpy(img).permute(2, 0, 1)[None].float()


def hsv_sat(img):
    mx = img.max(2); mn = img.min(2); return (mx - mn) / (mx + 1e-6)


def _cmag(x_lin):
    """Lab gradient magnitude on a linear-RGB image (matches daily_test.crr)."""
    lab = rgb_to_lab(x_lin)
    dy = lab[:, :, 1:] - lab[:, :, :-1]
    dx = lab[:, :, :, 1:] - lab[:, :, :, :-1]
    g = torch.zeros_like(lab[:, :1])
    g[:, :, 1:] += (dy ** 2).sum(1, keepdim=True)
    g[:, :, :, 1:] += (dx ** 2).sum(1, keepdim=True)
    return torch.sqrt(g + 1e-6)                       # (1,1,H,W)


def crr_masked(orig, corr, t, spatial_mask=None):
    """daily_test.crr definition; optional extra spatial mask (multiplies w)."""
    o = srgb_to_linear(_t(orig)); c = srgb_to_linear(_t(corr))
    w = compute_confusion_weight(o, t, 1.0)           # (1,1,H,W), machado
    if spatial_mask is not None:
        w = w * torch.from_numpy(spatial_mask.astype(np.float32))[None, None]
    so = simulate(o, t, 1.0, "brettel"); sc = simulate(c, t, 1.0, "brettel")
    ws = w.sum() + 1e-6
    num = (_cmag(sc) * w).sum() / ws
    den = (_cmag(so) * w).sum() / ws + 1e-6
    return float(num / den)


def laplacian_energy_map(img_lin_np):
    g = cv2.cvtColor((img_lin_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return lap ** 2


def diagnose(cat, t):
    img = il.load_rgb(f"outputs/daily_test/{cat}.jpg")
    r = il.correct(img, t, SEV, use_guided=False)     # current production path
    delta = r["delta_pre"]                            # sRGB-space upsampled delta
    corr = r["corrected"]
    sat = hsv_sat(img)
    mask = (sat > SAT_THR)
    tag = f"{cat}_{t}"

    # ── delta-scale sweep (masked-to-saturated CRR is the diagnostic axis) ──
    sweep = {}
    for a in ALPHAS:
        ca = np.clip(img + a * delta, 0.0, 1.0)
        sweep[a] = crr_masked(img, ca, t, spatial_mask=mask.astype(np.float32))
    crr_full = crr_masked(img, corr, t)               # full-image (comparable to daily_test)
    crr_mask1 = sweep[1.0]

    # ── confusion-axis (should be ~0) + visible chroma b_C sign, in linear RGB ──
    dlin = (srgb_to_linear(_t(corr)) - srgb_to_linear(_t(img)))[0].permute(1, 2, 0).numpy()  # (H,W,3)
    n = get_confusion_dir(t).numpy()                  # (3,)
    bC = get_basis(t)[1].numpy()                      # (3,) visible chroma axis
    m = mask & (np.linalg.norm(dlin, axis=2) > 1e-4)  # where there IS a delta, in sat region
    conf_comp = dlin @ n
    chroma_comp = dlin @ bC
    if m.sum() > 0:
        conf_abs_mean = float(np.abs(conf_comp[m]).mean())
        chroma_abs_mean = float(np.abs(chroma_comp[m]).mean())
        frac_pos = float((chroma_comp[m] > 0).mean())
    else:
        conf_abs_mean = chroma_abs_mean = frac_pos = float("nan")

    # ── local contrast (laplacian energy) hist: sim(orig) vs sim(corr) in mask ──
    o_lin = srgb_to_linear(_t(img)); c_lin = srgb_to_linear(_t(corr))
    from cvdlens_v2.color import linear_to_srgb
    so = linear_to_srgb(simulate(o_lin, t, 1.0, "brettel"))[0].permute(1, 2, 0).clamp(0, 1).numpy()
    sc = linear_to_srgb(simulate(c_lin, t, 1.0, "brettel"))[0].permute(1, 2, 0).clamp(0, 1).numpy()
    le_o = laplacian_energy_map(so)[mask]; le_c = laplacian_energy_map(sc)[mask]
    le_ratio = float((le_c.mean() + 1e-9) / (le_o.mean() + 1e-9))

    # ── verdict ──
    s = [sweep[a] for a in ALPHAS]                    # s[0]=1.0 (α=0)
    increasing = s[4] > s[2] > s[1]                   # α=2 > α=1 > α=0.5
    decreasing = s[4] < s[2] < s[1]
    if decreasing and s[2] < 1.0:
        verdict = "direction_error"
    elif increasing and s[4] > 1.0:
        verdict = "magnitude_deficit"
    elif increasing and s[4] <= 1.0:
        verdict = "magnitude_deficit(weak)"           # right way but too weak even at 2x
    else:
        verdict = "both"

    rec = dict(tag=tag, cat=cat, type=t, sev=SEV,
               crr_full=round(crr_full, 3), crr_mask_a1=round(crr_mask1, 3),
               sweep={f"a{a}": round(sweep[a], 3) for a in ALPHAS},
               conf_axis_abs_mean=round(conf_abs_mean, 6),
               chroma_axis_abs_mean=round(chroma_abs_mean, 6),
               chroma_frac_positive=round(frac_pos, 3),
               sim_lap_energy_ratio=round(le_ratio, 3),
               mask_px=int(mask.sum()), verdict=verdict)

    # ── figure ──
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax[0, 0].imshow(img); ax[0, 0].set_title(f"{tag} original"); ax[0, 0].axis("off")
    ax[0, 1].imshow(corr); ax[0, 1].set_title("corrected (α=1, no guided)"); ax[0, 1].axis("off")
    ax[0, 2].imshow(mask, cmap="gray"); ax[0, 2].set_title(f"sat>{SAT_THR} mask ({mask.sum()} px)"); ax[0, 2].axis("off")
    ax[1, 0].plot(ALPHAS, s, "o-"); ax[1, 0].axhline(1.0, color="r", ls="--")
    ax[1, 0].set_title(f"masked CRR vs δ-scale  →  {verdict}")
    ax[1, 0].set_xlabel("α (delta scale)"); ax[1, 0].set_ylabel("CRR (sat mask)")
    ax[1, 1].hist(chroma_comp[m], bins=60, color="teal")
    ax[1, 1].axvline(0, color="k", lw=0.8)
    ax[1, 1].set_title(f"<δ_lin, b_C> in mask  (pos frac {frac_pos:.2f})")
    ax[1, 2].hist(np.log10(le_o + 1e-6), bins=60, alpha=0.6, label="sim(orig)")
    ax[1, 2].hist(np.log10(le_c + 1e-6), bins=60, alpha=0.6, label="sim(corr)")
    ax[1, 2].set_title(f"local contrast (log lap E) ratio={le_ratio:.2f}")
    ax[1, 2].legend(fontsize=8); ax[1, 2].set_xlabel("log10 laplacian energy")
    fig.suptitle(f"[CRR<1 진단] {tag}  full CRR={crr_full:.3f}  mask CRR(α1)={crr_mask1:.3f}  "
                 f"conf-axis|·|={conf_abs_mean:.1e}(≈0 구조적)  →  {verdict}", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"{tag}_crrdiag.png", dpi=100, bbox_inches="tight"); plt.close(fig)

    print(f"[{tag}] full={crr_full:.3f} mask(α1)={crr_mask1:.3f} sweep={[round(x,3) for x in s]} "
          f"chroma+frac={frac_pos:.2f} lapE_ratio={le_ratio:.2f} → {verdict}")
    return rec


def main():
    recs = [diagnose(cat, t) for cat, t in TARGETS]
    (OUT / "crr_diag.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
    print(f"\n[save] {OUT/'crr_diag.json'} ({len(recs)} cases)")
    return recs


if __name__ == "__main__":
    main()
