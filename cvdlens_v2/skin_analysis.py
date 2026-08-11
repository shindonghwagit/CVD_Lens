"""
Skin-tone shift diagnosis for real-photo (portrait) CVD correction.

Question under test (MEASURE ONLY — no correction change here):
    "How much, and why, does skin move under CVDLens correction, and how much
     does lowering the default severity (1.0 → 0.7 → 0.5) reduce it — at what
     cost to overall contrast recovery (CRR)?"

Pipeline:
    1. Select 5 clear-portrait images from COCO val2017 by skin-mask coverage
       (deterministic sample; stems printed first).
    2. Skin mask: YCbCr range approximation (77≤Cb≤127, 133≤Cr≤173), 1× erosion
       to drop boundary noise. Mask montage saved for visual QA.
    3. Measure, for P and D on model_best:
       (a) mean w in skin vs non-skin   — does skin actually trigger confusion?
       (b) skin |Δ| at severity {1.0,0.7,0.5} — does lowering default help?
       (c) whole-image CRR (ratio_w) at the same severities — recovery cost.
    4. Outputs: outputs/v2_phase3/skin_analysis.json, _summary.md (table +
       options memo), skin_masks.png, skin_severity_compare.png.

Run: py -m cvdlens_v2.skin_analysis
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.model import CVDCorrectionNet
from cvdlens_v2.color import srgb_to_linear
from cvdlens_v2 import step3_metrics as M

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VAL_DIR = Path("C:/Users/SCH/coco/val2017")
CKPT = Path("outputs/phase3_kaggle/v2_phase1/model_best.pt")
OUT_DIR = Path("outputs/v2_phase3")
SIZE = 256
SEVERITIES = [1.0, 0.7, 0.5]
TYPES = ["p", "d"]
SELECT_SAMPLE = 1200       # how many val stems to scan for selection
SELECT_SEED = 20260811
N_SELECT = 5
COV_BAND = (0.06, 0.45)    # skin coverage band: face-sized skin, not a full skin-color frame

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def n_faces(srgb: torch.Tensor) -> int:
    """Haar frontal-face count (selection only — not used in measurement)."""
    bgr = (srgb[0].permute(1, 2, 0).numpy()[:, :, ::-1] * 255).astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=9, minSize=(40, 40))
    return len(faces)


# ── image + skin mask ────────────────────────────────────────────────────
def load_srgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)      # (1,3,H,W) sRGB [0,1]


def skin_mask(srgb: torch.Tensor) -> torch.Tensor:
    """
    YCbCr (BT.601) skin-range mask + 1× erosion. Returns (1,1,H,W) float {0,1}.
        skin ⇔ 77 ≤ Cb ≤ 127  and  133 ≤ Cr ≤ 173   (standard 8-bit range)
    """
    r, g, b = srgb[:, 0:1], srgb[:, 1:2], srgb[:, 2:3]     # [0,1]
    R, G, B = r * 255.0, g * 255.0, b * 255.0
    Cb = 128.0 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128.0 + 0.5 * R - 0.418688 * G - 0.081312 * B
    m = ((Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173)).float()
    # erosion(m) = 1 - dilate(1-m);  dilate = 3×3 max-pool (1 iteration)
    eroded = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
    return eroded


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of per-pixel scalar x over mask==1. x:(1,1,H,W) or (1,C,H,W)."""
    if x.shape[1] > 1:
        x = x.abs().mean(dim=1, keepdim=True)             # channel-mean magnitude
    denom = mask.sum().item()
    if denom < 1:
        return float("nan")
    return (x * mask).sum().item() / denom


# ── selection ────────────────────────────────────────────────────────────
def select_portraits() -> list[dict]:
    stems = sorted(p.stem for p in VAL_DIR.glob("*.jpg"))
    rng = np.random.default_rng(SELECT_SEED)
    sample = sorted(rng.choice(len(stems), size=min(SELECT_SAMPLE, len(stems)),
                               replace=False).tolist())
    scored = []
    for i in sample:
        stem = stems[i]
        srgb = load_srgb(VAL_DIR / f"{stem}.jpg")
        cov = skin_mask(srgb).mean().item()
        if not (COV_BAND[0] <= cov <= COV_BAND[1]):
            continue                          # skip full-frame skin-color false positives
        nf = n_faces(srgb)
        if nf < 1:
            continue                          # require a detected face → genuine portrait
        # rank: more faces first, then more skin (within band)
        scored.append((nf, cov, stem))
    scored.sort(reverse=True)
    top = scored[:N_SELECT]
    print(f"[select] scanned {len(sample)} stems (seed {SELECT_SEED}); "
          f"{len(scored)} portrait candidates (face≥1, cov∈{COV_BAND}); "
          f"top-{N_SELECT}:")
    picks = []
    for nf, cov, stem in top:
        print(f"    {stem}   faces={nf}  skin_coverage={cov:.3f}")
        picks.append({"stem": stem, "skin_coverage": round(cov, 4), "faces": nf})
    return picks


# ── mask QA montage ──────────────────────────────────────────────────────
def save_mask_montage(picks: list[dict]):
    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for j, p in enumerate(picks):
        srgb = load_srgb(VAL_DIR / f"{p['stem']}.jpg")
        m = skin_mask(srgb)
        img = srgb[0].permute(1, 2, 0).numpy()
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{p['stem']}\ncov={p['skin_coverage']:.3f} faces={p.get('faces','?')}", fontsize=8)
        overlay = img.copy()
        mk = m[0, 0].numpy() > 0.5
        overlay[mk] = 0.5 * overlay[mk] + 0.5 * np.array([0.0, 1.0, 0.0])
        axes[1, j].imshow(overlay)
        axes[1, j].set_title("skin mask (green)", fontsize=8)
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")
    fig.suptitle("Skin-mask QA (top: original, bottom: YCbCr mask ∘ erosion)", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "skin_masks.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


# ── severity comparison PNG (representative image, type P) ───────────────
def save_severity_compare(net, stem: str, cvd_type: str = "p"):
    srgb = load_srgb(VAL_DIR / f"{stem}.jpg")
    panels = [("original", srgb[0].permute(1, 2, 0).numpy())]
    for sev in [0.5, 0.7, 1.0]:
        with torch.no_grad():
            r = net(srgb, cvd_type=cvd_type, severity=sev)
        panels.append((f"severity {sev}", r["out_srgb"][0].permute(1, 2, 0).clamp(0, 1).numpy()))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (title, im) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Severity sweep — {stem} · type {cvd_type.upper()} "
                 f"(default lowered 1.0→0.7 for real photos)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "skin_severity_compare.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


# ── main measurement ─────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    picks = select_portraits()
    save_mask_montage(picks)

    net = CVDCorrectionNet(pretrained_backbone=False).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    per_image = []
    for p in picks:
        stem = p["stem"]
        srgb = load_srgb(VAL_DIR / f"{stem}.jpg")
        orig_lin = srgb_to_linear(srgb)
        mask = skin_mask(srgb)
        nonskin = 1.0 - mask
        rec = {"stem": stem, "skin_coverage": p["skin_coverage"],
               "skin_px": int(mask.sum().item()), "types": {}}
        for t in TYPES:
            trec = {"w_skin_vs_nonskin": {}, "skin_delta": {}, "crr_ratiow": {}}
            for sev in SEVERITIES:
                with torch.no_grad():
                    r = net(srgb, cvd_type=t, severity=sev)
                w = r["w"]                                  # (1,1,H,W) at this sev
                out_srgb = r["out_srgb"]
                # (a) w skin vs non-skin (report all sev; sev=1.0 is the reference)
                trec["w_skin_vs_nonskin"][str(sev)] = {
                    "skin": round(masked_mean(w, mask), 5),
                    "nonskin": round(masked_mean(w, nonskin), 5),
                }
                # (b) skin |Δ| (mean sRGB abs change over skin)
                dabs = (out_srgb - srgb).abs().mean(dim=1, keepdim=True)
                trec["skin_delta"][str(sev)] = round(masked_mean(dabs, mask), 5)
                # (c) whole-image CRR
                trec["crr_ratiow"][str(sev)] = round(
                    M.crr_ratio_w(r["out_linear"], orig_lin, w, t, sev), 5)
            rec["types"][t] = trec
        per_image.append(rec)
        print(f"[measure] {stem} done")

    # ── aggregates over the 5 images ─────────────────────────────────────
    def agg(getter):
        vals = [getter(rec) for rec in per_image]
        vals = [v for v in vals if v == v]  # drop NaN
        return round(float(np.mean(vals)), 5) if vals else float("nan")

    aggregates = {}
    for t in TYPES:
        aggregates[t] = {
            "w_skin_sev1.0": agg(lambda r: r["types"][t]["w_skin_vs_nonskin"]["1.0"]["skin"]),
            "w_nonskin_sev1.0": agg(lambda r: r["types"][t]["w_skin_vs_nonskin"]["1.0"]["nonskin"]),
            "skin_delta": {s: agg(lambda r, s=s: r["types"][t]["skin_delta"][str(s)])
                           for s in SEVERITIES},
            "crr_ratiow": {s: agg(lambda r, s=s: r["types"][t]["crr_ratiow"][str(s)])
                           for s in SEVERITIES},
        }

    rep_stem = picks[0]["stem"]
    save_severity_compare(net, rep_stem, "p")

    result = {
        "meta": {
            "purpose": "skin-tone shift diagnosis (measure only; no correction change)",
            "checkpoint": str(CKPT), "size": SIZE,
            "skin_mask": "YCbCr 77<=Cb<=127, 133<=Cr<=173, 1x erosion",
            "select_seed": SELECT_SEED, "select_sample": SELECT_SAMPLE,
            "severities": SEVERITIES, "types": TYPES,
            "representative_image": rep_stem,
        },
        "selected_stems": [p["stem"] for p in picks],
        "per_image": per_image,
        "aggregates": aggregates,
        "decision_options": _decision_memo(aggregates),
    }
    out_json = OUT_DIR / "skin_analysis.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {out_json}")

    _write_summary_md(result)
    _print_table(aggregates)


def _decision_memo(agg: dict) -> dict:
    """Options memo (documentation only — decision is the user's)."""
    return {
        "note": "구현 금지. 아래 수치는 aggregates 기반. 결정은 사용자.",
        "A_default_0.7_only": {
            "desc": "severity 0.7 기본값만으로 충분 → 종결.",
            "evidence": "aggregates.<type>.skin_delta['1.0'] vs ['0.7'] 감소폭, "
                        "그 대가는 crr_ratiow['1.0'] vs ['0.7'] 하락폭으로 판단.",
        },
        "B_inference_skin_attenuation": {
            "desc": "추론단 피부 감쇠 옵션: delta에 피부 마스크 감쇠(예: 1-α·skin_mask)를 "
                    "곱해 피부 이동만 줄인다. 기본 OFF, 논문 평가(severity 1.0, 감쇠 없음)와 "
                    "분리 명시.",
            "expected_effect": "skin_delta를 α배로 직접 감소(피부 국한). 비피부 회복은 불변.",
            "cost": "추론단 YCbCr 마스크 계산 1회(경량) + ONNX 그래프에 마스크 분기 추가. "
                    "마스크 오탐(나무/모래)에서 국소 저보정 위험 → mask QA 필요.",
        },
        "C_paper_future_work_only": {
            "desc": "논문 §7 향후 연구로만 서술(피부 인지 자연스러움 항). 코드 변경 없음.",
            "evidence": "본 skin_analysis 수치를 근거 문단으로 인용.",
        },
    }


def _fmt_row(label, d):
    return f"| {label} | " + " | ".join(f"{d[s]:.4f}" for s in SEVERITIES) + " |"


def _write_summary_md(result: dict):
    agg = result["aggregates"]
    lines = ["# Skin-tone shift diagnosis — summary", "",
             f"- Checkpoint: `{result['meta']['checkpoint']}`",
             f"- Selected (skin coverage, seed {result['meta']['select_seed']}): "
             + ", ".join(result["selected_stems"]),
             f"- Skin mask: {result['meta']['skin_mask']}",
             f"- Representative severity-sweep image: `{result['meta']['representative_image']}` (type P)",
             "", "## (a) Confusion weight w — skin vs non-skin (severity 1.0)", "",
             "| type | mean w (skin) | mean w (non-skin) |", "|---|---|---|"]
    for t in TYPES:
        lines.append(f"| {t.upper()} | {agg[t]['w_skin_sev1.0']:.4f} | {agg[t]['w_nonskin_sev1.0']:.4f} |")
    lines += ["", "## (b) Skin |Δ| (mean sRGB change in skin) vs severity", "",
              "| type | sev 1.0 | sev 0.7 | sev 0.5 |", "|---|---|---|---|"]
    for t in TYPES:
        lines.append(_fmt_row(t.upper(), agg[t]["skin_delta"]))
    lines += ["", "## (c) Whole-image CRR (ratio_w) vs severity — recovery cost", "",
              "| type | sev 1.0 | sev 0.7 | sev 0.5 |", "|---|---|---|---|"]
    for t in TYPES:
        lines.append(_fmt_row(t.upper(), agg[t]["crr_ratiow"]))
    lines += ["", "> 값은 선정 5장 평균. per-image 원자료는 skin_analysis.json.", "",
              "## 진단 후 선택지 (구현 금지 — 문서만)", ""]
    dm = result["decision_options"]
    lines.append(f"_{dm['note']}_\n")
    for key in ["A_default_0.7_only", "B_inference_skin_attenuation", "C_paper_future_work_only"]:
        o = dm[key]
        lines.append(f"### {key}")
        lines.append(f"- {o['desc']}")
        for k, v in o.items():
            if k != "desc":
                lines.append(f"  - {k}: {v}")
        lines.append("")
    (OUT_DIR / "skin_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {OUT_DIR / 'skin_analysis_summary.md'}")


def _print_table(agg: dict):
    print("\n" + "=" * 64)
    print("SKIN ANALYSIS — aggregates over 5 images")
    print("=" * 64)
    for t in TYPES:
        a = agg[t]
        print(f"\n[type {t.upper()}]")
        print(f"  (a) w  skin={a['w_skin_sev1.0']:.4f}  nonskin={a['w_nonskin_sev1.0']:.4f}  (sev 1.0)")
        print(f"  (b) skin |Δ|   " + "  ".join(f"s{s}={a['skin_delta'][s]:.4f}" for s in SEVERITIES))
        print(f"  (c) CRR ratio_w " + "  ".join(f"s{s}={a['crr_ratiow'][s]:.4f}" for s in SEVERITIES))


if __name__ == "__main__":
    main()
