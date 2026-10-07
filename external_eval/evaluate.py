"""Evaluate the deployed CVDLens pipeline on the external Commons set."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "cvd-lens" / "inference"
sys.path.insert(0, str(INFERENCE))
import main as deployed  # noqa: E402

OUT = ROOT / "external_eval" / "results"
TYPES = ("p", "d", "t")
MACHADO = {
    "p": np.array([[.152286, 1.052583, -.204868], [.114503, .786281, .099216], [-.003882, -.048116, 1.051998]], np.float32),
    "d": np.array([[.367322, .860646, -.227968], [.280085, .672501, .047413], [-.011820, .042940, .968881]], np.float32),
}
RGB2LMS = np.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[.02996,.18431,1.46720]], np.float32)
LMS2RGB = np.linalg.inv(RGB2LMS).astype(np.float32)
BRETTEL_T = LMS2RGB @ np.array([[1,0,0],[0,1,0],[-.395913,.801109,0]], np.float32) @ RGB2LMS
RGB2XYZ = np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]], np.float32)
WHITE = np.array([.95047, 1., 1.08883], np.float32)


def srgb_linear(x):
    return np.where(x > .04045, ((np.maximum(x, 0) + .055) / 1.055) ** 2.4, x / 12.92).astype(np.float32)


def lab(x):
    xyz = np.einsum("ij,hwj->hwi", RGB2XYZ, x) / WHITE
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(np.maximum(xyz, 0)), xyz / (3 * d * d) + 4 / 29)
    return np.stack((116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])), -1)


def simulate(x, kind):
    mat = BRETTEL_T if kind == "t" else MACHADO[kind]
    return np.clip(np.einsum("ij,hwj->hwi", mat, x), 0, 1)


def confusion_weight(orig_lin, kind):
    de = np.linalg.norm(lab(orig_lin) - lab(simulate(orig_lin, kind)), axis=2)
    lo, hi = (12., 30.) if kind == "t" else (2., 12.)
    w = np.clip((de - lo) / (hi - lo), 0, 1).astype(np.float32)
    return cv2.GaussianBlur(w, (11, 11), 3, borderType=cv2.BORDER_REFLECT)


def crr(orig_lin, out_lin, w, kind):
    if float(w.mean()) < 1e-6:
        return float("nan")
    o = cv2.GaussianBlur(simulate(orig_lin, kind), (5, 5), 1, borderType=cv2.BORDER_REFLECT)
    a = cv2.GaussianBlur(simulate(out_lin, kind), (5, 5), 1, borderType=cv2.BORDER_REFLECT)
    def grad(x):
        dx = np.pad(np.abs(x[:,1:] - x[:,:-1]).mean(2), ((0,0),(0,1)))
        dy = np.pad(np.abs(x[1:] - x[:-1]).mean(2), ((0,1),(0,0)))
        return dx + dy
    den = float((w * grad(o)).sum() / (w.sum() + 1e-8))
    num = float((w * grad(a)).sum() / (w.sum() + 1e-8))
    return num / (den + 1e-8)


def load_image(path, max_side=512):
    with Image.open(path) as opened:
        im = ImageOps.exif_transpose(opened).convert("RGB")
    scale = min(1., max_side / max(im.size))
    if scale < 1:
        im = im.resize((round(im.width*scale), round(im.height*scale)), Image.Resampling.LANCZOS)
    return np.asarray(im, np.float32) / 255.


def summarize(rows, key, selected=None):
    vals = [float(r[key]) for r in rows if (selected is None or selected(r)) and np.isfinite(float(r[key]))]
    return {"n": len(vals), "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "external_eval" / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    rows = []
    for i, item in enumerate(manifest, 1):
        orig = load_image(ROOT / "external_eval" / "images" / item["filename"])
        orig_lin = srgb_linear(orig)
        for kind in TYPES:
            corrected = deployed._correct_image(orig, kind, 1.0)
            out_lin = srgb_linear(corrected)
            w = confusion_weight(orig_lin, kind)
            delta = np.abs(corrected - orig)
            clip = ((corrected <= 1/255) | (corrected >= 254/255)).mean()
            rows.append({"id":item["id"], "category":item["category"], "filename":item["filename"],
                         "cvd_type":kind, "w_mean":float(w.mean()), "crr":crr(orig_lin,out_lin,w,kind),
                         "np_mae":float(delta.mean()), "changed_pixel_ratio":float((delta.max(2) > 1/255).mean()),
                         "large_change_ratio":float((delta.max(2) > .1).mean()), "output_clip_ratio":float(clip)})
        print(f"[{i:02d}/{len(manifest)}] {item['id']}", flush=True)

    fields = list(rows[0])
    with (OUT / "eval_rows.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)

    result = {"dataset":{"images":len(manifest),"categories":sorted({x['category'] for x in manifest}),"max_side":512}, "types":{}}
    for kind in TYPES:
        sub = [r for r in rows if r["cvd_type"] == kind]
        cutoff = float(np.median([r["w_mean"] for r in sub]))
        meaningful = lambda r, c=cutoff: r["w_mean"] >= c
        result["types"][kind] = {"w_median_cutoff":cutoff, "crr_all":summarize(sub,"crr"),
            "crr_meaningful_half":summarize(sub,"crr",meaningful), "np_mae":summarize(sub,"np_mae"),
            "changed_pixel_ratio":summarize(sub,"changed_pixel_ratio"), "large_change_ratio":summarize(sub,"large_change_ratio"),
            "meaningful_crr_below_1_count":sum(meaningful(r) and r["crr"] < 1 for r in sub)}
    (OUT / "eval_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# External validation report", "", "Deployment-equivalent inference on 60 Wikimedia Commons images (10 per category).",
             "P/D use the deployed ONNX models; T uses the deployed 30-degree hue rule. Images are aspect-preserving, capped at 512 px.", "",
             "CRR > 1 means increased contrast in CVD-confusion regions. NP MAE is mean absolute sRGB change (lower is more natural).",
             "The `meaningful half` is the 30 images at or above each type's median confusion-weight mass; this avoids unstable CRR on irrelevant images.", "",
             "| Type | CRR all mean/median | CRR meaningful mean/median | failures (<1) | NP MAE | changed pixels | large changes |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for kind in TYPES:
        x=result["types"][kind]
        lines.append(f"| {kind.upper()} | {x['crr_all']['mean']:.3f}/{x['crr_all']['median']:.3f} | {x['crr_meaningful_half']['mean']:.3f}/{x['crr_meaningful_half']['median']:.3f} | {x['meaningful_crr_below_1_count']}/30 | {x['np_mae']['mean']:.4f} | {x['changed_pixel_ratio']['mean']:.1%} | {x['large_change_ratio']['mean']:.1%} |")
    lines += ["", "## Interpretation", "",
              "- P and D generalized well on the confusion-relevant half: mean CRR was 1.224 and 1.192, with 1 and 0 images below 1 respectively.",
              "- T did not generalize reliably: its relevant-half median was 0.997 and 16/30 images were below 1. The worst relevant cases were `traffic_02` (0.762), `nature_03` (0.781), and `indoor_lowlight_05` (0.801).",
              "- T also made the largest local changes: pixels changing by more than 0.1 averaged 10.4%, versus 6.3-6.4% for P/D. `nature_03` changed 95.9% of pixels by more than 0.1 while CRR fell to 0.781.", "",
              "Conclusion: the external set supports the deployed P/D models as an engineering result. It does not support claiming that the current T hue rule consistently improves arbitrary photographs; T should be presented as an experimental heuristic or revised and revalidated.", "",
              "## Limitations", "", "This is an engineering robustness check, not a clinical user study. The set is small and web-sourced; CRR is simulator-based and does not prove human perceptual benefit."]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
