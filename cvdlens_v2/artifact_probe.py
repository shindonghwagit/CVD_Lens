"""
Investigation #1 — uniform-region spatial artifact (banding/blotches).

Reproduces the DEPLOYED server pipeline (cvd-lens/inference/main.py) exactly with
LOCAL ONNX inference (no web round-trip), then instruments it:
  - CIEDE2000 ΔE heatmap (original vs corrected) — where does the image change?
  - luminance guide map (what the bilateral grid slices on) — does it ripple in
    a "uniform" region?
  - 16×16 grid overlay on the 256-space delta + periodicity (autocorrelation)
    of a uniform-red crop's delta — is the blotch period the grid-cell size?
  - synthetic perfectly-uniform red control — does banding need input variation?

MEASURE ONLY. No model/inference edits. daltonize NOT imported. Simulation, if
needed elsewhere, uses the Brettel-1997 NumPy path in cvdlens_v2.simulation.

Run: py -m cvdlens_v2.artifact_probe
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as rt
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.color import srgb_to_linear, rgb_to_lab

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODEL_DIR = Path("cvd-lens/inference/model")
OUT = Path("outputs/artifact_analysis")
GRID_SPATIAL = 16          # bilateral grid H_g=W_g (model.py grid_spatial)
GRID_DEPTH = 8             # guide-axis bins D (model.py grid_depth)
MAX_SIDE = 2048            # main.py MAX_SIDE

_SESS = {t: rt.InferenceSession(str(MODEL_DIR / f"cvdlens_{t}.onnx"),
                                providers=["CPUExecutionProvider"])
         for t in ("p", "d", "t")}


# ── server pipeline (verbatim from main.py, plus intermediates returned) ──
def _run_float(rgb256, cvd_type, severity):
    chw = rgb256.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    sev = np.array([[severity]], dtype=np.float32)
    out = _SESS[cvd_type].run(["out_srgb"], {"srgb": chw, "severity": sev})[0]
    return out[0].transpose(1, 2, 0).astype(np.float32)


def _letterbox(img_f32, size=256):
    h, w = img_f32.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img_f32, (nw, nh), interpolation=interp)
    left, top = (size - nw) // 2, (size - nh) // 2
    right, bottom = size - nw - left, size - nh - top
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return canvas, (left, top, left + nw, top + nh)


def _cap_long_side(img_f32, max_side=MAX_SIDE):
    h, w = img_f32.shape[:2]
    m = max(h, w)
    if m > max_side:
        s = max_side / m
        img_f32 = cv2.resize(img_f32, (round(w * s), round(h * s)),
                             interpolation=cv2.INTER_AREA)
    return img_f32


def correct(img_f32, cvd_type, severity):
    """Returns dict: corrected(native), lb256, out256, delta256, guide256, box."""
    h, w = img_f32.shape[:2]
    lb, (x0, y0, x1, y1) = _letterbox(img_f32, 256)
    out = _run_float(lb, cvd_type, severity)
    delta256 = out - lb
    delta_box = delta256[y0:y1, x0:x1]
    delta_full = cv2.resize(delta_box, (w, h), interpolation=cv2.INTER_LINEAR)
    corrected = np.clip(img_f32 + delta_full, 0.0, 1.0)
    # luminance guide the model computes internally (model._luminance_guide)
    lin = srgb_to_linear(torch.from_numpy(lb).permute(2, 0, 1)[None].float())
    Y = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    guide = (2.0 * Y - 1.0)[0].numpy()                     # (256,256) in [-1,1]
    return dict(corrected=corrected, lb256=lb, out256=out, delta256=delta256,
                guide256=guide, box=(x0, y0, x1, y1))


# ── CIEDE2000 (NumPy; Lab via project's sRGB→Lab for consistency) ────────
def to_lab(img_f32):
    t = torch.from_numpy(img_f32).permute(2, 0, 1)[None].float()
    lab = rgb_to_lab(srgb_to_linear(t))[0].permute(1, 2, 0).numpy()
    return lab  # (H,W,3) L,a,b


def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    avg_Lp = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2); C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(avg_C**7 / (avg_C**7 + 25.0**7 + 1e-12)))
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2); C2p = np.sqrt(a2p**2 + b2**2)
    avg_Cp = (C1p + C2p) / 2.0
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    avg_hp = (h1p + h2p) / 2.0
    avg_hp = np.where(np.abs(h1p - h2p) > 180, avg_hp + 180, avg_hp) % 360
    T = (1 - 0.17 * np.cos(np.radians(avg_hp - 30))
         + 0.24 * np.cos(np.radians(2 * avg_hp))
         + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
         - 0.20 * np.cos(np.radians(4 * avg_hp - 63)))
    Sl = 1 + (0.015 * (avg_Lp - 50)**2) / np.sqrt(20 + (avg_Lp - 50)**2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    dtheta = 30 * np.exp(-(((avg_hp - 275) / 25)**2))
    Rc = 2 * np.sqrt(avg_Cp**7 / (avg_Cp**7 + 25.0**7 + 1e-12))
    Rt = -Rc * np.sin(np.radians(2 * dtheta))
    return np.sqrt((dLp/Sl)**2 + (dCp/Sc)**2 + (dHp/Sh)**2
                   + Rt * (dCp/Sc) * (dHp/Sh))


# ── analysis + figures ────────────────────────────────────────────────────
def _load(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return _cap_long_side(img)


def grid_overlay(ax, size=256):
    step = size / GRID_SPATIAL
    for k in range(1, GRID_SPATIAL):
        ax.axhline(k * step, color="cyan", lw=0.4, alpha=0.6)
        ax.axvline(k * step, color="cyan", lw=0.4, alpha=0.6)


def analyze(name, img, cvd_type, severity, red_crop_256=None):
    r = correct(img, cvd_type, severity)
    corrected = r["corrected"]
    de = ciede2000(to_lab(img), to_lab(corrected))          # native ΔE00
    dmag256 = np.linalg.norm(r["delta256"], axis=2)         # 256-space |delta|
    tag = f"{name}_{cvd_type}_s{severity}"

    cv2.imwrite(str(OUT / f"{tag}_original.png"),
                cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / f"{tag}_corrected.png"),
                cv2.cvtColor((corrected*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax[0, 0].imshow(img); ax[0, 0].set_title(f"{name} original"); ax[0, 0].axis("off")
    ax[0, 1].imshow(corrected); ax[0, 1].set_title(f"corrected {cvd_type} s{severity}"); ax[0, 1].axis("off")
    im = ax[0, 2].imshow(de, cmap="inferno")
    ax[0, 2].set_title(f"CIEDE2000 ΔE (max {de.max():.1f}, mean {de.mean():.2f})"); ax[0, 2].axis("off")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046)

    g = ax[1, 0].imshow(r["guide256"], cmap="viridis")
    ax[1, 0].set_title("luminance guide (256)"); ax[1, 0].axis("off")
    fig.colorbar(g, ax=ax[1, 0], fraction=0.046)
    d = ax[1, 1].imshow(dmag256, cmap="magma")
    grid_overlay(ax[1, 1]); ax[1, 1].set_title("|delta| @256 + 16×16 grid"); ax[1, 1].axis("off")
    fig.colorbar(d, ax=ax[1, 1], fraction=0.046)

    # periodicity on a uniform-red crop (256-space)
    period_txt = "no crop given"
    if red_crop_256 is not None:
        cx0, cy0, cx1, cy1 = red_crop_256
        patch = dmag256[cy0:cy1, cx0:cx1]
        row = patch.mean(axis=0) - patch.mean()
        ac = np.correlate(row, row, mode="full")[len(row)-1:]
        ac = ac / (ac[0] + 1e-12)
        # first local max after lag 0
        peaks = [k for k in range(2, len(ac)-1) if ac[k] > ac[k-1] and ac[k] > ac[k+1] and ac[k] > 0.2]
        expected = 256 / GRID_SPATIAL
        ax[1, 2].plot(ac)
        ax[1, 2].axvline(expected, color="r", ls="--", label=f"grid cell {expected:.0f}px")
        first = peaks[0] if peaks else None
        if first: ax[1, 2].axvline(first, color="g", ls=":", label=f"1st peak {first}px")
        ax[1, 2].set_title("delta autocorrelation (uniform-red crop row)")
        ax[1, 2].set_xlabel("lag (px @256)"); ax[1, 2].legend(fontsize=8)
        ax[1, 2].set_xlim(0, min(80, len(ac)))
        period_txt = f"1st_peak={first}px, grid_cell={expected:.0f}px"
    else:
        ax[1, 2].axis("off")

    fig.suptitle(f"[#1 probe] {tag}   ΔE00 max={de.max():.1f} mean={de.mean():.2f}   {period_txt}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_probe.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    stats = dict(tag=tag, de_max=float(de.max()), de_mean=float(de.mean()),
                 de_p99=float(np.percentile(de, 99)),
                 guide_std=float(r["guide256"].std()),
                 dmag256_max=float(dmag256.max()), period=period_txt)
    print(f"[{tag}] ΔE00 max={stats['de_max']:.2f} mean={stats['de_mean']:.2f} "
          f"p99={stats['de_p99']:.2f} | guide_std={stats['guide_std']:.4f} | {period_txt}")
    return stats, r, de


def synthetic_uniform_red():
    """Perfectly uniform saturated red (with optional faint luminance ramp)."""
    H = W = 512
    flat = np.zeros((H, W, 3), np.float32); flat[..., 0] = 0.80  # solid red
    ramp = flat.copy()
    ramp[..., 0] = np.clip(0.80 + np.linspace(-0.05, 0.05, W)[None, :], 0, 1)  # faint L ramp
    return flat, ramp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    # Real STOP sign — big uniform red field. Protan is the reported case.
    stop = _load(OUT / "stop_sign.jpg")
    # a uniform-red crop in 256-space (upper-left red octagon field): rough box
    results.append(analyze("stop", stop, "p", 1.0, red_crop_256=(70, 90, 120, 160))[0])
    results.append(analyze("stop", stop, "p", 0.6, red_crop_256=(70, 90, 120, 160))[0])
    results.append(analyze("stop", stop, "d", 1.0, red_crop_256=(70, 90, 120, 160))[0])

    # Synthetic controls: does banding need input variation?
    flat, ramp = synthetic_uniform_red()
    results.append(analyze("synth_flat", flat, "p", 1.0, red_crop_256=(40, 40, 216, 216))[0])
    results.append(analyze("synth_ramp", ramp, "p", 1.0, red_crop_256=(40, 40, 216, 216))[0])

    import json
    (OUT / "probe_stats.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[save] {OUT/'probe_stats.json'} ({len(results)} cases)")


if __name__ == "__main__":
    main()
