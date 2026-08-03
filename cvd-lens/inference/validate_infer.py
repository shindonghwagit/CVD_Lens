"""
Validation for the native-resolution delta-composite /infer path.

Three numeric checks (see the bug report / RUNBOOK):

  a. Fine-text sharpness — a small-print image through _correct_image keeps its
     high-frequency energy (ratio result/original > 0.95). The old 256→upscale
     path is reported alongside to show the regression it fixes.
  b. 1584 bus — the new full-res correction points the same way as the legacy
     256 path (cosine similarity of the mean RGB shift > 0.90). Saves a visual.
  c. 1761 do-nothing — mean |Δ| at native resolution stays < 0.005.

Run:  py validate_infer.py            (from cvd-lens/inference)
"""

from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import main  # loads the ONNX sessions

COCO = Path("C:/Users/SCH/coco/val2017")
OUT = Path("C:/Users/SCH/graduation_project/outputs/v2_phase2/infer_fix")
OUT.mkdir(parents=True, exist_ok=True)

TYPES = ("p", "d", "t")


def hf_energy(rgb_u8: np.ndarray) -> float:
    """High-frequency energy = mean squared Laplacian of luma. Blur → drops."""
    g = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float((lap ** 2).mean())


def old_path(arr_f32: np.ndarray, cvd_type: str, severity: float) -> np.ndarray:
    """Legacy pipeline: center-crop → 256 → model → return 256, upscaled to
    native for display. Returns float32 [0,1] at native resolution."""
    h, w = arr_f32.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = arr_f32[y0:y0 + side, x0:x0 + side]
    in256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
    out256 = np.clip(main._run_float(in256, cvd_type, severity), 0.0, 1.0)
    return cv2.resize(out256, (w, h), interpolation=cv2.INTER_LINEAR)


def load_native(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


# ── (a) fine-text sharpness ──────────────────────────────────────────────────
def make_text_image(size: int = 768) -> np.ndarray:
    """Square small-print image (high-frequency, deterministic)."""
    img = Image.new("RGB", (size, size), (245, 245, 245))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 13)
    lines = [
        "The quick brown fox jumps over the lazy dog. 0123456789 !@#$%^&*()",
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ  fine print",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do.",
    ]
    colors = [(20, 20, 20), (150, 20, 20), (20, 90, 20), (20, 20, 150)]
    y = 8
    i = 0
    while y < size - 16:
        d.text((8, y), f"{i:03d} {lines[i % len(lines)]}", font=font,
               fill=colors[i % len(colors)])
        y += 17
        i += 1
    return np.asarray(img, dtype=np.float32) / 255.0


def check_text():
    print("\n(a) fine-text sharpness  (high-freq energy ratio, want > 0.95)")
    arr = make_text_image()
    orig_u8 = main._to_u8(arr)
    hf_o = hf_energy(orig_u8)
    Image.fromarray(orig_u8).save(OUT / "text_original.png")

    worst = 1.0
    for t in TYPES:
        new_u8 = main._to_u8(main._correct_image(arr, t, 1.0))
        old_u8 = main._to_u8(old_path(arr, t, 1.0))
        r_new = hf_energy(new_u8) / hf_o
        r_old = hf_energy(old_u8) / hf_o
        worst = min(worst, r_new)
        print(f"    {t}:  new/orig={r_new:.3f}   (legacy 256 path={r_old:.3f})")
        Image.fromarray(new_u8).save(OUT / f"text_new_{t}.png")
    ok = worst > 0.95
    print(f"    → worst new/orig = {worst:.3f}  [{'PASS' if ok else 'FAIL'}]")
    return ok


# ── (b) direction preserved on 1584 ──────────────────────────────────────────
def check_direction():
    print("\n(b) 1584 correction direction  (cosine vs legacy 256, want > 0.90)")
    arr = load_native(COCO / "000000001584.jpg")
    arr = main._cap_long_side(arr)
    orig_u8 = main._to_u8(arr)

    worst = 1.0
    panels = [orig_u8]
    for t in TYPES:
        new = main._correct_image(arr, t, 1.0)
        old = old_path(arr, t, 1.0)
        d_new = (new - arr).reshape(-1, 3).mean(0)
        d_old = (old - arr).reshape(-1, 3).mean(0)
        cos = float(d_new @ d_old / (np.linalg.norm(d_new) * np.linalg.norm(d_old) + 1e-12))
        worst = min(worst, cos)
        print(f"    {t}:  cos={cos:+.3f}   shift_new={d_new.round(4)}  "
              f"shift_old={d_old.round(4)}")
        if t == "d":
            panels.append(main._to_u8(new))
    ok = worst > 0.90
    vis = np.concatenate(panels, axis=1)
    Image.fromarray(vis).save(OUT / "bus_1584_orig_vs_new_d.png")
    print(f"    → worst cosine = {worst:+.3f}  [{'PASS' if ok else 'FAIL'}]  "
          f"(visual: bus_1584_orig_vs_new_d.png)")
    return ok


# ── (c) do-nothing on 1761 ───────────────────────────────────────────────────
def check_donothing():
    print("\n(c) 1761 do-nothing  (mean |Δ| at native res, want < 0.005)")
    arr = load_native(COCO / "000000001761.jpg")
    arr = main._cap_long_side(arr)

    worst = 0.0
    for t in TYPES:
        out = main._correct_image(arr, t, 1.0)
        d = float(np.abs(out - arr).mean())
        worst = max(worst, d)
        print(f"    {t}:  mean|Δ|={d:.5f}")
    ok = worst < 0.005
    print(f"    → worst mean|Δ| = {worst:.5f}  [{'PASS' if ok else 'FAIL'}]")
    return ok


if __name__ == "__main__":
    print("=" * 70)
    print("Native-resolution delta-composite /infer validation")
    print("=" * 70)
    results = {
        "a_text_sharpness": check_text(),
        "b_direction_1584": check_direction(),
        "c_donothing_1761": check_donothing(),
    }
    print("\n" + "=" * 70)
    allok = all(results.values())
    for k, v in results.items():
        print(f"  {k:20s} {'PASS' if v else 'FAIL'}")
    print("  " + ("*** ALL PASS ***" if allok else "*** FAIL ***"))
    print(f"  artifacts → {OUT}")
    sys.exit(0 if allok else 1)
