"""
Investigation #1 (mechanism) — is interior blotching driven by the luminance
GUIDE (depth axis, D=8) rather than the 16×16 spatial grid?

Builds on artifact_probe.py finding: a perfectly-uniform red (guide_std=0) gets a
uniform interior delta (no banding) + a border halo. So interior blotches need
input variation. Here we auto-locate a uniform-red patch in the REAL stop sign
(256-space, where the correction originates) and test whether corrected HUE is a
step function of the guide value (= D=8 guide-bin quantization).

MEASURE ONLY. No daltonize import.
Run: py -m cvdlens_v2.artifact_probe2
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import cv2, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import colorsys

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.artifact_probe import correct, _load, to_lab, ciede2000, OUT, GRID_DEPTH
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass


def rgb_to_hue(img):
    """Vectorized hue in degrees (0-360) from sRGB (H,W,3)."""
    mx = img.max(2); mn = img.min(2); df = mx - mn + 1e-9
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    h = np.zeros_like(mx)
    idx = (mx == r); h[idx] = (60 * ((g-b)/df) % 360)[idx]
    idx = (mx == g); h[idx] = (60 * ((b-r)/df) + 120)[idx]
    idx = (mx == b); h[idx] = (60 * ((r-g)/df) + 240)[idx]
    return h


def find_uniform_red_box(lb, size=36):
    """Slide a window over the 256 letterbox; return box maximizing red-uniformity
    (high R, low G/B, low spatial variance)."""
    r, g, b = lb[..., 0], lb[..., 1], lb[..., 2]
    redness = (r > 0.45) & (r > g + 0.2) & (r > b + 0.15)
    best, best_box = -1, None
    H, W = lb.shape[:2]
    for y in range(0, H - size, 8):
        for x in range(0, W - size, 8):
            patch = redness[y:y+size, x:x+size]
            frac = patch.mean()
            if frac < 0.85:
                continue
            var = lb[y:y+size, x:x+size].reshape(-1, 3).var(0).sum()
            score = frac - 5 * var           # prefer red + low variance
            if score > best:
                best, best_box = score, (x, y, x+size, y+size)
    return best_box


def main():
    stop = _load(OUT / "stop_sign.jpg")
    r = correct(stop, "p", 1.0)
    lb, out256, guide = r["lb256"], r["out256"], r["guide256"]
    box = find_uniform_red_box(lb)
    if box is None:
        print("[!] no uniform-red box found"); return
    x0, y0, x1, y1 = box
    print(f"[box] uniform-red 256-crop = {box}")

    o = lb[y0:y1, x0:x1]; c = np.clip(out256[y0:y1, x0:x1], 0, 1)
    gcrop = guide[y0:y1, x0:x1]
    de = ciede2000(to_lab(o), to_lab(c))
    hue_o = rgb_to_hue(o); hue_c = rgb_to_hue(c)

    # guide → D-bin index (model maps guide[-1,1] → [0,D-1])
    dbin = (gcrop + 1.0) * 0.5 * (GRID_DEPTH - 1)

    metrics = dict(
        box=box,
        orig_R_mean=float(o[..., 0].mean()), orig_R_std=float(o[..., 0].std()),
        orig_spatial_var=float(o.reshape(-1, 3).var(0).sum()),
        guide_min=float(gcrop.min()), guide_max=float(gcrop.max()),
        guide_range=float(gcrop.max() - gcrop.min()),
        dbin_min=float(dbin.min()), dbin_max=float(dbin.max()),
        dbin_span=float(dbin.max() - dbin.min()),
        corrected_hue_mean=float(hue_c.mean()), corrected_hue_std=float(hue_c.std()),
        orig_hue_std=float(hue_o.std()),
        deltaE_mean=float(de.mean()), deltaE_std=float(de.std()),
    )
    print("[metrics]", json.dumps(metrics, indent=1))

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax[0, 0].imshow(o); ax[0, 0].set_title(f"orig crop (R={metrics['orig_R_mean']:.2f}±{metrics['orig_R_std']:.3f})"); ax[0, 0].axis("off")
    ax[0, 1].imshow(c); ax[0, 1].set_title(f"corrected crop (hue std={metrics['corrected_hue_std']:.1f}°)"); ax[0, 1].axis("off")
    im = ax[0, 2].imshow(de, cmap="inferno"); ax[0, 2].set_title(f"ΔE00 in crop (std={de.std():.2f})"); ax[0, 2].axis("off")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046)

    g = ax[1, 0].imshow(gcrop, cmap="viridis"); ax[1, 0].set_title(f"guide (range={metrics['guide_range']:.3f}, D-span={metrics['dbin_span']:.2f} bins)"); ax[1, 0].axis("off")
    fig.colorbar(g, ax=ax[1, 0], fraction=0.046)
    h = ax[1, 1].imshow(hue_c, cmap="hsv"); ax[1, 1].set_title("corrected hue (deg)"); ax[1, 1].axis("off")
    fig.colorbar(h, ax=ax[1, 1], fraction=0.046)

    ax[1, 2].scatter(dbin.ravel(), hue_c.ravel(), s=2, alpha=0.3)
    for k in range(GRID_DEPTH):
        ax[1, 2].axvline(k, color="gray", ls=":", lw=0.5)
    ax[1, 2].set_xlabel("guide D-bin index (0..7)"); ax[1, 2].set_ylabel("corrected hue (deg)")
    ax[1, 2].set_title("hue vs guide-bin — steps ⇒ D=8 quantization")

    fig.suptitle("[#1 mechanism] uniform-red crop of real STOP · protan s1.0 — guide-driven hue variation", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "mechanism_stop_p.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    (OUT / "mechanism_stats.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[save] {OUT/'mechanism_stop_p.png'}, mechanism_stats.json")


if __name__ == "__main__":
    main()
