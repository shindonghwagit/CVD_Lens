"""
CVDLens evaluation metrics visualization for presentation
Generates 4 charts: PSNR, SSIM, Sim-SSIM before/after, Summary
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
ONNX_PATH = ROOT.parent / "cvd-lens" / "inference" / "model" / "cvdlens_fp32.onnx"
ISH_DIR   = ROOT.parent / "cvd-lens" / "public" / "ishihara"
OUT_DIR   = ROOT / "outputs" / "visualization" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CVD_KEYS   = ['p', 'd', 't']
CVD_LABELS = ['Protanopia', 'Deuteranopia', 'Tritanopia']
CVD_COLORS = ['#c0392b', '#27ae60', '#2980b9']
CVD_CHANNEL = {'p': 0.0, 'd': 0.5, 't': 1.0}

# ── style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ── ONNX model ─────────────────────────────────────────────────
print("Loading ONNX model...")
sess      = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
IN_NAME   = sess.get_inputs()[0].name
OUT_NAME  = sess.get_outputs()[0].name

def infer(img_arr, cvd):
    SIZE = 256
    img  = np.array(Image.fromarray(img_arr).resize((SIZE, SIZE), Image.LANCZOS))
    x    = img.astype(np.float32) / 255.0
    x    = np.concatenate([x.transpose(2,0,1),
                           np.full((1,SIZE,SIZE), CVD_CHANNEL[cvd], dtype=np.float32)], axis=0)[None]
    y    = sess.run([OUT_NAME], {IN_NAME: x})[0][0]
    return np.clip(y.transpose(1,2,0) * 255, 0, 255).astype(np.uint8)

# ── Brettel daltonize (ground truth) ──────────────────────────
_R2L = np.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.02996,0.18431,1.46720]], dtype=np.float64)
_L2R = np.linalg.inv(_R2L)
_DAL  = {
    'p': np.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]], dtype=np.float64),
    'd': np.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]], dtype=np.float64),
    't': np.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]], dtype=np.float64),
}
_ERR = np.array([[0,0,0],[0.7,1,0],[0.7,0,1]], dtype=np.float64)

def daltonize(img_arr, cvd):
    H, W = img_arr.shape[:2]
    rgb  = img_arr.astype(np.float64).reshape(-1,3).T / 255.0
    lms  = _R2L @ rgb
    sim  = _L2R @ (_DAL[cvd] @ lms)
    out  = np.clip(rgb + _ERR @ (rgb - sim), 0, 1)
    return (out.T.reshape(H,W,3) * 255).astype(np.uint8)

def simulate(img_arr, cvd):
    H, W = img_arr.shape[:2]
    rgb  = img_arr.astype(np.float64).reshape(-1,3).T / 255.0
    sim  = _L2R @ (_DAL[cvd] @ (_R2L @ rgb))
    return (np.clip(sim.T.reshape(H,W,3), 0, 1) * 255).astype(np.uint8)

# ── test images ────────────────────────────────────────────────
def make_colorful_patches(n=20, size=256, seed=42):
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(n):
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        for _ in range(rng.integers(4, 9)):
            x1, y1 = rng.integers(0, size-20, 2)
            x2 = x1 + rng.integers(20, size-x1)
            y2 = y1 + rng.integers(20, size-y1)
            patch[x1:x2, y1:y2] = rng.integers(30, 255, 3)
        from PIL.ImageFilter import GaussianBlur
        imgs.append(np.array(Image.fromarray(patch).filter(GaussianBlur(radius=8))))
    return imgs

real_imgs = [np.array(Image.open(p).convert('RGB')) for p in sorted(ISH_DIR.glob("*.jpg"))[:5]]
real_imgs += [np.array(Image.open(p).convert('RGB')) for p in sorted(ISH_DIR.glob("*.png"))[:2]]
test_imgs = make_colorful_patches(20) + real_imgs
print(f"Test images: {len(test_imgs)} total")

# ── compute metrics ────────────────────────────────────────────
print("Computing metrics...")
psnr_vals = {k: [] for k in CVD_KEYS}
ssim_vals = {k: [] for k in CVD_KEYS}
sim_before = {k: [] for k in CVD_KEYS}
sim_after  = {k: [] for k in CVD_KEYS}

for i, img in enumerate(test_imgs):
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{len(test_imgs)}")
    img_256 = np.array(Image.fromarray(img).resize((256,256), Image.LANCZOS))
    for cvd in CVD_KEYS:
        gt   = daltonize(img, cvd)
        pred = infer(img, cvd)
        gt_256 = np.array(Image.fromarray(gt).resize((256,256), Image.LANCZOS))

        psnr_vals[cvd].append(psnr(gt_256, pred, data_range=255))
        ssim_vals[cvd].append(ssim(gt_256, pred, data_range=255, channel_axis=2))

        s_orig = simulate(img_256, cvd)
        s_corr = simulate(pred, cvd)
        sim_before[cvd].append(ssim(s_orig, img_256, data_range=255, channel_axis=2))
        sim_after[cvd].append(ssim(s_corr, img_256, data_range=255, channel_axis=2))

psnr_means = [np.mean(psnr_vals[k]) for k in CVD_KEYS]
ssim_means = [np.mean(ssim_vals[k]) for k in CVD_KEYS]
sim_b_means = [np.mean(sim_before[k]) for k in CVD_KEYS]
sim_a_means = [np.mean(sim_after[k]) for k in CVD_KEYS]

print("\nResults:")
for i, k in enumerate(CVD_KEYS):
    print(f"  {CVD_LABELS[i]:15} PSNR={psnr_means[i]:.2f} SSIM={ssim_means[i]:.4f} SimSSIM={sim_b_means[i]:.4f}->{sim_a_means[i]:.4f}")

# ══════════════════════════════════════════════════════════════
# Figure 1 — PSNR bar chart
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(CVD_LABELS))
bars = ax.bar(x, psnr_means, width=0.5, color=CVD_COLORS, alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)

# reference lines
for val, label, ls in [(30,'30 dB — Acceptable','dotted'),(35,'35 dB — Good','-.'),(40,'40 dB — Excellent','--')]:
    ax.axhline(val, color='#555', linewidth=1.2, linestyle=ls, label=label, zorder=2)

# value labels on bars
for bar, val in zip(bars, psnr_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.2f} dB', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(CVD_LABELS, fontsize=12)
ax.set_ylabel('PSNR (dB)', fontsize=12)
ax.set_title('PSNR — Model Output vs. Brettel Ground Truth', fontsize=13, fontweight='bold', pad=14)
ax.set_ylim(0, max(psnr_means) * 1.15)
ax.legend(fontsize=9, loc='lower right')
ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
out = OUT_DIR / 'metric_psnr.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out}')

# ══════════════════════════════════════════════════════════════
# Figure 2 — SSIM bar chart
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(x, ssim_means, width=0.5, color=CVD_COLORS, alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)

for val, label, ls in [(0.90,'0.90 — Acceptable','dotted'),(0.95,'0.95 — Good','-.'),(0.98,'0.98 — Excellent','--')]:
    ax.axhline(val, color='#555', linewidth=1.2, linestyle=ls, label=label, zorder=2)

for bar, val in zip(bars, ssim_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(CVD_LABELS, fontsize=12)
ax.set_ylabel('SSIM', fontsize=12)
ax.set_title('SSIM — Model Output vs. Brettel Ground Truth', fontsize=13, fontweight='bold', pad=14)
ax.set_ylim(0.97, 1.002)
ax.legend(fontsize=9, loc='lower right')
ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
out = OUT_DIR / 'metric_ssim.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════
# Figure 3 — Simulation-based SSIM before / after
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

w = 0.32
x2 = np.arange(len(CVD_LABELS))
b1 = ax.bar(x2 - w/2, sim_b_means, width=w, label='Before Correction',
            color='#aaaaaa', alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
b2 = ax.bar(x2 + w/2, sim_a_means, width=w, label='After Correction',
            color=CVD_COLORS, alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)

for bar, val in zip(b1, sim_b_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, color='#555')
for bar, val, color in zip(b2, sim_a_means, CVD_COLORS):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

# delta annotations
for i in range(len(CVD_LABELS)):
    delta = sim_a_means[i] - sim_b_means[i]
    color = '#27ae60' if delta >= 0 else '#e74c3c'
    sign  = '+' if delta >= 0 else ''
    ax.text(x2[i], max(sim_b_means[i], sim_a_means[i]) + 0.012,
            f'{sign}{delta:.4f}', ha='center', fontsize=10, fontweight='bold', color=color)

ax.set_xticks(x2)
ax.set_xticklabels(CVD_LABELS, fontsize=12)
ax.set_ylabel('SSIM vs. Original', fontsize=12)
ax.set_title('Simulation-based SSIM\nSSIM( sim(image), original ) — Before vs. After Correction',
             fontsize=12, fontweight='bold', pad=14)
ymin = min(sim_b_means + sim_a_means) - 0.04
ax.set_ylim(ymin, max(sim_b_means + sim_a_means) + 0.06)
ax.legend(fontsize=10)
ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
out = OUT_DIR / 'metric_sim_ssim.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════
# Figure 4 — Summary (all metrics in one view)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CVDLens — Evaluation Metrics Summary', fontsize=14, fontweight='bold', y=1.02)

# --- PSNR ---
ax = axes[0]
bars = ax.bar(CVD_LABELS, psnr_means, color=CVD_COLORS, alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
ax.axhline(40, color='#555', linewidth=1.2, linestyle='--', label='40 dB threshold', zorder=2)
for bar, val in zip(bars, psnr_means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('PSNR (dB)\nvs. Brettel GT', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(psnr_means)*1.18)
ax.legend(fontsize=8)
ax.tick_params(labelsize=9)

# --- SSIM ---
ax = axes[1]
bars = ax.bar(CVD_LABELS, ssim_means, color=CVD_COLORS, alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
ax.axhline(0.98, color='#555', linewidth=1.2, linestyle='--', label='0.98 threshold', zorder=2)
for bar, val in zip(bars, ssim_means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('SSIM\nvs. Brettel GT', fontsize=11, fontweight='bold')
ax.set_ylim(0.97, 1.002)
ax.legend(fontsize=8)
ax.tick_params(labelsize=9)

# --- Sim-SSIM ---
ax = axes[2]
w = 0.3
x3 = np.arange(3)
ax.bar(x3 - w/2, sim_b_means, width=w, label='Before', color='#aaaaaa', alpha=0.85, edgecolor='white', zorder=3)
ax.bar(x3 + w/2, sim_a_means, width=w, label='After',  color=CVD_COLORS, alpha=0.85, edgecolor='white', zorder=3)
for i in range(3):
    delta = sim_a_means[i] - sim_b_means[i]
    color = '#27ae60' if delta >= 0 else '#e74c3c'
    sign  = '+' if delta >= 0 else ''
    ax.text(x3[i], max(sim_b_means[i], sim_a_means[i]) + 0.01,
            f'{sign}{delta:.4f}', ha='center', fontsize=9, fontweight='bold', color=color)
ax.set_xticks(x3)
ax.set_xticklabels(CVD_LABELS, fontsize=9)
ax.set_title('Sim-SSIM\nBefore vs. After Correction', fontsize=11, fontweight='bold')
ax.set_ylim(min(sim_b_means+sim_a_means)-0.05, max(sim_b_means+sim_a_means)+0.07)
ax.legend(fontsize=9)
ax.tick_params(labelsize=9)

plt.tight_layout()
out = OUT_DIR / 'metric_summary.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

print(f'\nAll charts saved to: {OUT_DIR}')
