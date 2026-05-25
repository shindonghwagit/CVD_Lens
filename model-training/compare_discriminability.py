"""
Color Discriminability Comparison: Brettel Algorithm vs CVDLens Model

CVD 사용자 관점에서 색상 분리도(CIEDE2000)를 측정하여
Brettel 알고리즘과 신경망 모델을 공정하게 비교합니다.

지표: CVD 시뮬레이션 후 랜덤 픽셀 쌍 간 평균 CIEDE2000 거리
  - 값이 클수록 CVD 사용자가 더 많은 색을 구분 가능
  - Brettel과 Model 모두 동일한 기준으로 측정 (Brettel이 정답 아님)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from skimage.color import rgb2lab, deltaE_ciede2000
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
ONNX_PATH = ROOT.parent / "cvd-lens" / "inference" / "model" / "cvdlens_fp32.onnx"
ISH_DIR   = ROOT.parent / "cvd-lens" / "public" / "ishihara"
OUT_DIR   = ROOT / "outputs" / "visualization" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CVD_KEYS   = ['p', 'd', 't']
CVD_LABELS = ['Protanopia', 'Deuteranopia', 'Tritanopia']
CVD_COLORS = ['#c0392b', '#27ae60', '#2980b9']
CVD_CHANNEL = {'p': 0.0, 'd': 0.5, 't': 1.0}

# ── ONNX model ─────────────────────────────────────────────────
print("Loading ONNX model...")
sess     = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
IN_NAME  = sess.get_inputs()[0].name
OUT_NAME = sess.get_outputs()[0].name

def infer(img_arr, cvd):
    SIZE = 256
    img  = np.array(Image.fromarray(img_arr).resize((SIZE, SIZE), Image.LANCZOS))
    x    = img.astype(np.float32) / 255.0
    x    = np.concatenate([x.transpose(2,0,1),
                           np.full((1,SIZE,SIZE), CVD_CHANNEL[cvd], np.float32)], axis=0)[None]
    y    = sess.run([OUT_NAME], {IN_NAME: x})[0][0]
    return np.clip(y.transpose(1,2,0) * 255, 0, 255).astype(np.uint8)

# ── Brettel daltonize ──────────────────────────────────────────
_R2L = np.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.02996,0.18431,1.46720]], dtype=np.float64)
_L2R = np.linalg.inv(_R2L)
_MAT = {
    'p': np.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]], dtype=np.float64),
    'd': np.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]], dtype=np.float64),
    't': np.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]], dtype=np.float64),
}
_ERR = np.array([[0,0,0],[0.7,1,0],[0.7,0,1]], dtype=np.float64)

def brettel(img_arr, cvd):
    H, W = img_arr.shape[:2]
    rgb  = img_arr.astype(np.float64).reshape(-1,3).T / 255.0
    lms  = _R2L @ rgb
    sim  = _L2R @ (_MAT[cvd] @ lms)
    out  = np.clip(rgb + _ERR @ (rgb - sim), 0, 1)
    return (out.T.reshape(H,W,3) * 255).astype(np.uint8)

def simulate(img_arr, cvd):
    H, W = img_arr.shape[:2]
    rgb  = img_arr.astype(np.float64).reshape(-1,3).T / 255.0
    sim  = _L2R @ (_MAT[cvd] @ (_R2L @ rgb))
    return (np.clip(sim.T.reshape(H,W,3), 0, 1) * 255).astype(np.uint8)

# ── Color Discriminability (CIEDE2000) ─────────────────────────
def color_discriminability(img_arr, n_pairs=2000, seed=42):
    """
    랜덤 픽셀 쌍 간 평균 CIEDE2000 거리.
    값이 클수록 색상이 다양하게 분포 = CVD 사용자가 더 잘 구분 가능.
    """
    rng  = np.random.default_rng(seed)
    H, W = img_arr.shape[:2]
    lab  = rgb2lab(img_arr.astype(np.float32) / 255.0)
    flat = lab.reshape(-1, 3)
    n    = flat.shape[0]

    idx1 = rng.integers(0, n, n_pairs)
    idx2 = rng.integers(0, n, n_pairs)
    same = idx1 == idx2
    idx2[same] = (idx2[same] + 1) % n

    dE = deltaE_ciede2000(flat[idx1], flat[idx2])
    return float(np.mean(dE))

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
print(f"Test images: {len(test_imgs)} total\n")

# ── evaluate ───────────────────────────────────────────────────
print(f"{'CVD Type':<16} {'Method':<12} {'Discriminability (ΔE)':>22}  {'vs Original':>12}")
print("-" * 66)

results = {cvd: {} for cvd in CVD_KEYS}

for cvd, label in zip(CVD_KEYS, CVD_LABELS):
    scores = {'original': [], 'brettel': [], 'model': []}

    for i, img in enumerate(test_imgs):
        img_256 = np.array(Image.fromarray(img).resize((256, 256), Image.LANCZOS))

        sim_orig    = simulate(img_256, cvd)
        sim_brettel = simulate(brettel(img_256, cvd), cvd)
        sim_model   = simulate(infer(img_256, cvd), cvd)

        scores['original'].append(color_discriminability(sim_orig))
        scores['brettel'].append(color_discriminability(sim_brettel))
        scores['model'].append(color_discriminability(sim_model))

    m_orig    = np.mean(scores['original'])
    m_brettel = np.mean(scores['brettel'])
    m_model   = np.mean(scores['model'])

    results[cvd] = {'original': m_orig, 'brettel': m_brettel, 'model': m_model}

    print(f"{label:<16} {'Original':<12} {m_orig:>22.4f}  {'(baseline)':>12}")
    print(f"{label:<16} {'Brettel':<12} {m_brettel:>22.4f}  {m_brettel - m_orig:>+12.4f}")
    print(f"{label:<16} {'Model':<12} {m_model:>22.4f}  {m_model - m_orig:>+12.4f}")
    print()

# ── chart ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig.suptitle('Color Discriminability Comparison\nBrettel Algorithm vs CVDLens Model\n'
             '(mean CIEDE2000 after CVD simulation — higher = more distinguishable)',
             fontsize=12, fontweight='bold')

for i, (cvd, label, color) in enumerate(zip(CVD_KEYS, CVD_LABELS, CVD_COLORS)):
    ax   = axes[i]
    r    = results[cvd]
    vals = [r['original'], r['brettel'], r['model']]
    xpos = np.arange(3)

    bar_colors = ['#aaaaaa', '#e67e22', color]
    bars = ax.bar(xpos, vals, color=bar_colors, alpha=0.88,
                  edgecolor='white', linewidth=1.5, zorder=3)

    # value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # improvement arrows
    for j, (method, val) in enumerate([('Brettel', r['brettel']), ('Model', r['model'])]):
        delta = val - r['original']
        sign  = '+' if delta >= 0 else ''
        col   = '#27ae60' if delta >= 0 else '#e74c3c'
        ax.text(j + 1, val + 0.35, f'{sign}{delta:.2f}',
                ha='center', fontsize=9, color=col, fontweight='bold')

    ax.set_xticks(xpos)
    ax.set_xticklabels(['Original\n(baseline)', 'Brettel\nAlgorithm', 'CVDLens\nModel'], fontsize=10)
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_ylabel('Mean CIEDE2000 (ΔE)', fontsize=10)
    ymin = min(vals) * 0.92
    ax.set_ylim(ymin, max(vals) * 1.18)
    ax.tick_params(labelsize=9)

plt.tight_layout()
out = OUT_DIR / 'metric_discriminability.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart saved: {out}")
