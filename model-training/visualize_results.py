"""
CVDLens correction result visualization
Usage:
  python visualize_results.py                        # uses Ishihara plates
  python visualize_results.py img1.jpg img2.png ...  # uses given images
  python visualize_results.py --dir C:/path/to/imgs  # uses all images in folder
"""

import sys, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── paths ────────────────────────────────────────────────
ROOT      = Path(__file__).parent
ONNX_PATH = ROOT.parent / "cvd-lens" / "inference" / "model" / "cvdlens_fp32.onnx"
ISH_DIR   = ROOT.parent / "cvd-lens" / "public" / "ishihara"
OUT_DIR   = ROOT / "outputs" / "visualization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# use a font that supports Korean (fallback to default if not found)
_kr_fonts = [f.name for f in fm.fontManager.ttflist if 'Gothic' in f.name or 'Malgun' in f.name or 'NanumGothic' in f.name]
if _kr_fonts:
    plt.rcParams['font.family'] = _kr_fonts[0]
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'

# ── ONNX ────────────────────────────────────────────────
sess     = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
IN_NAME  = sess.get_inputs()[0].name
OUT_NAME = sess.get_outputs()[0].name
CVD_CH   = {'p': 0.0, 'd': 0.5, 't': 1.0}

def correct(img: np.ndarray, cvd: str) -> np.ndarray:
    SIZE = 256
    x = np.array(Image.fromarray(img).resize((SIZE, SIZE), Image.LANCZOS), dtype=np.float32) / 255.0
    x = np.concatenate([x.transpose(2,0,1),
                        np.full((1,SIZE,SIZE), CVD_CH[cvd], dtype=np.float32)], axis=0)[None]
    y = sess.run([OUT_NAME], {IN_NAME: x})[0][0]
    return np.clip(y.transpose(1,2,0) * 255, 0, 255).astype(np.uint8)

# ── CVD simulator (Brettel 1997) ─────────────────────────
_R2L = np.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.02996,0.18431,1.46720]])
_L2R = np.linalg.inv(_R2L)
_SIM = {
    'p': np.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]]),
    'd': np.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]]),
    't': np.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]]),
}

def simulate(img: np.ndarray, cvd: str) -> np.ndarray:
    H, W = img.shape[:2]
    rgb  = img.astype(np.float64).reshape(-1,3).T / 255.0
    sim  = _L2R @ (_SIM[cvd] @ (_R2L @ rgb))
    return (np.clip(sim.T.reshape(H,W,3), 0, 1) * 255).astype(np.uint8)

# ── load test images ─────────────────────────────────────
def load_images(paths) -> list[tuple[str, np.ndarray]]:
    result = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  [skip] not found: {p}")
            continue
        try:
            img = np.array(Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS))
            result.append((p.stem, img))
            print(f"  loaded: {p.name}")
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
    return result

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('images', nargs='*', help='image file paths')
parser.add_argument('--dir', help='folder of images')
args = parser.parse_args()

if args.dir:
    folder = Path(args.dir)
    paths  = sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.png')) + sorted(folder.glob('*.jpeg'))
    print(f"Loading from folder: {folder} ({len(paths)} images)")
    test_images = load_images(paths[:6])  # max 6
elif args.images:
    print(f"Loading {len(args.images)} specified images")
    test_images = load_images(args.images)
else:
    # fallback: Ishihara plates
    print("No images specified — using Ishihara plates")
    test_images = []
    for pid in ["01","06","08","11"]:
        p = ISH_DIR / f"Ishihara_{pid}.jpg"
        if p.exists():
            img = np.array(Image.open(p).convert('RGB').resize((256,256), Image.LANCZOS))
            test_images.append((f"Ishihara {pid}", img))

if not test_images:
    print("No images loaded. Exiting.")
    sys.exit(1)

n = len(test_images)
print(f"\nGenerating figures for {n} image(s)...\n")

# ════════════════════════════════════════════════════════
# Figure 1: Original vs 3 CVD corrections
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(n, 4, figsize=(4*3.8, n*3.8), facecolor='white')
if n == 1: axes = axes[None,:]

col_info = [
    ('Original',               '#222222'),
    ('Protanopia Correction',  '#c0392b'),
    ('Deuteranopia Correction','#27ae60'),
    ('Tritanopia Correction',  '#2980b9'),
]
for c, (title, color) in enumerate(col_info):
    axes[0][c].set_title(title, fontsize=11, fontweight='bold', color=color, pad=10)

for r, (name, img) in enumerate(test_images):
    axes[r][0].imshow(img)
    axes[r][0].set_ylabel(name, fontsize=9, rotation=90, labelpad=8, va='center')
    for c, cvd in enumerate(['p','d','t'], 1):
        axes[r][c].imshow(correct(img, cvd))
    for c in range(4):
        axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        for spine in axes[r][c].spines.values():
            spine.set_linewidth(0.5); spine.set_color('#cccccc')

fig.suptitle('CVDLens — Color Vision Deficiency Correction Results', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
out1 = OUT_DIR / 'result_correction.png'
fig.savefig(out1, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out1}')

# ════════════════════════════════════════════════════════
# Figure 2: Before / After simulation (per image, main CVD type)
# ════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(n, 4, figsize=(4*3.8, n*3.8), facecolor='white')
if n == 1: axes2 = axes2[None,:]

col2_info = [
    ('Original',                       '#222222'),
    ('CVD Simulation\n(Before Correction)',  '#e74c3c'),
    ('CVDLens Corrected',              '#27ae60'),
    ('CVD Simulation\n(After Correction)',   '#2ecc71'),
]
for c, (title, color) in enumerate(col2_info):
    axes2[0][c].set_title(title, fontsize=10, fontweight='bold', color=color, pad=10)

for r, (name, img) in enumerate(test_images):
    cvd   = 't' if 'Ishihara 11' in name else 'd'
    corr  = correct(img, cvd)
    for c, frame in enumerate([img, simulate(img, cvd), corr, simulate(corr, cvd)]):
        axes2[r][c].imshow(frame)
        axes2[r][c].set_xticks([]); axes2[r][c].set_yticks([])
        for spine in axes2[r][c].spines.values():
            spine.set_linewidth(0.5); spine.set_color('#cccccc')
    axes2[r][0].set_ylabel(name, fontsize=9, rotation=90, labelpad=8, va='center')

fig2.suptitle('CVDLens — Before / After CVD Simulation Comparison\n(Deuteranopia · Tritanopia)', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
out2 = OUT_DIR / 'result_simulation_compare.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out2}')

# ════════════════════════════════════════════════════════
# Figure 3: 3 CVD types — sim before vs sim after (first image)
# ════════════════════════════════════════════════════════
img0 = test_images[0][1]
cvd_list = [('p','Protanopia'), ('d','Deuteranopia'), ('t','Tritanopia')]

fig3, axes3 = plt.subplots(2, 4, figsize=(4*3.8, 2*3.8), facecolor='white')
row_labels = ['Before Correction\n(CVD Simulation of Original)',
              'After Correction\n(CVD Simulation of CVDLens Output)']

for row in range(2):
    axes3[row][0].imshow(img0 if row == 0 else correct(img0, 'd'))
    axes3[row][0].set_title('Original' if row == 0 else 'CVDLens\n(Deuteranopia)', fontsize=10, fontweight='bold')
    axes3[row][0].set_ylabel(row_labels[row], fontsize=8, rotation=90, labelpad=8, va='center')
    for col, (cvd, label) in enumerate(cvd_list, 1):
        frame = simulate(img0, cvd) if row == 0 else simulate(correct(img0, cvd), cvd)
        axes3[row][col].imshow(frame)
        axes3[row][col].set_title(f'{label}\nSimulation', fontsize=10)
    for col in range(4):
        axes3[row][col].set_xticks([]); axes3[row][col].set_yticks([])
        for spine in axes3[row][col].spines.values():
            spine.set_linewidth(0.5); spine.set_color('#cccccc')

fig3.suptitle('CVDLens — Effect on All 3 CVD Types', fontsize=13, fontweight='bold')
plt.tight_layout()
out3 = OUT_DIR / 'result_3cvd_compare.png'
fig3.savefig(out3, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out3}')

print(f'\nAll figures saved to: {OUT_DIR}')
