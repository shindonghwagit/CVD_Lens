"""
Phase 2 mask predictor visualization.
Usage:
    python visualize_mask.py --ckpt C:/Users/SCH/Downloads/last.ckpt
"""
import sys, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data.mask_dataset import simulate_cvd, make_pseudo_mask, CVD_TYPES
from model.mask_lit_module import MaskLitModule

CVD_NAMES = {'p': 'Protanopia', 'd': 'Deuteranopia', 't': 'Tritanopia'}

def predict_mask(module, original: np.ndarray, cvd_key: str) -> np.ndarray:
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    size = 256
    img = np.array(Image.fromarray(original).resize((size, size), Image.LANCZOS))
    simulated = simulate_cvd(img, cvd_key)
    diff = (np.abs(simulated.astype(np.float32) - img.astype(np.float32))
            .mean(axis=2, keepdims=True) / 255.0)
    cvd_val = CVD_TYPES[cvd_key]

    orig_t = to_tensor(Image.fromarray(img))
    sim_t  = to_tensor(Image.fromarray(simulated))
    diff_t = torch.from_numpy(diff.transpose(2, 0, 1)).float()
    cvd_t  = torch.full((1, size, size), cvd_val, dtype=torch.float32)
    inp = torch.cat([orig_t, sim_t, diff_t, cvd_t], dim=0).unsqueeze(0)

    with torch.no_grad():
        pred = module(inp)[0, 0].cpu().numpy()

    gt = make_pseudo_mask(img, simulated)
    return img, simulated, gt, pred


def main(ckpt_path: str, image_paths: list, out_dir: str):
    print(f"Loading checkpoint: {ckpt_path}")
    module = MaskLitModule.load_from_checkpoint(ckpt_path, map_location='cpu')
    module.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"  [skip] {img_path}")
            continue

        original = np.array(Image.open(img_path).convert('RGB'))
        print(f"  Processing: {img_path.name}")

        fig, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor='white')
        col_titles = ['Original', 'CVD Simulation', 'GT Mask', 'Pred Mask']
        for c, title in enumerate(col_titles):
            axes[0][c].set_title(title, fontsize=11, fontweight='bold', pad=8)

        for r, cvd_key in enumerate(['p', 'd', 't']):
            img, simulated, gt, pred = predict_mask(module, original, cvd_key)

            axes[r][0].imshow(img)
            axes[r][1].imshow(simulated)
            axes[r][2].imshow(gt, cmap='hot', vmin=0, vmax=1)
            axes[r][3].imshow(pred, cmap='hot', vmin=0, vmax=1)

            axes[r][0].set_ylabel(CVD_NAMES[cvd_key], fontsize=10, rotation=90, labelpad=8, va='center')

            gt_mean   = gt.mean()
            pred_mean = pred.mean()
            gt_max    = gt.max()
            pred_max  = pred.max()

            axes[r][2].set_xlabel(f'mean={gt_mean:.3f}  max={gt_max:.3f}', fontsize=8)
            axes[r][3].set_xlabel(f'mean={pred_mean:.3f}  max={pred_max:.3f}', fontsize=8)

            for c in range(4):
                axes[r][c].set_xticks([])
                axes[r][c].set_yticks([])

        fig.suptitle(f'{img_path.stem} — GT vs Pred Mask', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_path = out_dir / f'mask_{img_path.stem}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--images', nargs='*', default=[])
    parser.add_argument('--out-dir', default='./outputs/mask_viz')
    args = parser.parse_args()

    if not args.images:
        ish_dir = Path(__file__).parent.parent / 'cvd-lens' / 'public' / 'ishihara'
        args.images = sorted(ish_dir.glob('*.jpg'))[:4]

    main(args.ckpt, args.images, args.out_dir)
