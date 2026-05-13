"""
학습된 모델로 샘플 이미지 추론 - 결과 시각화

사용법:
    py infer_sample.py --ckpt C:/Users/SCH/Downloads/best-epoch=026-val_loss=0.0265.ckpt --image <이미지경로>
    py infer_sample.py --ckpt C:/Users/SCH/Downloads/best-epoch=026-val_loss=0.0265.ckpt  # 인터넷에서 샘플 다운로드
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from model import CVDLitModule

# prepare_dataset.py 의 daltonize 구현 재사용
_RGB_TO_LMS = np.array([
    [17.8824,  43.5161,  4.11935],
    [ 3.45565, 27.1554,  3.86714],
    [ 0.02996,  0.18431, 1.46720],
], dtype=np.float64)
_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)
_CVD_MATRICES = {
    'p': np.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]], dtype=np.float64),
    'd': np.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]], dtype=np.float64),
    't': np.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]], dtype=np.float64),
}
_ERR2MOD = np.array([[0,0,0],[0.7,1,0],[0.7,0,1]], dtype=np.float64)

CVD_TYPES = {'p': 0.0, 'd': 0.5, 't': 1.0}
CVD_LABELS = {'p': 'Protanopia', 'd': 'Deuteranopia', 't': 'Tritanopia'}


def simulate_cvd(img_array: np.ndarray, cvd_key: str) -> np.ndarray:
    """색맹 시뮬레이션 (보정 전 - 색맹이 실제로 보는 화면)"""
    H, W = img_array.shape[:2]
    rgb_flat = img_array.astype(np.float64).reshape(-1, 3).T / 255.0
    lms = _RGB_TO_LMS @ rgb_flat
    sim_lms = _CVD_MATRICES[cvd_key] @ lms
    sim_rgb = np.clip((_LMS_TO_RGB @ sim_lms).T.reshape(H, W, 3), 0, 1)
    return (sim_rgb * 255).astype(np.uint8)


def daltonize(img_array: np.ndarray, cvd_key: str) -> np.ndarray:
    """Daltonize 보정 (전통 알고리즘)"""
    H, W = img_array.shape[:2]
    rgb_flat = img_array.astype(np.float64).reshape(-1, 3).T / 255.0
    lms = _RGB_TO_LMS @ rgb_flat
    sim_lms = _CVD_MATRICES[cvd_key] @ lms
    sim_rgb = _LMS_TO_RGB @ sim_lms
    err = rgb_flat - sim_rgb
    err_mod = _ERR2MOD @ err
    corrected = np.clip(rgb_flat + err_mod, 0, 1)
    return (corrected.T.reshape(H, W, 3) * 255).astype(np.uint8)


def run_model(net, img_array: np.ndarray, cvd_key: str) -> np.ndarray:
    """CVDLens 모델 추론"""
    cvd_val = CVD_TYPES[cvd_key]
    img_t = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0  # (3, H, W)
    cvd_ch = torch.full((1, img_t.shape[1], img_t.shape[2]), cvd_val)     # (1, H, W)
    inp = torch.cat([img_t, cvd_ch], dim=0).unsqueeze(0)                  # (1, 4, H, W)

    with torch.no_grad():
        out = net(inp).squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def crop_center(img_array: np.ndarray, size: int = 256) -> np.ndarray:
    h, w = img_array.shape[:2]
    if h < size or w < size:
        scale = max(size / h, size / w)
        new_h, new_w = int(h * scale) + 1, int(w * scale) + 1
        img_array = np.array(Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS))
        h, w = img_array.shape[:2]
    top  = (h - size) // 2
    left = (w - size) // 2
    return img_array[top:top+size, left:left+size]


def main(args):
    # ── 이미지 로드 ────────────────────────────────────────
    if args.image:
        img = Image.open(args.image).convert("RGB")
        img_array = crop_center(np.array(img), 256)
        print(f"이미지 로드: {args.image}")
    else:
        # 인터넷 샘플 다운로드
        import urllib.request
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        tmp = Path("outputs/sample_input.png")
        tmp.parent.mkdir(exist_ok=True)
        print(f"샘플 이미지 다운로드...")
        urllib.request.urlretrieve(url, tmp)
        img = Image.open(tmp).convert("RGB")
        img_array = crop_center(np.array(img), 256)

    # ── 모델 로드 ──────────────────────────────────────────
    print(f"모델 로드: {args.ckpt}")
    module = CVDLitModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    module.eval()
    net = module.model

    # ── 시각화: 3가지 CVD 타입 × (원본 / 색맹시뮬 / Daltonize / CVDLens) ──
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("CVDLens 보정 결과 비교\n(256×256 crop)", fontsize=14, fontweight="bold")

    col_titles = ["원본", "색맹 시뮬레이션", "Daltonize (기존)", "CVDLens (AI)"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold")

    for row, (cvd_key, cvd_label) in enumerate(CVD_LABELS.items()):
        simulated  = simulate_cvd(img_array, cvd_key)
        daltonized = daltonize(img_array, cvd_key)
        corrected  = run_model(net, img_array, cvd_key)

        axes[row][0].imshow(img_array);   axes[row][0].set_ylabel(cvd_label, fontsize=11)
        axes[row][1].imshow(simulated)
        axes[row][2].imshow(daltonized)
        axes[row][3].imshow(corrected)

        for ax in axes[row]:
            ax.axis("off")

        # PSNR 계산 (원본 vs 보정)
        mse_dal = np.mean((img_array.astype(float) - daltonized.astype(float))**2)
        mse_cvd = np.mean((img_array.astype(float) - corrected.astype(float))**2)
        psnr_dal = 10 * np.log10(255**2 / mse_dal) if mse_dal > 0 else float('inf')
        psnr_cvd = 10 * np.log10(255**2 / mse_cvd) if mse_cvd > 0 else float('inf')
        axes[row][2].set_xlabel(f"PSNR: {psnr_dal:.1f} dB", fontsize=9)
        axes[row][3].set_xlabel(f"PSNR: {psnr_cvd:.1f} dB", fontsize=9)

    plt.tight_layout()
    out_path = Path("outputs/sample_result.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\n결과 저장: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  required=True, help="체크포인트 경로")
    parser.add_argument("--image", default=None,  help="입력 이미지 경로 (없으면 샘플 다운로드)")
    main(parser.parse_args())
