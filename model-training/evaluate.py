"""
CVDLens 모델 평가 스크립트
1. PSNR / SSIM — ONNX 모델 출력 vs Brettel daltonize 알고리즘 (ground truth)
2. 이시하라 가독성 — 보정 전후 숫자 인식률 (EasyOCR)
"""

import sys, os, math
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ── 경로 설정 ──────────────────────────────────────────
ROOT      = Path(__file__).parent
ONNX_PATH = ROOT.parent / "cvd-lens" / "inference" / "model" / "cvdlens_fp32.onnx"
ISH_DIR   = ROOT.parent / "cvd-lens" / "public" / "ishihara"
OUT_DIR   = ROOT / "outputs" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Brettel daltonize (학습 시 사용한 알고리즘) ────────
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
CVD_CHANNEL = {'p': 0.0, 'd': 0.5, 't': 1.0}
CVD_LABEL   = {'p': '적색맹(Protanopia)', 'd': '녹색맹(Deuteranopia)', 't': '청색맹(Tritanopia)'}

def daltonize(img_arr: np.ndarray, cvd: str) -> np.ndarray:
    H, W = img_arr.shape[:2]
    rgb  = img_arr.astype(np.float64).reshape(-1,3).T / 255.0
    lms  = _RGB_TO_LMS @ rgb
    sim  = _LMS_TO_RGB @ (_CVD_MATRICES[cvd] @ lms)
    out  = np.clip(rgb + _ERR2MOD @ (rgb - sim), 0, 1)
    return (out.T.reshape(H, W, 3) * 255).astype(np.uint8)

# ── ONNX 추론 ──────────────────────────────────────────
sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
INPUT_NAME  = sess.get_inputs()[0].name
OUTPUT_NAME = sess.get_outputs()[0].name

def infer(img_arr: np.ndarray, cvd: str) -> np.ndarray:
    """img_arr: HxWx3 uint8 → HxWx3 uint8 보정 결과"""
    SIZE = 256
    img  = np.array(Image.fromarray(img_arr).resize((SIZE, SIZE), Image.LANCZOS))
    x    = img.astype(np.float32) / 255.0  # HxWx3
    x    = np.transpose(x, (2,0,1))        # 3xHxW
    cvd_ch = np.full((1, SIZE, SIZE), CVD_CHANNEL[cvd], dtype=np.float32)
    x    = np.concatenate([x, cvd_ch], axis=0)[None]  # 1x4xHxW
    y    = sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0][0]  # 3xHxW
    y    = np.transpose(y, (1,2,0))
    return np.clip(y * 255, 0, 255).astype(np.uint8)

# ══════════════════════════════════════════════════════
# Part 1 : PSNR / SSIM 측정
# ══════════════════════════════════════════════════════
print("=" * 60)
print("Part 1 : PSNR / SSIM (모델 출력 vs Brettel 알고리즘)")
print("=" * 60)

# 테스트 이미지 수집 (sample_result.png + Ishihara 컬러 이미지)
# 자연 이미지처럼 보이는 이미지를 합성 생성 (COCO 없으므로)
def make_colorful_patches(n=20, size=256, seed=42):
    """다양한 색감의 패치를 합성 생성 (평가용)"""
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(n):
        # 여러 색 영역으로 구성된 패치
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        num_regions = rng.integers(4, 9)
        for _ in range(num_regions):
            x1, y1 = rng.integers(0, size-20, 2)
            x2, y2 = x1+rng.integers(20, size-x1), y1+rng.integers(20, size-y1)
            color = rng.integers(30, 255, 3)
            patch[x1:x2, y1:y2] = color
        # 가우시안 스무딩 효과 (PIL)
        pil = Image.fromarray(patch).filter(__import__('PIL.ImageFilter', fromlist=['GaussianBlur']).GaussianBlur(radius=8))
        imgs.append(np.array(pil))
    return imgs

# 실제 이미지도 포함 (Ishihara 플레이트를 natural image 대용)
real_imgs = []
for p in sorted(ISH_DIR.glob("*.jpg"))[:5]:
    real_imgs.append(np.array(Image.open(p).convert('RGB')))
for p in sorted(ISH_DIR.glob("*.png"))[:2]:
    real_imgs.append(np.array(Image.open(p).convert('RGB')))

synth_imgs = make_colorful_patches(n=20)
test_imgs  = synth_imgs + real_imgs
print(f"테스트 이미지: 합성 {len(synth_imgs)}장 + 실제 {len(real_imgs)}장 = {len(test_imgs)}장\n")

results = {cvd: {'psnr': [], 'ssim': []} for cvd in ['p','d','t']}

for i, img in enumerate(test_imgs):
    for cvd in ['p', 'd', 't']:
        gt   = daltonize(img, cvd)          # ground truth (Brettel)
        pred = infer(img, cvd)              # 모델 출력

        # ground truth도 256으로 맞추기
        gt_256 = np.array(Image.fromarray(gt).resize((256,256), Image.LANCZOS))

        p_val = psnr(gt_256, pred, data_range=255)
        s_val = ssim(gt_256, pred, data_range=255, channel_axis=2)
        results[cvd]['psnr'].append(p_val)
        results[cvd]['ssim'].append(s_val)

print(f"{'CVD 타입':<20} {'PSNR (dB)':>12} {'SSIM':>10}")
print("-" * 44)
all_psnr, all_ssim = [], []
for cvd in ['p','d','t']:
    p_mean = np.mean(results[cvd]['psnr'])
    s_mean = np.mean(results[cvd]['ssim'])
    all_psnr.extend(results[cvd]['psnr'])
    all_ssim.extend(results[cvd]['ssim'])
    print(f"{CVD_LABEL[cvd]:<20} {p_mean:>12.2f} {s_mean:>10.4f}")

print("-" * 44)
print(f"{'전체 평균':<20} {np.mean(all_psnr):>12.2f} {np.mean(all_ssim):>10.4f}")

print("\n[참고 기준]")
print("  PSNR 30dB+  : 양호 / 35dB+ : 우수 / 40dB+ : 매우 우수")
print("  SSIM 0.90+  : 양호 / 0.95+ : 우수 / 0.98+ : 매우 우수")

# ══════════════════════════════════════════════════════
# Part 2 : 이시하라 가독성 (EasyOCR)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 2 : 이시하라 가독성 테스트 (EasyOCR)")
print("=" * 60)

try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    PLATES = [
        ("01", "74",  "d"), ("03", "16",  "d"), ("04", "2",   "d"),
        ("05", "29",  "d"), ("06", "7",   "d"), ("07", "45",  "d"),
        ("08", "5",   "d"), ("09", "97",  "d"), ("10", "8",   "d"),
        ("11", "42",  "t"), ("12", "3",   "d"), ("13", "42",  "d"),
        ("14", "27",  "d"), ("15", "12",  "d"),
    ]

    def read_number(img_arr: np.ndarray) -> str:
        results = reader.readtext(img_arr, detail=0, allowlist='0123456789')
        return ''.join(results).strip() if results else ""

    print(f"\n{'플레이트':<6} {'정답':>5} {'원본 인식':>10} {'보정 후 인식':>12} {'개선':>6}")
    print("-" * 45)

    orig_correct = 0
    corr_correct = 0
    total = 0

    for plate_id, answer, cvd in PLATES:
        ext = "png" if plate_id in ("13","14","15") else "jpg"
        path = ISH_DIR / f"Ishihara_{plate_id}.{ext}"
        if not path.exists():
            continue

        img_arr = np.array(Image.open(path).convert('RGB'))
        corr    = infer(img_arr, cvd)

        orig_text = read_number(img_arr)
        corr_text = read_number(corr)

        orig_ok = answer in orig_text
        corr_ok = answer in corr_text
        if orig_ok: orig_correct += 1
        if corr_ok: corr_correct += 1
        total += 1

        improved = "↑" if (not orig_ok and corr_ok) else ("=" if orig_ok == corr_ok else "↓")
        print(f"  {plate_id:<4} {answer:>5} {orig_text or '-':>10} {corr_text or '-':>12}  {improved}")

    print("-" * 45)
    print(f"원본 인식률  : {orig_correct}/{total} ({orig_correct/total*100:.0f}%)")
    print(f"보정 후 인식률: {corr_correct}/{total} ({corr_correct/total*100:.0f}%)")

    # 보정 이미지 저장
    for plate_id, answer, cvd in PLATES[:4]:
        ext = "png" if plate_id in ("13","14","15") else "jpg"
        path = ISH_DIR / f"Ishihara_{plate_id}.{ext}"
        if not path.exists(): continue
        img_arr = np.array(Image.open(path).convert('RGB'))
        corr    = infer(img_arr, cvd)
        Image.fromarray(corr).save(OUT_DIR / f"corrected_{plate_id}_{cvd}.png")
    print(f"\n보정 이미지 저장: {OUT_DIR}")

except ImportError:
    print("EasyOCR 미설치 → OCR 건너뜀")
    print("수동 확인용 보정 이미지를 저장합니다...")

    PLATES = [
        ("01","74","d"),("03","16","d"),("05","29","d"),
        ("06","7","d"), ("08","5","d"), ("11","42","t"),
    ]
    for plate_id, answer, cvd in PLATES:
        ext = "png" if plate_id in ("13","14","15") else "jpg"
        path = ISH_DIR / f"Ishihara_{plate_id}.{ext}"
        if not path.exists(): continue
        img_arr = np.array(Image.open(path).convert('RGB'))
        corr    = infer(img_arr, cvd)
        # 원본과 보정 이미지를 나란히 붙여서 저장
        orig_256 = np.array(Image.fromarray(img_arr).resize((256,256)))
        side_by_side = np.concatenate([orig_256, corr], axis=1)
        Image.fromarray(side_by_side).save(OUT_DIR / f"compare_{plate_id}_{cvd}.png")
    print(f"비교 이미지 저장 완료: {OUT_DIR}")
    print("outputs/eval/ 폴더에서 눈으로 확인해 주세요.")

# ══════════════════════════════════════════════════════
# Part 3 : CVD 시뮬레이션 기반 SSIM
#   sim(보정본) vs 원본 의 SSIM  >  sim(원본) vs 원본 의 SSIM
#   → 색각 이상자가 보정 이미지를 봤을 때 정상인이 원본을 보는 것과
#     얼마나 가까워졌는지 측정
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 3 : CVD Simulation-based SSIM")
print("  SSIM( sim(corrected), original ) vs SSIM( sim(original), original )")
print("=" * 60)

_R2L = np.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.02996,0.18431,1.46720]], dtype=np.float64)
_L2R = np.linalg.inv(_R2L)
_SIM_MAT = {
    'p': np.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]], dtype=np.float64),
    'd': np.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]], dtype=np.float64),
    't': np.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]], dtype=np.float64),
}

def simulate_cvd(img_arr: np.ndarray, cvd: str) -> np.ndarray:
    H, W = img_arr.shape[:2]
    rgb  = img_arr.astype(np.float64).reshape(-1,3).T / 255.0
    sim  = _L2R @ (_SIM_MAT[cvd] @ (_R2L @ rgb))
    return (np.clip(sim.T.reshape(H,W,3), 0, 1) * 255).astype(np.uint8)

sim_results = {cvd: {'before': [], 'after': []} for cvd in ['p','d','t']}

for img in test_imgs:
    img_256 = np.array(Image.fromarray(img).resize((256,256), Image.LANCZOS))
    for cvd in ['p', 'd', 't']:
        corr     = infer(img, cvd)                   # 모델 보정본 (256x256)
        sim_orig = simulate_cvd(img_256, cvd)        # CVD 사용자가 원본을 볼 때
        sim_corr = simulate_cvd(corr, cvd)           # CVD 사용자가 보정본을 볼 때

        ssim_before = ssim(sim_orig, img_256, data_range=255, channel_axis=2)
        ssim_after  = ssim(sim_corr, img_256, data_range=255, channel_axis=2)
        sim_results[cvd]['before'].append(ssim_before)
        sim_results[cvd]['after'].append(ssim_after)

print(f"\n{'CVD 타입':<20} {'보정 전 SSIM':>14} {'보정 후 SSIM':>14} {'개선폭':>10}")
print("-" * 62)
all_before, all_after = [], []
for cvd in ['p','d','t']:
    b = np.mean(sim_results[cvd]['before'])
    a = np.mean(sim_results[cvd]['after'])
    all_before.append(b); all_after.append(a)
    print(f"{CVD_LABEL[cvd]:<20} {b:>14.4f} {a:>14.4f} {a-b:>+10.4f}")
print("-" * 62)
print(f"{'전체 평균':<20} {np.mean(all_before):>14.4f} {np.mean(all_after):>14.4f} {np.mean(all_after)-np.mean(all_before):>+10.4f}")
print("\n해석: 보정 후 SSIM > 보정 전 SSIM → CVD 사용자가 보정본을 통해 원본에 가까운 정보를 얻음")
