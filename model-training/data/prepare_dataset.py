"""
Phase 1 - Week 1: COCO 2017 다운로드 + daltonize 학습 쌍 생성

사용법:
    # 1. COCO 다운로드
    py prepare_dataset.py download --data-dir ./raw

    # 2. 학습 쌍 생성
    py prepare_dataset.py generate --raw-dir ./raw --out-dir ./processed --num-train 20000

    # 3. 검증 (생성된 데이터 확인)
    py prepare_dataset.py verify --processed-dir ./processed
"""

import argparse
import json
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────
# Daltonize 알고리즘 직접 구현 (라이브러리 의존 없음)
# Brettel et al. (1997) + Fidaner daltonization pipeline
# RGB → LMS → 시뮬레이션 → 오차 계산 → 보정
# ──────────────────────────────────────────────

# RGB → LMS 변환 행렬 (Hunt-Pointer-Estevez, D65)
_RGB_TO_LMS = np.array([
    [17.8824,  43.5161,  4.11935],
    [ 3.45565, 27.1554,  3.86714],
    [ 0.02996,  0.18431, 1.46720],
], dtype=np.float64)

_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)

# CVD 타입별 LMS 시뮬레이션 행렬
_CVD_MATRICES = {
    'p': np.array([  # Protanopia: L-cone 없음
        [0,      2.02344, -2.52581],
        [0,      1,        0      ],
        [0,      0,        1      ],
    ], dtype=np.float64),
    'd': np.array([  # Deuteranopia: M-cone 없음
        [1,       0,      0],
        [0.494207, 0,     1.24827],
        [0,        0,     1],
    ], dtype=np.float64),
    't': np.array([  # Tritanopia: S-cone 없음
        [1,       0,       0      ],
        [0,       1,       0      ],
        [-0.395913, 0.801109, 0   ],
    ], dtype=np.float64),
}

# 오차 회전 행렬 (색맹이 볼 수 있는 스펙트럼으로 이동)
_ERR2MOD = np.array([
    [0,   0, 0],
    [0.7, 1, 0],
    [0.7, 0, 1],
], dtype=np.float64)


# CVD 타입 정의 (설계서 5.4 참조)
CVD_TYPES = {
    'p': 0.0,   # Protanopia  (제1색맹, L-cone 이상)
    'd': 0.5,   # Deuteranopia (제2색맹, M-cone 이상)
    't': 1.0,   # Tritanopia  (제3색맹, S-cone 이상)
}

# COCO 2017 다운로드 URL
COCO_URLS = {
    'train': 'http://images.cocodataset.org/zips/train2017.zip',
    'val':   'http://images.cocodataset.org/zips/val2017.zip',
}


# ──────────────────────────────────────────────
# 유틸 함수
# ──────────────────────────────────────────────

def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """URL에서 파일을 스트리밍 다운로드"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"  이미 존재함, 스킵: {dest_path.name}")
        return

    print(f"  다운로드: {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dest_path.name) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """zip 파일 압축 해제"""
    print(f"  압축 해제: {zip_path.name} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)


def extract_random_crops(img_array: np.ndarray, crop_size: int = 256, num_crops: int = 2) -> list:
    """
    이미지에서 랜덤 크롭을 추출한다.
    이미지가 crop_size보다 작으면 패딩 후 크롭.
    """
    h, w = img_array.shape[:2]

    # 이미지가 너무 작으면 리사이즈
    if h < crop_size or w < crop_size:
        scale = max(crop_size / h, crop_size / w)
        new_h, new_w = int(h * scale) + 1, int(w * scale) + 1
        img_pil = Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS)
        img_array = np.array(img_pil)
        h, w = img_array.shape[:2]

    crops = []
    for _ in range(num_crops):
        top  = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        crop = img_array[top:top + crop_size, left:left + crop_size]
        crops.append(crop)

    return crops


def apply_daltonize(img_array: np.ndarray, cvd_key: str) -> np.ndarray:
    """
    Daltonize 알고리즘으로 CVD 보정 이미지 생성 (직접 구현, 라이브러리 불필요).

    파이프라인: RGB → LMS → CVD 시뮬레이션 → 오차 → 보정 → RGB
    (Brettel et al. 1997 + Fidaner daltonization pipeline)

    Args:
        img_array: uint8 (H, W, 3) RGB
        cvd_key:   'p' | 'd' | 't'

    Returns:
        uint8 (H, W, 3) 보정된 RGB
    """
    H, W = img_array.shape[:2]

    # 1. [0, 1] 정규화 후 (3, H*W) 행렬로 변환
    rgb_flat = img_array.astype(np.float64).reshape(-1, 3).T / 255.0  # (3, N)

    # 2. RGB → LMS
    lms = _RGB_TO_LMS @ rgb_flat                         # (3, N)

    # 3. LMS → CVD 시뮬레이션 (색맹 시점)
    sim_lms = _CVD_MATRICES[cvd_key] @ lms               # (3, N)

    # 4. 시뮬레이션 LMS → RGB
    sim_rgb = _LMS_TO_RGB @ sim_lms                      # (3, N)

    # 5. 오차: 색맹이 잃어버린 색상 정보
    err = rgb_flat - sim_rgb                             # (3, N)

    # 6. 오차를 색맹이 볼 수 있는 채널로 회전
    err_mod = _ERR2MOD @ err                             # (3, N)

    # 7. 원본에 보정값 더하기 후 [0,1] 클리핑 → uint8
    corrected = np.clip(rgb_flat + err_mod, 0, 1)
    return (corrected.T.reshape(H, W, 3) * 255).astype(np.uint8)


# ──────────────────────────────────────────────
# 커맨드: download
# ──────────────────────────────────────────────

def cmd_download(args):
    """COCO 2017 train/val 이미지 다운로드 및 압축 해제"""
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for split, url in COCO_URLS.items():
        zip_path = data_dir / f"{split}2017.zip"
        img_dir  = data_dir / f"{split}2017"

        if img_dir.exists() and any(img_dir.iterdir()):
            print(f"[{split}] 이미지 폴더 이미 존재함: {img_dir}")
            continue

        download_file(url, zip_path)
        extract_zip(zip_path, data_dir)

        # zip 삭제 (용량 절약)
        zip_path.unlink()
        print(f"[{split}] 완료. 이미지 수: {len(list(img_dir.glob('*.jpg')))}")

    print("\n✓ COCO 다운로드 완료")


# ──────────────────────────────────────────────
# 커맨드: generate
# ──────────────────────────────────────────────

def generate_split(
    image_paths: list,
    out_dir: Path,
    num_images: int,
    num_crops: int,
    crop_size: int = 256,
    split_name: str = "train",
) -> int:
    """
    이미지 목록에서 (input, target, cvd_type) 학습 쌍을 생성하여 npz로 저장.
    반환값: 생성된 쌍의 수
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이미 생성된 파일 수 확인 (재시작 대비)
    existing = set(p.stem for p in out_dir.glob("*.npz"))

    selected = image_paths[:num_images]
    count = 0
    errors = 0

    pbar = tqdm(selected, desc=f"[{split_name}] 학습 쌍 생성")
    for img_path in pbar:
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
        except Exception as e:
            errors += 1
            continue

        crops = extract_random_crops(img_array, crop_size=crop_size, num_crops=num_crops)

        for crop_idx, crop in enumerate(crops):
            for cvd_key, cvd_value in CVD_TYPES.items():
                pair_id = f"{img_path.stem}_c{crop_idx}_{cvd_key}"

                # 이미 생성된 쌍 스킵 (재시작 안전)
                if pair_id in existing:
                    count += 1
                    continue

                try:
                    target = apply_daltonize(crop, cvd_key)
                except Exception as e:
                    errors += 1
                    continue

                np.savez_compressed(
                    out_dir / f"{pair_id}.npz",
                    input=crop,
                    target=target,
                    cvd_type=np.float32(cvd_value),
                )
                count += 1

        pbar.set_postfix(pairs=count, errors=errors)

    print(f"  → {split_name}: {count}쌍 생성 (오류: {errors})")
    return count


def cmd_generate(args):
    """COCO 이미지에서 daltonize 학습 쌍 생성"""
    raw_dir = Path(args.raw_dir)
    out_dir  = Path(args.out_dir)

    train_imgs = sorted((raw_dir / "train2017").glob("*.jpg"))
    val_imgs   = sorted((raw_dir / "val2017").glob("*.jpg"))

    if not train_imgs:
        print(f"[ERROR] train2017 이미지 없음: {raw_dir / 'train2017'}")
        print("       먼저 `py prepare_dataset.py download` 를 실행하세요.")
        return

    print(f"사용 가능한 이미지: train={len(train_imgs)}, val={len(val_imgs)}")
    print(f"설정: train {args.num_train}장 × {len(CVD_TYPES)}타입 × 2크롭 = 예상 {args.num_train * len(CVD_TYPES) * 2}쌍")
    print(f"      val   {args.num_val}장   × {len(CVD_TYPES)}타입 × 1크롭 = 예상 {args.num_val * len(CVD_TYPES)}쌍\n")

    # Train 생성
    generate_split(train_imgs, out_dir / "train", args.num_train, num_crops=2, split_name="train")

    # Val 생성 (val2017에서 앞 num_val장)
    val_selected = val_imgs[:args.num_val + args.num_test]
    generate_split(val_selected[:args.num_val], out_dir / "val", args.num_val, num_crops=1, split_name="val")

    # Test 생성 (val2017에서 뒤 num_test장)
    generate_split(val_selected[args.num_val:], out_dir / "test", args.num_test, num_crops=1, split_name="test")

    print("\n✓ 학습 쌍 생성 완료")
    print(f"  위치: {out_dir}")


# ──────────────────────────────────────────────
# 커맨드: verify
# ──────────────────────────────────────────────

def cmd_verify(args):
    """생성된 데이터셋 검증 및 샘플 시각화"""
    processed_dir = Path(args.processed_dir)

    for split in ['train', 'val', 'test']:
        split_dir = processed_dir / split
        if not split_dir.exists():
            print(f"[{split}] 없음: {split_dir}")
            continue

        pairs = list(split_dir.glob("*.npz"))
        print(f"\n[{split}] {len(pairs)}쌍")

        if not pairs:
            continue

        # 첫 번째 파일 내용 확인
        sample = np.load(pairs[0])
        print(f"  키: {list(sample.keys())}")
        print(f"  input shape: {sample['input'].shape}, dtype: {sample['input'].dtype}")
        print(f"  target shape: {sample['target'].shape}, dtype: {sample['target'].dtype}")
        print(f"  cvd_type: {sample['cvd_type']}")
        print(f"  파일 예시: {pairs[0].name}")

        # CVD 타입 분포 확인 (처음 100개만 샘플링)
        cvd_counts = {'p': 0, 'd': 0, 't': 0}
        for p in pairs[:100]:
            name = p.stem
            for k in cvd_counts:
                if f"_{k}" in name:
                    cvd_counts[k] += 1
        print(f"  CVD 분포 (샘플 100): {cvd_counts}")

    # 시각화 (matplotlib 있을 때만)
    try:
        import matplotlib.pyplot as plt

        train_dir = processed_dir / "train"
        if not train_dir.exists():
            return

        pairs = list(train_dir.glob("*.npz"))[:9]  # 3 × 3 그리드
        if len(pairs) < 3:
            return

        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        fig.suptitle("CVDLens - 학습 데이터 샘플\n(원본 | Protanopia | Deuteranopia | Tritanopia)", fontsize=12)

        # 같은 원본에서 3가지 CVD 타입 묶기
        # 파일명 패턴: {img_id}_c{crop_idx}_{cvd_key}.npz
        stems = {}
        for p in (processed_dir / "train").glob("*.npz"):
            base = "_".join(p.stem.split("_")[:-1])  # cvd_key 제거
            if base not in stems:
                stems[base] = {}
            cvd_k = p.stem.split("_")[-1]
            stems[base][cvd_k] = p

        shown = 0
        for base, cvd_files in stems.items():
            if shown >= 3:
                break
            if not all(k in cvd_files for k in ['p', 'd', 't']):
                continue

            row = shown
            # 원본 (input은 모두 같으므로 p에서 가져옴)
            data_p = np.load(cvd_files['p'])
            axes[row][0].imshow(data_p['input'])
            axes[row][0].set_title("원본")
            axes[row][0].axis('off')

            for col, (k, label) in enumerate([('p', 'Protanopia'), ('d', 'Deuteranopia'), ('t', 'Tritanopia')]):
                data = np.load(cvd_files[k])
                axes[row][col + 1].imshow(data['target'])
                axes[row][col + 1].set_title(f"{label}\n(cvd={CVD_TYPES[k]})")
                axes[row][col + 1].axis('off')

            shown += 1

        plt.tight_layout()
        save_path = processed_dir / "sample_preview.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"\n✓ 샘플 시각화 저장: {save_path}")
        plt.close()

    except ImportError:
        print("\n(matplotlib 없음 - 시각화 스킵)")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CVDLens Phase 1 - 데이터셋 준비")
    sub = parser.add_subparsers(dest="cmd")

    # download 서브커맨드
    p_dl = sub.add_parser("download", help="COCO 2017 이미지 다운로드")
    p_dl.add_argument("--data-dir", default="./raw", help="다운로드 위치 (기본: ./raw)")

    # generate 서브커맨드
    p_gen = sub.add_parser("generate", help="daltonize 학습 쌍 생성")
    p_gen.add_argument("--raw-dir",       default="./raw",       help="COCO 이미지 위치 (기본: ./raw)")
    p_gen.add_argument("--out-dir",       default="./processed", help="출력 위치 (기본: ./processed)")
    p_gen.add_argument("--num-train",     type=int, default=20000, help="학습 이미지 수 (기본: 20000)")
    p_gen.add_argument("--num-val",       type=int, default=2000,  help="검증 이미지 수 (기본: 2000)")
    p_gen.add_argument("--num-test",      type=int, default=2000,  help="테스트 이미지 수 (기본: 2000)")

    # verify 서브커맨드
    p_ver = sub.add_parser("verify", help="생성된 데이터셋 검증")
    p_ver.add_argument("--processed-dir", default="./processed", help="처리된 데이터 위치 (기본: ./processed)")

    args = parser.parse_args()

    if args.cmd == "download":
        cmd_download(args)
    elif args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "verify":
        cmd_verify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
