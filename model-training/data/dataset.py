"""
Phase 2 - PyTorch Dataset (On-the-fly daltonize)

COCO 이미지에서 직접 읽어 daltonize를 실시간 적용.
.npz 사전 생성 불필요 → 디스크 공간 절약.

입력: (4, 256, 256) — RGB 3ch + CVD type 1ch
타겟: (3, 256, 256) — 보정된 RGB
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ──────────────────────────────────────────────
# Daltonize 알고리즘 (Brettel et al. 1997)
# ──────────────────────────────────────────────

_RGB_TO_LMS = np.array([
    [17.8824,  43.5161,  4.11935],
    [ 3.45565, 27.1554,  3.86714],
    [ 0.02996,  0.18431, 1.46720],
], dtype=np.float64)

_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)

_CVD_MATRICES = {
    'p': np.array([
        [0,      2.02344, -2.52581],
        [0,      1,        0      ],
        [0,      0,        1      ],
    ], dtype=np.float64),
    'd': np.array([
        [1,       0,      0],
        [0.494207, 0,     1.24827],
        [0,        0,     1],
    ], dtype=np.float64),
    't': np.array([
        [1,       0,       0      ],
        [0,       1,       0      ],
        [-0.395913, 0.801109, 0   ],
    ], dtype=np.float64),
}

_ERR2MOD = np.array([
    [0,   0, 0],
    [0.7, 1, 0],
    [0.7, 0, 1],
], dtype=np.float64)

CVD_TYPES = {
    'p': 0.0,   # Protanopia
    'd': 0.5,   # Deuteranopia
    't': 1.0,   # Tritanopia
}


def _apply_daltonize(img_array: np.ndarray, cvd_key: str) -> np.ndarray:
    H, W     = img_array.shape[:2]
    rgb_flat = img_array.astype(np.float64).reshape(-1, 3).T / 255.0
    lms      = _RGB_TO_LMS @ rgb_flat
    sim_lms  = _CVD_MATRICES[cvd_key] @ lms
    sim_rgb  = _LMS_TO_RGB @ sim_lms
    err_mod  = _ERR2MOD @ (rgb_flat - sim_rgb)
    corrected = np.clip(rgb_flat + err_mod, 0, 1)
    return (corrected.T.reshape(H, W, 3) * 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class CVDDataset(Dataset):
    """
    COCO 이미지에서 직접 읽어 daltonize를 실시간 적용하는 Dataset.
    .npz 사전 생성 불필요.

    Args:
        image_paths: COCO 이미지 경로 리스트
        crop_size:   크롭 크기 (기본 256)
        num_crops:   이미지당 크롭 수 (기본 1)
        augment:     augmentation 적용 여부
    """

    def __init__(
        self,
        image_paths: list,
        crop_size: int = 256,
        num_crops: int = 1,
        augment: bool = False,
    ):
        self.crop_size = crop_size
        self.augment   = augment

        if len(image_paths) == 0:
            raise FileNotFoundError("이미지 경로 리스트가 비어있음")

        # (img_path, crop_idx, cvd_key) 샘플 목록
        self.samples = [
            (img_path, crop_idx, cvd_key)
            for img_path in image_paths
            for crop_idx in range(num_crops)
            for cvd_key in CVD_TYPES.keys()
        ]

        self.to_tensor = transforms.ToTensor()

        if augment:
            self._aug_params = {'hflip': True, 'brightness': 0.1, 'contrast': 0.1}
        else:
            self._aug_params = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, crop_idx, cvd_key = self.samples[idx]
        cvd_value = CVD_TYPES[cvd_key]

        try:
            img_array = np.array(Image.open(img_path).convert('RGB'))
        except Exception:
            img_array = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

        crop   = self._extract_crop(img_array, seed=idx)
        target = _apply_daltonize(crop, cvd_key)

        if self._aug_params is not None:
            crop_pil, target_pil = self._apply_augmentation(
                Image.fromarray(crop), Image.fromarray(target)
            )
        else:
            crop_pil   = Image.fromarray(crop)
            target_pil = Image.fromarray(target)

        input_tensor  = self.to_tensor(crop_pil)    # (3, H, W)
        target_tensor = self.to_tensor(target_pil)  # (3, H, W)

        cvd_channel  = torch.full((1, self.crop_size, self.crop_size), fill_value=cvd_value, dtype=torch.float32)
        input_tensor = torch.cat([input_tensor, cvd_channel], dim=0)  # (4, H, W)

        return input_tensor, target_tensor

    def _extract_crop(self, img_array: np.ndarray, seed: int) -> np.ndarray:
        h, w      = img_array.shape[:2]
        crop_size = self.crop_size

        if h < crop_size or w < crop_size:
            scale     = max(crop_size / h, crop_size / w)
            new_h     = int(h * scale) + 1
            new_w     = int(w * scale) + 1
            img_array = np.array(Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS))
            h, w      = img_array.shape[:2]

        rng  = random.Random(seed)
        top  = rng.randint(0, h - crop_size)
        left = rng.randint(0, w - crop_size)
        return img_array[top:top + crop_size, left:left + crop_size]

    def _apply_augmentation(self, input_img, target_img):
        p    = self._aug_params
        seed = random.randint(0, 2**32 - 1)

        def augment_single(img):
            random.seed(seed)
            if p['hflip'] and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            jitter = transforms.ColorJitter(brightness=p['brightness'], contrast=p['contrast'])
            torch.manual_seed(seed)
            fn  = jitter.get_params(jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)
            img = transforms.functional.adjust_brightness(img, fn[1])
            img = transforms.functional.adjust_contrast(img, fn[2])
            return img

        return augment_single(input_img), augment_single(target_img)


# ──────────────────────────────────────────────
# DataLoader 빌더
# ──────────────────────────────────────────────

def build_dataloaders(
    coco_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    num_train: int = 10000,
    num_val: int = 2000,
    num_test: int = 2000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    COCO 이미지에서 직접 DataLoader를 생성한다.

    Args:
        coco_dir:    COCO 루트 (train2017 / val2017 하위 폴더 포함)
        batch_size:  배치 크기
        num_workers: DataLoader 워커 수
        num_train:   학습 이미지 수
        num_val:     검증 이미지 수
        num_test:    테스트 이미지 수
    """
    coco_dir = Path(coco_dir)

    train_imgs   = sorted((coco_dir / "train2017").glob("*.jpg"))
    all_val_imgs = sorted((coco_dir / "val2017").glob("*.jpg"))

    if not train_imgs:
        raise FileNotFoundError(f"train2017 이미지 없음: {coco_dir / 'train2017'}")
    if not all_val_imgs:
        raise FileNotFoundError(f"val2017 이미지 없음: {coco_dir / 'val2017'}")

    train_imgs = train_imgs[:num_train]
    val_imgs   = all_val_imgs[:num_val]
    test_imgs  = all_val_imgs[num_val:num_val + num_test]

    train_ds = CVDDataset(train_imgs, num_crops=2, augment=True)
    val_ds   = CVDDataset(val_imgs,   num_crops=1, augment=False)
    test_ds  = CVDDataset(test_imgs,  num_crops=1, augment=False)

    print("Dataset 정보:")
    print(f"  [train] {len(train_ds)}쌍 ({num_train}장 × 2크롭 × 3타입)")
    print(f"  [val]   {len(val_ds)}쌍 ({num_val}장 × 1크롭 × 3타입)")
    print(f"  [test]  {len(test_ds)}쌍 ({num_test}장 × 1크롭 × 3타입)")

    import platform
    safe_workers = 0 if platform.system() == "Windows" else num_workers

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=safe_workers, pin_memory=True, persistent_workers=(safe_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=safe_workers, pin_memory=True, persistent_workers=(safe_workers > 0))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=safe_workers, pin_memory=True, persistent_workers=(safe_workers > 0))

    return train_loader, val_loader, test_loader
