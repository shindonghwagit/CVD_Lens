"""
PyTorch Dataset (β v1 — selective correction)

Daltonize target 생성은 제거됨. β의 spatial-weighted loss는 GT 이미지를
요구하지 않고, 원본 자체를 자연스러움 기준 + spatial weight 산출 근원으로
사용한다.

각 샘플:
    input_tensor: (4, H, W) — RGB 3ch + CVD type 1ch
    orig_tensor:  (3, H, W) — 원본 RGB [0, 1]
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CVD_TYPES = {
    'p': 0.0,   # Protanopia
    'd': 0.5,   # Deuteranopia
    't': 1.0,   # Tritanopia
}


class CVDDataset(Dataset):
    """
    COCO 이미지에서 직접 읽어 (input_4ch, orig_3ch) 쌍을 반환.

    Args:
        image_paths: COCO 이미지 경로 리스트
        crop_size:   크롭 크기 (기본 256)
        num_crops:   이미지당 크롭 수 (기본 1)
        augment:     좌우 반전 + 가벼운 ColorJitter (학습 시에만)
    """

    def __init__(
        self,
        image_paths: list,
        crop_size: int = 256,
        num_crops: int = 1,
        augment: bool = False,
    ):
        self.crop_size = crop_size
        self.augment = augment

        if len(image_paths) == 0:
            raise FileNotFoundError("이미지 경로 리스트가 비어있음")

        self.samples = [
            (img_path, crop_idx, cvd_key)
            for img_path in image_paths
            for crop_idx in range(num_crops)
            for cvd_key in CVD_TYPES.keys()
        ]

        self.to_tensor = transforms.ToTensor()
        self.jitter = (
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
            if augment else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, _crop_idx, cvd_key = self.samples[idx]
        cvd_value = CVD_TYPES[cvd_key]

        try:
            img_array = np.array(Image.open(img_path).convert('RGB'))
        except Exception:
            img_array = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

        crop = self._extract_crop(img_array, seed=idx)
        crop_pil = Image.fromarray(crop)

        if self.augment:
            if random.random() < 0.5:
                crop_pil = crop_pil.transpose(Image.FLIP_LEFT_RIGHT)
            crop_pil = self.jitter(crop_pil)

        orig_tensor = self.to_tensor(crop_pil)  # (3, H, W) in [0, 1]

        cvd_channel = torch.full(
            (1, self.crop_size, self.crop_size),
            fill_value=cvd_value,
            dtype=torch.float32,
        )
        input_tensor = torch.cat([orig_tensor, cvd_channel], dim=0)  # (4, H, W)

        return input_tensor, orig_tensor

    def _extract_crop(self, img_array: np.ndarray, seed: int) -> np.ndarray:
        h, w = img_array.shape[:2]
        crop_size = self.crop_size

        if h < crop_size or w < crop_size:
            scale = max(crop_size / h, crop_size / w)
            new_h = int(h * scale) + 1
            new_w = int(w * scale) + 1
            img_array = np.array(
                Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS)
            )
            h, w = img_array.shape[:2]

        rng = random.Random(seed)
        top = rng.randint(0, h - crop_size)
        left = rng.randint(0, w - crop_size)
        return img_array[top:top + crop_size, left:left + crop_size]


def build_dataloaders(
    coco_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    num_train: int = 10000,
    num_val: int = 2000,
    num_test: int = 2000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    coco_dir = Path(coco_dir)

    train_imgs = sorted((coco_dir / "train2017").glob("*.jpg"))
    all_val_imgs = sorted((coco_dir / "val2017").glob("*.jpg"))

    if not train_imgs:
        raise FileNotFoundError(f"train2017 이미지 없음: {coco_dir / 'train2017'}")
    if not all_val_imgs:
        raise FileNotFoundError(f"val2017 이미지 없음: {coco_dir / 'val2017'}")

    train_imgs = train_imgs[:num_train]
    val_imgs = all_val_imgs[:num_val]
    test_imgs = all_val_imgs[num_val:num_val + num_test]

    train_ds = CVDDataset(train_imgs, num_crops=2, augment=True)
    val_ds = CVDDataset(val_imgs, num_crops=1, augment=False)
    test_ds = CVDDataset(test_imgs, num_crops=1, augment=False)

    print("Dataset 정보:")
    print(f"  [train] {len(train_ds)}쌍 ({num_train}장 × 2크롭 × 3타입)")
    print(f"  [val]   {len(val_ds)}쌍 ({num_val}장 × 1크롭 × 3타입)")
    print(f"  [test]  {len(test_ds)}쌍 ({num_test}장 × 1크롭 × 3타입)")

    import platform
    safe_workers = 0 if platform.system() == "Windows" else num_workers

    kwargs = {
        "batch_size": batch_size,
        "num_workers": safe_workers,
        "pin_memory": True,
        "persistent_workers": safe_workers > 0,
    }
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )
