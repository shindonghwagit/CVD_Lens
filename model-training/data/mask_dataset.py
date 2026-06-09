"""
Phase 2 dataset for selective color correction.

Input tensor:
    original RGB (3) + CVD simulation RGB (3) + difference map (1)
    + CVD type channel (1) = 8 channels

Target tensor:
    pseudo confusion mask (1 channel), generated from color information loss
    between the original image and its CVD simulation.
"""

import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_RGB_TO_LMS = np.array(
    [
        [17.8824, 43.5161, 4.11935],
        [3.45565, 27.1554, 3.86714],
        [0.02996, 0.18431, 1.46720],
    ],
    dtype=np.float64,
)

_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)

_CVD_MATRICES = {
    "p": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    "d": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]], dtype=np.float64),
    "t": np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]], dtype=np.float64),
}

CVD_TYPES = {"p": 0.0, "d": 0.5, "t": 1.0}


def simulate_cvd(img: np.ndarray, cvd_key: str) -> np.ndarray:
    """Return CVD-simulated RGB image as uint8."""
    h, w = img.shape[:2]
    rgb = img.astype(np.float64).reshape(-1, 3).T / 255.0
    lms = _RGB_TO_LMS @ rgb
    sim = _LMS_TO_RGB @ (_CVD_MATRICES[cvd_key] @ lms)
    return (np.clip(sim.T.reshape(h, w, 3), 0, 1) * 255).astype(np.uint8)


def make_pseudo_mask(
    original: np.ndarray,
    simulated: np.ndarray,
    low: float = 5.0,
    high: float = 25.0,
) -> np.ndarray:
    """Generate a soft pseudo confusion mask in [0, 1] using Lab delta-E.

    Lab delta-E reflects perceptual color difference rather than raw RGB distance.
    Global thresholds (low, high) ensure absolute color loss is preserved across
    images — unlike per-image max normalization which treats every image equally
    regardless of actual confusion magnitude.
    """
    from skimage.color import rgb2lab

    lab_orig = rgb2lab(original.astype(np.uint8))
    lab_sim = rgb2lab(simulated.astype(np.uint8))

    delta_e = np.sqrt(np.sum((lab_orig - lab_sim) ** 2, axis=2)).astype(np.float32)
    mask = np.clip((delta_e - low) / (high - low), 0.0, 1.0)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=5))
    return (np.array(mask_img, dtype=np.float32) / 255.0).clip(0.0, 1.0)


class MaskDataset(Dataset):
    def __init__(
        self,
        image_paths: list,
        crop_size: int = 256,
        num_crops: int = 1,
        augment: bool = False,
    ):
        if len(image_paths) == 0:
            raise FileNotFoundError("image_paths is empty")

        self.crop_size = crop_size
        self.augment = augment
        self.samples = [
            (img_path, crop_idx, cvd_key)
            for img_path in image_paths
            for crop_idx in range(num_crops)
            for cvd_key in CVD_TYPES.keys()
        ]
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, _crop_idx, cvd_key = self.samples[idx]
        cvd_value = CVD_TYPES[cvd_key]

        try:
            img_arr = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            img_arr = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

        original = self._crop(img_arr, seed=idx)
        simulated = simulate_cvd(original, cvd_key)
        diff = (
            np.abs(simulated.astype(np.float32) - original.astype(np.float32))
            .mean(axis=2, keepdims=True)
            / 255.0
        )
        mask_gt = make_pseudo_mask(original, simulated)

        if self.augment and random.random() < 0.5:
            original = original[:, ::-1, :].copy()
            simulated = simulated[:, ::-1, :].copy()
            diff = diff[:, ::-1, :].copy()
            mask_gt = mask_gt[:, ::-1].copy()

        orig_t = self.to_tensor(Image.fromarray(original))
        sim_t = self.to_tensor(Image.fromarray(simulated))
        diff_t = torch.from_numpy(diff.transpose(2, 0, 1)).float()
        cvd_t = torch.full((1, self.crop_size, self.crop_size), cvd_value, dtype=torch.float32)

        inp = torch.cat([orig_t, sim_t, diff_t, cvd_t], dim=0)
        mask_t = torch.from_numpy(mask_gt).unsqueeze(0).float()
        return inp, mask_t

    def _crop(self, img: np.ndarray, seed: int) -> np.ndarray:
        h, w = img.shape[:2]
        size = self.crop_size

        if h < size or w < size:
            scale = max(size / h, size / w)
            img = np.array(
                Image.fromarray(img).resize(
                    (int(w * scale) + 1, int(h * scale) + 1),
                    Image.LANCZOS,
                )
            )
            h, w = img.shape[:2]

        rng = random.Random(seed)
        top = rng.randint(0, h - size)
        left = rng.randint(0, w - size)
        return img[top : top + size, left : left + size]


def build_mask_dataloaders(
    coco_dir: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 4,
    num_train: int = 10000,
    num_val: int = 2000,
    num_test: int = 2000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    import platform

    coco_dir = Path(coco_dir)
    train_imgs = sorted((coco_dir / "train2017").glob("*.jpg"))[:num_train]
    all_val = sorted((coco_dir / "val2017").glob("*.jpg"))
    val_imgs = all_val[:num_val]
    test_imgs = all_val[num_val : num_val + num_test]

    train_ds = MaskDataset(train_imgs, num_crops=2, augment=True)
    val_ds = MaskDataset(val_imgs, num_crops=1, augment=False)
    test_ds = MaskDataset(test_imgs, num_crops=1, augment=False)

    print("Mask dataset:")
    print(f"  train samples: {len(train_ds):,}")
    print(f"  val samples:   {len(val_ds):,}")
    print(f"  test samples:  {len(test_ds):,}")

    nw = 0 if platform.system() == "Windows" else num_workers
    kwargs = {
        "batch_size": batch_size,
        "num_workers": nw,
        "pin_memory": True,
        "persistent_workers": nw > 0,
    }

    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )
