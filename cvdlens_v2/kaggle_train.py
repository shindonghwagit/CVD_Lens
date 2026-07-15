"""
Kaggle entrypoint for CVDCorrectionNet training.

Usage on Kaggle:
    1. Attach public COCO dataset (e.g. "coco-2017-dataset").
    2. Upload this repo as a Kaggle Dataset OR clone via `!git clone ...`
       into /kaggle/working/graduation_project.
    3. In a Kaggle Notebook cell, run:

           !cd /kaggle/working/graduation_project && \\
               python -m cvdlens_v2.kaggle_train --resume

       Or drop this cell verbatim (auto-detects paths, restarts safely):

           import subprocess, sys
           subprocess.check_call([sys.executable, "-m", "cvdlens_v2.kaggle_train",
                                  "--iters", "2000", "--pretrained"])

Resume behaviour:
    Checkpoints go to /kaggle/working/v2_phase1/. `--resume` will pick up
    from `model_latest.pt` — safe across Kaggle's 12-hour session limit.
    Re-run the SAME command in a new session; training continues from the
    last saved step.

GPU:
    Auto-detects `torch.cuda.device_count()` and enables `DataParallel` if
    >1 GPU (Kaggle T4×2 default). Backbone is small enough that single-T4
    with batch 16 is usually sufficient — DataParallel doubles batch, not
    throughput proportionally.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cvdlens_v2.train import main as train_main


# Common paths for COCO on Kaggle.  The exact dataset path depends on
# which public dataset the user attaches; the first that exists wins.
KAGGLE_COCO_CANDIDATES = [
    "/kaggle/input/coco-2017-dataset/coco2017/train2017",
    "/kaggle/input/coco-2017/train2017",
    "/kaggle/input/coco2017/train2017",
    "/kaggle/input/coco/train2017",
]
KAGGLE_OUT_DIR = "/kaggle/working/v2_phase1"
KAGGLE_VAL_DIR_CANDIDATES = [
    "/kaggle/input/coco-2017-dataset/coco2017/val2017",
    "/kaggle/input/coco-2017/val2017",
    "/kaggle/input/coco2017/val2017",
]


def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None


def resolve_paths():
    data_dir = _first_existing(KAGGLE_COCO_CANDIDATES)
    val_dir = _first_existing(KAGGLE_VAL_DIR_CANDIDATES)
    if data_dir is None:
        raise FileNotFoundError(
            "No COCO train2017 directory found. Attach a public COCO "
            "dataset to the notebook.  Tried: "
            + "\n  ".join(KAGGLE_COCO_CANDIDATES))
    if val_dir is not None:
        # The 10-image val bank references /kaggle/input paths; validate_multi
        # currently hard-codes C:/Users/SCH/coco/val2017 — override at env level.
        os.environ["CVDLENS_VAL_DIR"] = val_dir
    return data_dir, val_dir


def _maybe_symlink_val():
    """
    validate_multi/train `_load_val_image` reads from
    C:/Users/SCH/coco/val2017 — override by symlinking or by
    monkey-patching the loader path.  We do the symlink on Linux.
    """
    if os.name != "posix":
        return  # Windows won't hit this
    val_dir = os.environ.get("CVDLENS_VAL_DIR")
    if val_dir is None:
        return
    target = Path("/root/coco/val2017")   # arbitrary — must match loader
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.symlink(val_dir, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Default ON for Kaggle: safe across 12h limit")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lambda-tv", type=float, default=0.03)
    args_kaggle = parser.parse_args()

    data_dir, val_dir = resolve_paths()
    print(f"[kaggle] train dir: {data_dir}")
    print(f"[kaggle] val dir:   {val_dir}")
    print(f"[kaggle] out dir:   {KAGGLE_OUT_DIR}")

    # Set up val path expected by train._load_val_image / validate_multi
    # (both hard-code C:/Users/SCH/coco/val2017).  On Kaggle we monkey-patch.
    if val_dir is not None:
        import cvdlens_v2.train as tr
        _orig = tr._load_val_image

        def _load_kaggle(stem, size, device):
            from PIL import Image
            from torchvision import transforms
            import torch
            p = Path(val_dir) / f"{stem}.jpg"
            img = Image.open(p).convert("RGB").resize(
                (size, size), Image.LANCZOS)
            return transforms.ToTensor()(img).unsqueeze(0).to(device)

        tr._load_val_image = _load_kaggle

    # Build train.py args
    train_args = argparse.Namespace(
        data_dir=data_dir,
        out_dir=KAGGLE_OUT_DIR,
        crop=256,
        batch_size=args_kaggle.batch_size,
        workers=args_kaggle.workers,
        lr=args_kaggle.lr,
        weight_decay=0.0,
        iters=args_kaggle.iters,
        epochs=0,
        val_every=args_kaggle.val_every,
        log_every=args_kaggle.log_every,
        lambda_tv=args_kaggle.lambda_tv,
        pretrained=args_kaggle.pretrained,
        use_lpips=False,       # heavy on CPU/small GPUs; enable later if needed
        seed=0,
        smoke=False,
        resume=args_kaggle.resume,
        data_parallel=True,     # auto-uses N GPUs if available
    )
    Path(KAGGLE_OUT_DIR).mkdir(parents=True, exist_ok=True)
    train_main(train_args)


if __name__ == "__main__":
    main()
