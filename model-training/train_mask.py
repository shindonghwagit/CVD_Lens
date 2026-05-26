"""
Train the Phase 2 Attention U-Net mask predictor.

Usage:
    python train_mask.py --coco-dir /path/to/coco
    python train_mask.py --coco-dir /path/to/coco --resume outputs/checkpoints_mask/last.ckpt
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.mask_dataset import build_mask_dataloaders
from model.mask_lit_module import MaskLitModule


def main(args):
    pl.seed_everything(42, workers=True)

    train_loader, val_loader, _ = build_mask_dataloaders(
        coco_dir=args.coco_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
    )

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        module = MaskLitModule.load_from_checkpoint(args.resume)
    else:
        module = MaskLitModule(lr=args.lr, bce_w=args.bce_w, dice_w=args.dice_w)

    ckpt_dir = Path(args.output_dir) / "checkpoints_mask"
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="mask-best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            save_last=True,
            mode="min",
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            mode="min",
            verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {accelerator}")
    print(f"Mask loss weights: BCE={args.bce_w}, Dice={args.dice_w}")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=str(Path(args.output_dir) / "logs"),
            name="cvdlens_mask",
        ),
        log_every_n_steps=50,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
    )

    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)

    print("\nTraining complete")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val_loss:   {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVDLens Phase 2 mask predictor training")
    parser.add_argument("--coco-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bce-w", type=float, default=1.0)
    parser.add_argument("--dice-w", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-train", type=int, default=30000)
    parser.add_argument("--num-val", type=int, default=3000)
    parser.add_argument("--num-test", type=int, default=2000)
    main(parser.parse_args())
