"""
Phase 2 - 학습 스크립트

사용법:
    # 기본 학습
    python3 train.py --data-dir /home/sch/coco/processed

    # 하이퍼파라미터 조정
    python3 train.py --data-dir /home/sch/coco/processed --batch-size 8 --lr 5e-4

    # 학습 재개
    python3 train.py --data-dir /home/sch/coco/processed --resume checkpoints/last.ckpt
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import build_dataloaders
from model import CVDLitModule


def main(args):
    pl.seed_everything(42, workers=True)

    # ──────────────────────────────────────────
    # 데이터
    # ──────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        coco_dir=args.coco_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
    )

    # ──────────────────────────────────────────
    # 모델
    # ──────────────────────────────────────────
    if args.resume:
        print(f"체크포인트에서 재개: {args.resume}")
        module = CVDLitModule.load_from_checkpoint(args.resume)
    else:
        module = CVDLitModule(
            lr=args.lr,
            preserve_w=args.preserve_w,
            visibility_w=args.visibility_w,
            structure_w=args.structure_w,
            color_w=args.color_w,
        )
    print(
        f"β v1 loss weights: preserve={args.preserve_w}, visibility={args.visibility_w}, "
        f"structure={args.structure_w}, color={args.color_w}"
    )

    # ──────────────────────────────────────────
    # 콜백
    # ──────────────────────────────────────────
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            save_last=True,
            mode="min",
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            mode="min",
            verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    # ──────────────────────────────────────────
    # 트레이너
    # ──────────────────────────────────────────
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"학습 장치: {accelerator} (CUDA: {torch.cuda.is_available()})")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=str(Path(args.output_dir) / "logs"),
            name="cvdlens",
        ),
        log_every_n_steps=50,
        deterministic=False,   # 속도 우선
        precision="16-mixed" if torch.cuda.is_available() else 32,  # AMP
    )

    # ──────────────────────────────────────────
    # 학습
    # ──────────────────────────────────────────
    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)

    print(f"\n학습 완료. 최적 체크포인트: {trainer.checkpoint_callback.best_model_path}")
    print(f"최적 val_loss: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVDLens 모델 학습")

    parser.add_argument("--coco-dir",    type=str, required=True,      help="COCO 루트 디렉토리 (train2017 / val2017 하위 폴더 포함)")
    parser.add_argument("--output-dir",  type=str, default="./outputs", help="체크포인트 + 로그 저장 위치 (기본: ./outputs)")
    parser.add_argument("--resume",      type=str, default=None,        help="재개할 체크포인트 경로 (.ckpt)")
    parser.add_argument("--batch-size",  type=int, default=16,          help="배치 크기 (기본: 16, GPU 메모리 부족 시 8로 줄이기)")
    parser.add_argument("--lr",          type=float, default=1e-3,      help="초기 학습률 (기본: 1e-3)")
    parser.add_argument("--max-epochs",  type=int, default=100,         help="최대 에폭 수 (기본: 100, EarlyStopping 적용)")
    parser.add_argument("--num-workers", type=int, default=4,           help="DataLoader 워커 수 (기본: 4, Windows는 0)")
    parser.add_argument("--num-train",   type=int, default=10000,       help="학습 이미지 수 (기본: 10000)")
    parser.add_argument("--num-val",     type=int, default=2000,        help="검증 이미지 수 (기본: 2000)")
    parser.add_argument("--num-test",    type=int, default=2000,        help="테스트 이미지 수 (기본: 2000)")

    # β v1 spatial-weighted multi-objective loss weights
    parser.add_argument("--preserve-w",   type=float, default=1.0, help="α — 비혼동 영역 원본 보존")
    parser.add_argument("--visibility-w", type=float, default=0.7, help="β — 혼동 영역 CVD 가시성")
    parser.add_argument("--structure-w",  type=float, default=0.5, help="η — 전체 구조 보존 (SSIM)")
    parser.add_argument("--color-w",      type=float, default=0.1, help="γ — 색 분포 보존 (mean/std)")

    args = parser.parse_args()
    main(args)
