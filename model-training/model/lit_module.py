"""
CVDLitModule: 학습/검증/테스트 루프 (β v1 — selective correction)

Batch 구성:
    input_t: (B, 4, H, W)  — RGB 3ch + CVD type 1ch
    orig_t:  (B, 3, H, W)  — 원본 RGB (loss 내 spatial weighting 기준)
"""

import torch
import pytorch_lightning as pl
from torchmetrics.functional.image import (
    structural_similarity_index_measure as ssim_fn,
)

from .network import build_model
from .losses import CVDCorrectionLoss, compute_confusion_weight, simulate_cvd_batch


class CVDLitModule(pl.LightningModule):
    def __init__(
        self,
        lr:           float = 1e-3,
        preserve_w:   float = 0.5,
        visibility_w: float = 1.5,
        structure_w:  float = 0.3,
        color_w:      float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model()
        self.criterion = CVDCorrectionLoss(
            preserve_w=preserve_w,
            visibility_w=visibility_w,
            structure_w=structure_w,
            color_w=color_w,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def _split_batch(batch):
        input_t, orig_t = batch                             # (B,4,H,W), (B,3,H,W)
        cvd_val = input_t[:, 3, 0, 0]                       # (B,)
        return input_t, orig_t, cvd_val

    def _step(self, batch):
        input_t, orig_t, cvd_val = self._split_batch(batch)
        pred = self(input_t)
        loss, components = self.criterion(pred, orig_t, cvd_val)
        return pred, orig_t, cvd_val, loss, components

    def training_step(self, batch, batch_idx):
        _, _, _, loss, components = self._step(batch)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in components.items():
            self.log(f"train_{k}", v, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, orig_t, cvd_val, loss, components = self._step(batch)

        # 검증 지표: 자연스러움(SSIM vs orig) + 가시성(CVD_sim(pred) vs orig)
        # global = 전체 픽셀 평균, weighted = 혼동 영역만 평균 (β 의도 직접 반영)
        ssim_natural = ssim_fn(pred, orig_t, data_range=1.0)
        with torch.no_grad():
            sim_orig = simulate_cvd_batch(orig_t, cvd_val)
            sim_pred = simulate_cvd_batch(pred, cvd_val)
            w = compute_confusion_weight(orig_t, sim_orig)        # (B, 1, H, W)

            diff = (sim_pred - orig_t).abs()
            visibility_l1_global = diff.mean()
            visibility_l1_weighted = (w * diff).sum() / (w.sum() + 1e-6)

        self.log("val_loss",                 loss,                   prog_bar=True, sync_dist=True)
        self.log("val_ssim_natural",         ssim_natural,           prog_bar=True, sync_dist=True)
        self.log("val_visibility_global",    visibility_l1_global,   prog_bar=True, sync_dist=True)
        self.log("val_visibility_weighted",  visibility_l1_weighted, prog_bar=True, sync_dist=True)
        for k, v in components.items():
            self.log(f"val_{k}", v, sync_dist=True)

    def test_step(self, batch, batch_idx):
        pred, orig_t, cvd_val, loss, components = self._step(batch)

        ssim_natural = ssim_fn(pred, orig_t, data_range=1.0)
        with torch.no_grad():
            sim_orig = simulate_cvd_batch(orig_t, cvd_val)
            sim_pred = simulate_cvd_batch(pred, cvd_val)
            w = compute_confusion_weight(orig_t, sim_orig)

            diff = (sim_pred - orig_t).abs()
            visibility_l1_global = diff.mean()
            visibility_l1_weighted = (w * diff).sum() / (w.sum() + 1e-6)

        self.log("test_loss",                loss,                   sync_dist=True)
        self.log("test_ssim_natural",        ssim_natural,           sync_dist=True)
        self.log("test_visibility_global",   visibility_l1_global,   sync_dist=True)
        self.log("test_visibility_weighted", visibility_l1_weighted, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
