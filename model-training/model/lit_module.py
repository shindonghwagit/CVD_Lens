"""
CVDLitModule: 학습/검증/테스트 루프
"""

import torch
import pytorch_lightning as pl
from torchmetrics.functional.image import (
    structural_similarity_index_measure as ssim_fn,
    peak_signal_noise_ratio as psnr_fn,
)

from .network import build_model
from .losses import CVDCorrectionLoss


class CVDLitModule(pl.LightningModule):
    def __init__(
        self,
        lr:     float = 1e-3,
        l1_w:   float = 1.0,
        ssim_w: float = 0.5,
        perc_w: float = 0.1,
        sim_w:  float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model     = build_model()
        self.criterion = CVDCorrectionLoss(
            l1_w=l1_w, ssim_w=ssim_w, perc_w=perc_w, sim_w=sim_w
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_t, target_t = batch
        pred = self(input_t)
        loss, components = self.criterion(pred, target_t, input_t)

        self.log("train_loss",      loss,                      prog_bar=True,  on_step=True,  on_epoch=True)
        self.log("train_loss_l1",   components["loss_l1"],     prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_loss_ssim", components["loss_ssim"],   prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_loss_perc", components["loss_perc"],   prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_loss_sim",  components["loss_sim"],    prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_t, target_t = batch
        pred = self(input_t)
        loss, components = self.criterion(pred, target_t, input_t)

        psnr = psnr_fn(pred, target_t, data_range=1.0)
        ssim = ssim_fn(pred, target_t, data_range=1.0)

        self.log("val_loss",      loss,  prog_bar=True, sync_dist=True)
        self.log("val_psnr",      psnr,  prog_bar=True, sync_dist=True)
        self.log("val_ssim",      ssim,  prog_bar=True, sync_dist=True)
        self.log("val_loss_l1",   components["loss_l1"],   sync_dist=True)
        self.log("val_loss_ssim", components["loss_ssim"], sync_dist=True)
        self.log("val_loss_perc", components["loss_perc"], sync_dist=True)
        self.log("val_loss_sim",  components["loss_sim"],  sync_dist=True)

    def test_step(self, batch, batch_idx):
        input_t, target_t = batch
        pred = self(input_t)
        loss, components = self.criterion(pred, target_t, input_t)

        psnr = psnr_fn(pred, target_t, data_range=1.0)
        ssim = ssim_fn(pred, target_t, data_range=1.0)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_psnr", psnr, sync_dist=True)
        self.log("test_ssim", ssim, sync_dist=True)
        self.log("test_loss_sim", components["loss_sim"], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=0.5,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
