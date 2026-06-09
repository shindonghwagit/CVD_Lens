"""
PyTorch Lightning module for the Phase 2 mask predictor.
"""

import pytorch_lightning as pl
import torch

from .mask_losses import MaskLoss
from .mask_network import build_mask_model


class MaskLitModule(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, l1_w: float = 1.0, dice_w: float = 0.2):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_mask_model()
        self.criterion = MaskLoss(l1_w=l1_w, dice_w=dice_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))

    @staticmethod
    def _iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_bin = (pred > 0.5).float()
        target_bin = (target > 0.5).float()
        intersection = (pred_bin * target_bin).sum()
        union = pred_bin.sum() + target_bin.sum() - intersection
        return intersection / (union + 1e-6)

    def training_step(self, batch, batch_idx):
        inp, mask_gt = batch
        pred = torch.sigmoid(self.model(inp))
        loss, components = self.criterion(pred, mask_gt)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_l1", components["loss_l1"], on_step=False, on_epoch=True)
        self.log("train_loss_dice", components["loss_dice"], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, mask_gt = batch
        pred = torch.sigmoid(self.model(inp))
        loss, components = self.criterion(pred, mask_gt)
        iou = self._iou(pred, mask_gt)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_iou", iou, prog_bar=True, sync_dist=True)
        self.log("val_loss_l1", components["loss_l1"], prog_bar=True, sync_dist=True)
        self.log("val_loss_dice", components["loss_dice"], prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inp, mask_gt = batch
        pred = torch.sigmoid(self.model(inp))
        loss, _ = self.criterion(pred, mask_gt)
        iou = self._iou(pred, mask_gt)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_iou", iou, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
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
