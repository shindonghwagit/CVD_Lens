"""
Loss for the Phase 2 mask predictor.

The model predicts a soft correction mask. BCE handles pixel-level mask
regression/classification, and Dice loss stabilizes overlap for imbalanced
foreground/background regions.
"""

import torch
import torch.nn as nn


class MaskLoss(nn.Module):
    def __init__(self, bce_w: float = 1.0, dice_w: float = 1.0):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        bce_loss = self.bce(pred, target)

        smooth = 1e-6
        pred_flat = torch.sigmoid(pred).reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )

        total = self.bce_w * bce_loss + self.dice_w * dice_loss
        return total, {"loss_bce": bce_loss.item(), "loss_dice": dice_loss.item()}


if __name__ == "__main__":
    criterion = MaskLoss()
    pred = torch.rand(2, 1, 256, 256)
    target = (torch.rand(2, 1, 256, 256) > 0.5).float()
    loss, components = criterion(pred, target)
    print(f"total: {loss.item():.4f}")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")
