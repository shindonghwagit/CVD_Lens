"""
Loss for the Phase 2 mask predictor.

SmoothL1 treats the task as soft mask regression (each pixel predicts how much
correction to apply), which matches the Lab delta-E pseudo-GT better than BCE.
Dice stabilizes overlap for sparse masks.

Both losses receive sigmoid(logits) — i.e. pred is already in [0, 1].
"""

import torch
import torch.nn as nn


class MaskLoss(nn.Module):
    def __init__(self, l1_w: float = 1.0, dice_w: float = 1.0):
        super().__init__()
        self.l1_w = l1_w
        self.dice_w = dice_w
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l1_loss = self.smooth_l1(pred, target)

        smooth = 1e-6
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )

        total = self.l1_w * l1_loss + self.dice_w * dice_loss
        return total, {"loss_l1": l1_loss.item(), "loss_dice": dice_loss.item()}


if __name__ == "__main__":
    criterion = MaskLoss()
    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    loss, components = criterion(pred, target)
    print(f"total: {loss.item():.4f}")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")
