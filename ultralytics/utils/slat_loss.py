# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Wise-ShapeIoU recovered from the experimental WiseIouLoss implementation."""

import torch
from torch import nn


class WiseShapeIoULoss(nn.Module):
    """Combine ShapeIoU geometry with non-monotonic focusing based on a running IoU-loss mean.

    Args:
        scale (float): Target shape exponent. The recovered experiment call uses 0.0.
        momentum (float): Exponential moving average update coefficient.
        alpha (float): Exponential focusing base.
        delta (float): Focusing reference value.
    """

    def __init__(self, scale=0.0, momentum=0.01, alpha=1.7, delta=2.7):
        """Initialize the recovered focusing parameters and running mean."""
        super().__init__()
        self.scale, self.momentum, self.alpha, self.delta = scale, momentum, alpha, delta
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def forward(self, pred, target):
        """Return one loss per positive xyxy box pair without reducing the sample dimension."""
        # Keep geometry and the moving mean in float32 under mixed precision.
        pred, target = pred.float(), target.float()
        wh1 = (pred[..., 2:] - pred[..., :2]).clamp_min(1e-7)
        wh2 = (target[..., 2:] - target[..., :2]).clamp_min(1e-7)
        intersection = torch.minimum(pred[..., 2:], target[..., 2:]) - torch.maximum(pred[..., :2], target[..., :2])
        intersection = intersection.clamp_min(0).prod(-1)
        union = (wh1.prod(-1) + wh2.prod(-1) - intersection).clamp_min(1e-7)
        iou_loss = 1.0 - intersection / union
        if self.training and torch.is_grad_enabled() and iou_loss.numel():
            with torch.no_grad():
                self.iou_mean.lerp_(iou_loss.detach().mean(), self.momentum)

        # The original ShapeIoU adds epsilon to height before calculating shape weights.
        width1, height1 = wh1[..., 0], wh1[..., 1] + 1e-7
        width2, height2 = wh2[..., 0], wh2[..., 1] + 1e-7
        ws, hs = width2.pow(self.scale), height2.pow(self.scale)
        ww, hh = 2 * ws / (ws + hs), 2 * hs / (ws + hs)
        enclosing = torch.maximum(pred[..., 2:], target[..., 2:]) - torch.minimum(pred[..., :2], target[..., :2])
        center_delta = (pred[..., :2] + pred[..., 2:] - target[..., :2] - target[..., 2:]) / 2
        distance = hh * center_delta[..., 0].square() + ww * center_delta[..., 1].square()
        distance = distance / (enclosing.square().sum(-1) + 1e-7)
        omega_w = hh * (width1 - width2).abs() / torch.maximum(width1, width2)
        omega_h = ww * (height1 - height2).abs() / torch.maximum(height1, height2)
        shape = (1 - (-omega_w).exp()).pow(4) + (1 - (-omega_h).exp()).pow(4)
        beta = iou_loss.detach() / self.iou_mean.clamp_min(1e-7)
        focusing = beta / (self.delta * self.alpha ** (beta - self.delta))
        return (iou_loss + distance + 0.5 * shape) * focusing
