"""Loss functions for cell-type classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('weight', weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


def build_criterion(
    loss_name: str,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    gamma: float = 2.0,
) -> nn.Module:
    """Factory for loss functions."""
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=weight,
                                   label_smoothing=label_smoothing)
    if loss_name == 'focal':
        return FocalLoss(weight=weight, gamma=gamma)
    raise ValueError(f"Unknown loss '{loss_name}'. Use 'cross_entropy' or 'focal'.")
