"""Loss functions for cell-type classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0,
                 reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


def build_criterion(loss_name, weight=None, label_smoothing=0.0, gamma=2.0):
    """Factory for loss functions.

    Parameters
    ----------
    loss_name : str
        "cross_entropy" or "focal"
    weight : Tensor, optional
        Class weights.
    label_smoothing : float
        Label smoothing (cross_entropy only).
    gamma : float
        Focusing parameter (focal only).
    """
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=weight,
                                   label_smoothing=label_smoothing)
    if loss_name == 'focal':
        return FocalLoss(weight=weight, gamma=gamma)
    raise ValueError(f"Unknown loss '{loss_name}'. Use 'cross_entropy' or 'focal'.")
