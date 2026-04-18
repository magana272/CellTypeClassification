from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from allen_brain.models.blocks import SEBlock as MLP_SEBlock


class ResBlock(nn.Module):
    """Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + SE."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int = 5, dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = kernel // 2
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.drop = nn.Dropout(dropout)
        self.se = MLP_SEBlock(out_ch)
        self.act = nn.ReLU()
        self.proj = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv1(self.act(self.bn1(x)))
        out = self.drop(out)
        out = self.conv2(self.act(self.bn2(out)))
        out = self.se(out)
        return out + identity


class CellTypeCNN(nn.Module):
    """Residual 1D-CNN with variable depth, widening channels, SE attention, and dual-pool head."""

    def __init__(
        self,
        seq_len: int,
        n_classes: int,
        dropout: float = 0.1,
        n_stages: int = 3,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        channels = [min(32 * 2 ** (i // 2), 256) for i in range(n_stages)]
        kernels = [max(7 - 2 * i, 3) for i in range(n_stages)]

        stages: list[nn.Sequential] = []
        in_ch = 32
        for ch, k in zip(channels, kernels):
            stages.append(nn.Sequential(
                ResBlock(in_ch, ch, kernel=k, dropout=dropout),
                nn.MaxPool1d(3),
            ))
            in_ch = ch
        self.stages = nn.ModuleList(stages)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        head_dim = 2 * channels[-1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(head_dim),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(head_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            if self.use_checkpointing and self.training:
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.classifier(x)



from typing import Any

import optuna

from allen_brain.models.config import (
    TrainConfig,
    CNNHParams,
    CNNModelKwargs,
)


class CNNTrainConfig(TrainConfig):

    def suggest_hparams(self, trial: optuna.trial.Trial) -> CNNHParams:
        lr = trial.suggest_float('lr', 2e-3, 2e-2, log=True)
        wd = trial.suggest_float('weight_decay', 5e-5, 5e-4, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.3)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.06)
        optimizer = trial.suggest_categorical('optimizer', ['adamw', 'adam'])
        loss = trial.suggest_categorical('loss', ['cross_entropy', 'focal'])
        n_stages = trial.suggest_int('n_stages', 4, 5)
        focal_gamma = (trial.suggest_float('focal_gamma', 0.5, 5.0)
                       if loss == 'focal' else 2.0)
        normalize = trial.suggest_categorical(
            'normalize', ['none', 'log', 'standard', 'log+standard'])
        return CNNHParams(
            lr=lr, weight_decay=wd, dropout=dropout,
            label_smoothing=label_smoothing, optimizer=optimizer,
            loss=loss, focal_gamma=focal_gamma, normalize=normalize,
            n_stages=n_stages,
        )

    def model_kwargs_from_params(self, params: CNNHParams) -> CNNModelKwargs:
        return CNNModelKwargs(
            dropout=params.dropout,
            n_stages=params.n_stages,
            use_checkpointing=True,
        )

    def infer_model_kwargs(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        n_stages = sum(1 for k in state_dict
                       if k.startswith('stages.') and k.endswith('.0.conv1.weight'))
        if n_stages > 0:
            kw['n_stages'] = n_stages
        return kw


TRAIN_CONFIG = CNNTrainConfig()
