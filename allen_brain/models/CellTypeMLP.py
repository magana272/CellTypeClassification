from __future__ import annotations

import torch
from torch import nn

from allen_brain.models.blocks import SEBlock as MLP_SEBlock


class MLPBlock(nn.Module):
    """Simple MLP block: Linear -> ReLU -> Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLP_Model(nn.Module):
    """MLP with variable depth, SE attention on first layer."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        dropout: float = 0.3,
        n_layers: int = 2,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        dims = [max(hidden_dim // (2 ** i), 64) for i in range(n_layers)]

        self.first = MLPBlock(input_dim, dims[0], dropout=dropout)
        self.se = MLP_SEBlock(dims[0])

        layers: list[MLPBlock] = []
        for i in range(1, n_layers):
            layers.append(MLPBlock(dims[i - 1], dims[i], dropout=dropout))
        self.hidden = nn.ModuleList(layers)

        self.classifier = nn.Linear(dims[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.first(x)
        x = self.se(x.unsqueeze(-1)).squeeze(-1)
        for layer in self.hidden:
            x = layer(x)
        return self.classifier(x)



from typing import Any

import optuna

from allen_brain.models.config import (
    TrainConfig,
    MLPHParams,
    MLPModelKwargs,
)


class MLPTrainConfig(TrainConfig):

    def suggest_hparams(self, trial: optuna.trial.Trial) -> MLPHParams:
        lr = trial.suggest_float('lr', 2e-5, 2e-4, log=True)
        wd = trial.suggest_float('weight_decay', 1e-7, 5e-6, log=True)
        dropout = trial.suggest_float('dropout', 0.05, 0.25)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.12)
        optimizer = trial.suggest_categorical('optimizer', ['adamw', 'adam'])
        loss = trial.suggest_categorical('loss', ['cross_entropy', 'focal'])
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512])
        focal_gamma = (trial.suggest_float('focal_gamma', 0.5, 5.0)
                       if loss == 'focal' else 2.0)
        normalize = trial.suggest_categorical(
            'normalize', ['none', 'log', 'standard', 'log+standard'])
        return MLPHParams(
            lr=lr, weight_decay=wd, dropout=dropout,
            label_smoothing=label_smoothing, optimizer=optimizer,
            loss=loss, focal_gamma=focal_gamma, normalize=normalize,
            n_layers=n_layers, hidden_dim=hidden_dim,
        )

    def model_kwargs_from_params(self, params: MLPHParams) -> MLPModelKwargs:
        return MLPModelKwargs(
            dropout=params.dropout,
            n_layers=params.n_layers,
            hidden_dim=params.hidden_dim,
        )

    def infer_model_kwargs(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if 'first.fc.0.weight' in state_dict:
            kw['hidden_dim'] = state_dict['first.fc.0.weight'].shape[0]
        n_hidden = sum(1 for k in state_dict
                       if k.startswith('hidden.') and k.endswith('.fc.0.weight'))
        kw['n_layers'] = 1 + n_hidden
        return kw


TRAIN_CONFIG = MLPTrainConfig()
