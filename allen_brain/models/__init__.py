"""
models — Model registry and factory for cell-type classifiers.

Supported model names:
    CellTypeCNN         5-block 1D CNN  (seq_len, n_classes)
    CellTypeCNN_3Layer  3-block 1D CNN  (seq_len, n_classes)
    TOSICA              Transformer     (n_genes, n_pathways, n_classes, mask, ...)

Usage:
    from models import get_model, AVAILABLE_MODELS, needs_channel_dim
"""

import torch
import numpy as np

from .config import TrainConfig
from .CellTypeMLP import MLP_Model, TRAIN_CONFIG as _mlp_tc
from .CellTypeCNN import CellTypeCNN, ResBlock, TRAIN_CONFIG as _cnn_tc
from .CellTypeAttention import TOSICA, MaskedEmbedding, TRAIN_CONFIG as _tosica_tc
from .CellTypeGNN import CellTypeGNN, TRAIN_CONFIG as _gnn_tc

__all__ = [
    "CellTypeCNN",
    "CellTypeMLP",
    "ResBlock",
    "SEBlock",
    "MLPBlock",
    "CellTypeTOSICA",
    "MaskedEmbedding",
    "AVAILABLE_MODELS",
    "get_model",
    "needs_channel_dim",
]

AVAILABLE_MODELS = ["CellTypeCNN", "CellTypeTOSICA", "CellTypeMLP", "CellTypeGNN"]

_CHANNEL_DIM_MODELS = {"CellTypeCNN"}


def needs_channel_dim(model_name: str) -> bool:
    """Return True if the model expects a leading channel dimension."""
    return model_name in _CHANNEL_DIM_MODELS


def _identity_mask(n_genes: int) -> torch.Tensor:
    """Create a default identity-like pathway mask (each gene = its own pathway)."""
    return torch.eye(n_genes)


def get_model(
    name: str,
    n_genes: int,
    n_classes: int,
    *,
    mask: torch.Tensor | None = None,
    n_pathways: int | None = None,
    embed_dim: int = 48,
    n_heads: int = 4,
    n_layers: int = 2,
    n_stages: int = 3,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    use_checkpointing: bool = False,
) -> torch.nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name       : one of AVAILABLE_MODELS
    n_genes    : number of input genes (= seq_len for CNNs, = n_genes for TOSICA)
    n_classes  : number of cell-type classes
    mask       : (n_genes, n_pathways) binary tensor for TOSICA  (optional)
    n_pathways : override pathway count; defaults to n_genes when mask is None
    n_layers   : depth for MLP, TOSICA, GNN
    n_stages   : depth for CNN (number of ResBlock stages)
    hidden_dim : hidden dimension for MLP (default 512) and GNN (default 256)
    """
    if name == "CellTypeCNN":
        return CellTypeCNN(seq_len=n_genes, n_classes=n_classes,
                           dropout=dropout, n_stages=n_stages,
                           use_checkpointing=use_checkpointing)

    if name == "CellTypeMLP":
        return MLP_Model(input_dim=n_genes, n_classes=n_classes,
                         dropout=dropout, n_layers=n_layers,
                         hidden_dim=hidden_dim)

    if name == "CellTypeGNN":
        return CellTypeGNN(in_dim=n_genes, hidden_dim=hidden_dim,
                           n_classes=n_classes, dropout=dropout,
                           n_layers=n_layers)

    if name == "CellTypeTOSICA":
        if mask is None:
            n_pw = n_pathways or n_genes
            mask = _identity_mask(n_genes) if n_pw == n_genes else torch.ones(n_genes, n_pw)
        else:
            n_pw = mask.shape[1]
        return TOSICA(
            n_genes=n_genes,
            n_pathways=n_pw,
            n_classes=n_classes,
            mask=mask,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown model '{name}'. Choose from: {AVAILABLE_MODELS}"
    )


TRAIN_CONFIGS: dict[str, TrainConfig] = {
    'CellTypeMLP': _mlp_tc,
    'CellTypeCNN': _cnn_tc,
    'CellTypeTOSICA': _tosica_tc,
    'CellTypeGNN': _gnn_tc,
}


def get_train_config(name: str) -> TrainConfig | None:
    """Look up model-specific training config by name."""
    return TRAIN_CONFIGS.get(name)
