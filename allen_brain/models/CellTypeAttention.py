from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

console = Console()


_DEFAULT_GMT_URL = ('https://data.broadinstitute.org/gsea-msigdb/msigdb/'
                    'release/2023.2.Hs/c2.cp.reactome.v2023.2.Hs.symbols.gmt')


# ---------------------------------------------------------------------------
# PathwayMaskBuilder — encapsulates GMT parsing and mask construction
# ---------------------------------------------------------------------------

class PathwayMaskBuilder:
    """Downloads GMT files and builds (n_genes, n_pathways) binary masks."""

    def __init__(
        self,
        gmt_path: str = 'data/reactome.gmt',
        gmt_url: str = _DEFAULT_GMT_URL,
        min_overlap: int = 5,
        max_pathways: int = 300,
        max_gene_set_size: int = 300,
    ) -> None:
        self.gmt_path = gmt_path
        self.gmt_url = gmt_url
        self.min_overlap = min_overlap
        self.max_pathways = max_pathways
        self.max_gene_set_size = max_gene_set_size

    def download_gmt(self) -> bool:
        os.makedirs(os.path.dirname(self.gmt_path) or '.', exist_ok=True)
        if os.path.exists(self.gmt_path):
            return True
        try:
            console.print(f'Downloading Reactome GMT to {self.gmt_path}...')
            r = requests.get(self.gmt_url, timeout=120)
            r.raise_for_status()
            with open(self.gmt_path, 'wb') as f:
                f.write(r.content)
            return True
        except Exception as e:
            console.print(f'[yellow]GMT download failed[/yellow] ({e}); falling back to identity mask.')
            return False

    def parse_gmt(self) -> dict[str, list[str]]:
        gmt: dict[str, list[str]] = {}
        with open(self.gmt_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                genes = [g for g in parts[2:] if g]
                if len(genes) <= self.max_gene_set_size:
                    gmt[parts[0]] = genes
        return gmt

    def select_pathways(
        self, gene_set: set[str],
    ) -> list[tuple[str, list[str]]]:
        gmt = self.parse_gmt()
        kept: list[tuple[str, list[str]]] = []
        for name, genes in gmt.items():
            overlap = [g for g in genes if g in gene_set]
            if len(overlap) >= self.min_overlap:
                kept.append((name, overlap))
        kept.sort(key=lambda x: len(x[1]), reverse=True)
        return kept[:self.max_pathways]

    @staticmethod
    def _pathways_to_mask(
        kept: list[tuple[str, list[str]]], gene_names: Sequence[str],
    ) -> torch.Tensor:
        gene_to_row = {g: i for i, g in enumerate(gene_names)}
        mask = np.zeros((len(gene_names), len(kept)), dtype=np.float32)
        for j, (_, genes) in enumerate(kept):
            for g in genes:
                i = gene_to_row.get(g)
                if i is not None:
                    mask[i, j] = 1.0
        return torch.from_numpy(mask)

    def build_mask(self, gene_names: Sequence[str]) -> tuple[torch.Tensor, int]:
        """Build a (n_genes, n_pathways) binary mask from Reactome pathways."""
        if not self.download_gmt():
            return torch.eye(len(gene_names)), len(gene_names)
        gmt = self.parse_gmt()
        console.print(f'Gene sets loaded: {len(gmt):,}')
        kept = self.select_pathways(set(gene_names))
        if not kept:
            console.print('[yellow]No pathways matched; using identity mask.[/yellow]')
            return torch.eye(len(gene_names)), len(gene_names)
        mask = self._pathways_to_mask(kept, gene_names)
        console.print(f'Mask: {tuple(mask.shape)}, sparsity {1 - mask.mean().item():.2%}')
        return mask, len(kept)


# Backward-compat module-level wrappers

def _parse_gmt(path: str, max_gene_set_size: int = 300) -> dict[str, list[str]]:
    return PathwayMaskBuilder(gmt_path=path, max_gene_set_size=max_gene_set_size).parse_gmt()


def _select_pathways(
    gmt: dict[str, list[str]], gene_set: set[str],
    min_overlap: int = 5, max_pathways: int = 300,
) -> list[tuple[str, list[str]]]:
    builder = PathwayMaskBuilder(min_overlap=min_overlap, max_pathways=max_pathways)
    # Inject pre-parsed gmt so we don't re-read from disk
    kept: list[tuple[str, list[str]]] = []
    for name, genes in gmt.items():
        overlap = [g for g in genes if g in gene_set]
        if len(overlap) >= min_overlap:
            kept.append((name, overlap))
    kept.sort(key=lambda x: len(x[1]), reverse=True)
    return kept[:max_pathways]


def download_gmt(path: str, url: str) -> bool:
    return PathwayMaskBuilder(gmt_path=path, gmt_url=url).download_gmt()


def _pathways_to_mask(
    kept: list[tuple[str, list[str]]], gene_names: Sequence[str],
) -> torch.Tensor:
    return PathwayMaskBuilder._pathways_to_mask(kept, gene_names)


def build_pathway_mask(
    gene_names: Sequence[str],
    gmt_path: str = 'data/reactome.gmt',
    gmt_url: str = _DEFAULT_GMT_URL,
    min_overlap: int = 5,
    max_pathways: int = 300,
    max_gene_set_size: int = 300,
) -> tuple[torch.Tensor, int]:
    """Build a (n_genes, n_pathways) binary mask from Reactome pathways."""
    builder = PathwayMaskBuilder(
        gmt_path=gmt_path, gmt_url=gmt_url, min_overlap=min_overlap,
        max_pathways=max_pathways, max_gene_set_size=max_gene_set_size,
    )
    return builder.build_mask(gene_names)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class MaskedEmbedding(nn.Module):

    def __init__(
        self, n_genes: int, n_pathways: int, embed_dim: int, mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_genes     = n_genes
        self.n_pathways  = n_pathways
        self.embed_dim   = embed_dim
        self.register_buffer('mask', mask)
        self.weight = nn.Parameter(torch.randn(embed_dim, n_genes, n_pathways) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(embed_dim, n_pathways))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight * self.mask.unsqueeze(0)
        out = torch.einsum('bg,egp->bep', x, W) + self.bias
        return out


class TOSICA(nn.Module):

    def __init__(
        self,
        n_genes: int,
        n_pathways: int,
        n_classes: int,
        mask: torch.Tensor,
        embed_dim: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        unknown_threshold: float = 0.95,
    ) -> None:
        super().__init__()
        self.n_pathways        = n_pathways
        self.embed_dim         = embed_dim
        self.unknown_threshold = unknown_threshold

        self.embedding = MaskedEmbedding(n_genes, n_pathways, embed_dim, mask)

        self.cls_token = nn.Parameter(torch.randn(1, embed_dim, 1))

        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, n_pathways + 1) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self._attn_weights: torch.Tensor | None = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, n_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        tokens = self.embedding(x)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=2)
        tokens = tokens + self.pos_embed
        tokens = tokens.permute(0, 2, 1)
        out = self.transformer(tokens)
        cls_out = out[:, 0, :]
        logits = self.classifier(cls_out)

        if return_attention:
            pathway_out = out[:, 1:, :]
            cls_rep     = cls_out.unsqueeze(1)
            attn_scores = torch.bmm(cls_rep, pathway_out.transpose(1, 2))
            attn_scores = F.softmax(attn_scores.squeeze(1) / math.sqrt(self.embed_dim), dim=-1)
            return logits, attn_scores

        return logits

    def predict_with_unknown(
        self, x: torch.Tensor, threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if threshold is None:
            threshold = self.unknown_threshold
        logits = self.forward(x)
        probs  = F.softmax(logits, dim=-1)
        max_p, preds = probs.max(dim=-1)
        preds[max_p < threshold] = -1
        return preds, max_p



from typing import Any

import optuna

from allen_brain.models.config import (
    TrainConfig,
    TransformerHParams,
    TransformerModelKwargs,
) 


class TransformerTrainConfig(TrainConfig):

    def suggest_hparams(self, trial: optuna.trial.Trial) -> TransformerHParams:
        lr = trial.suggest_float('lr', 2e-3, 2e-2, log=True)
        wd = trial.suggest_float('weight_decay', 2e-4, 2e-3, log=True)
        dropout = trial.suggest_categorical('dropout', [0.4137788066524075])
        label_smoothing = trial.suggest_categorical('label_smoothing', [0.06092275383467414])
        optimizer = trial.suggest_categorical('optimizer', ['adam'])
        loss = trial.suggest_categorical('loss', ['focal'])
        n_layers = trial.suggest_int('n_layers', 1, 5, step=1)
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        embed_dim = trial.suggest_categorical('embed_dim', [48, 64])
        focal_gamma = (trial.suggest_float('focal_gamma', 0.3, 1.5)
                       if loss == 'focal' else 2.0)
        normalize = trial.suggest_categorical(
            'normalize', ['none', 'log', 'standard', 'log+standard'])
        return TransformerHParams(
            lr=lr, weight_decay=wd, dropout=dropout,
            label_smoothing=label_smoothing, optimizer=optimizer,
            loss=loss, focal_gamma=focal_gamma, normalize=normalize,
            n_layers=n_layers, n_heads=n_heads, embed_dim=embed_dim,
        )

    def model_kwargs_from_params(self, params: TransformerHParams) -> TransformerModelKwargs:
        return TransformerModelKwargs(
            dropout=params.dropout,
            n_layers=params.n_layers,
            n_heads=params.n_heads,
            embed_dim=params.embed_dim,
        )

    def infer_model_kwargs(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        n_layers = sum(1 for k in state_dict
                       if k.startswith('transformer.') and 'self_attn.in_proj_weight' in k)
        if n_layers > 0:
            kw['n_layers'] = n_layers
        return kw


TRAIN_CONFIG = TransformerTrainConfig()
