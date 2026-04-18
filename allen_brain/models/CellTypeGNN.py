from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

console = Console()


# ---------------------------------------------------------------------------
# GraphBuilder — encapsulates graph construction utilities
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Builds PyG ``Data`` objects for GNN training and evaluation."""

    def __init__(
        self,
        k_neighbors: int = 15,
        normalize: str | None = None,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.normalize = normalize

    def load_combined_xy(
        self, data_dir: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """Load all splits into a single pre-allocated X buffer."""
        sizes: dict[str, int] = {}
        y_parts: dict[str, np.ndarray] = {}
        for split in ('train', 'val', 'test'):
            y_parts[split] = np.load(os.path.join(data_dir, f'y_{split}.npy'))
            sizes[split] = len(y_parts[split])
        n_total = sum(sizes.values())
        X_train_mm = np.load(os.path.join(data_dir, 'X_train.npy'), mmap_mode='r')
        n_features = X_train_mm.shape[1]

        X_all = np.empty((n_total, n_features), dtype=np.float32)
        offset = 0
        for split in ('train', 'val', 'test'):
            X_mm = (X_train_mm if split == 'train'
                    else np.load(os.path.join(data_dir, f'X_{split}.npy'), mmap_mode='r'))
            X_all[offset:offset + sizes[split]] = X_mm
            offset += sizes[split]
            del X_mm
        y_all = np.concatenate([y_parts['train'], y_parts['val'],
                                y_parts['test']]).astype(np.int64, copy=False)
        return X_all, y_all, sizes

    @staticmethod
    def build_masks(
        sizes: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create boolean train/val/test masks from split sizes."""
        n_total = sum(sizes.values())
        masks = {s: torch.zeros(n_total, dtype=torch.bool) for s in ('train', 'val', 'test')}
        offset = 0
        for s in ('train', 'val', 'test'):
            masks[s][offset:offset + sizes[s]] = True
            offset += sizes[s]
        return masks['train'], masks['val'], masks['test']

    @staticmethod
    def _torch_knn(
        X_all: np.ndarray, k: int, batch_size: int = 256,
    ) -> np.ndarray:
        X = torch.from_numpy(X_all).to('cuda')
        X = X / X.norm(dim=1, keepdim=True)
        indices = torch.empty(X.shape[0], k + 1, dtype=torch.long)
        for i in range(0, X.shape[0], batch_size):
            sim = X[i:i+batch_size] @ X.T
            indices[i:i+batch_size] = sim.topk(k + 1, dim=1).indices
        return indices.cpu().numpy()

    @staticmethod
    def build_knn_edges(X_all: np.ndarray, k: int) -> torch.Tensor:
        """Build symmetric k-NN edge index using cosine distance."""
        n_total = X_all.shape[0]
        console.print(f'Building k={k} cosine-NN graph on {n_total:,} cells...')
        indices = GraphBuilder._torch_knn(X_all, k)
        if indices is None:
            from sklearn.neighbors import NearestNeighbors
            console.print('  (sklearn brute-force — install faiss-cpu for 10x speed)')
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine',
                                    algorithm='brute').fit(X_all)
            _, indices = nbrs.kneighbors(X_all)
        src = np.repeat(np.arange(n_total), k)
        dst = indices[:, 1:].reshape(-1)
        src_sym = np.concatenate([src, dst])
        dst_sym = np.concatenate([dst, src])
        edge_index = torch.tensor(np.stack([src_sym, dst_sym], axis=0), dtype=torch.long)
        return torch.unique(edge_index, dim=1)

    def build_graph_data(self, data_dir: str) -> Data:
        """Load data and build a PyG Data object with k-NN edges and split masks."""
        X_all, y_all, sizes = self.load_combined_xy(data_dir)
        tr_m, vl_m, te_m = self.build_masks(sizes)
        normalize = self.normalize

        if normalize and normalize != 'none':
            n_train = sizes['train']
            if normalize in ('log', 'log+standard'):
                console.print('GNN: applying log normalization...')
                lib = X_all.sum(axis=1, keepdims=True)
                lib = np.maximum(lib, 1.0)
                X_all = np.log1p(X_all / lib * 1e4)
            if normalize in ('standard', 'log+standard'):
                from sklearn.preprocessing import StandardScaler
                console.print('GNN: applying StandardScaler (fit on train)...')
                scaler = StandardScaler()
                X_all[:n_train] = scaler.fit_transform(X_all[:n_train])
                X_all[n_train:] = scaler.transform(X_all[n_train:])

        edge_index = self.build_knn_edges(X_all, self.k_neighbors)
        n_total = X_all.shape[0]
        console.print(f'Graph: {n_total:,} nodes, {edge_index.shape[1]:,} edges, '
              f'avg deg {edge_index.shape[1] / n_total:.1f}')
        return Data(x=torch.from_numpy(X_all), edge_index=edge_index,
                    y=torch.from_numpy(y_all),
                    train_mask=tr_m, val_mask=vl_m, test_mask=te_m)

    @staticmethod
    def build_eval_graph(
        X: np.ndarray, y: np.ndarray, k_neighbors: int = 15,
    ) -> Data:
        """Build a PyG Data object for evaluation (all nodes are test)."""
        edge_index = GraphBuilder.build_knn_edges(X, k_neighbors)
        n = X.shape[0]
        return Data(x=torch.from_numpy(X), edge_index=edge_index,
                    y=torch.from_numpy(y),
                    test_mask=torch.ones(n, dtype=torch.bool))

    @staticmethod
    def masked_class_weights(
        y: torch.Tensor,
        mask: torch.Tensor,
        n_classes: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute balanced class weights using only masked (training) nodes."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        counts = np.bincount(y[mask].cpu().numpy(), minlength=n_classes).astype(np.float32)
        w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
        return w / w.sum() * n_classes


# ---------------------------------------------------------------------------
# Backward-compat module-level wrappers
# ---------------------------------------------------------------------------

def load_combined_xy(data_dir: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    return GraphBuilder().load_combined_xy(data_dir)

def build_masks(sizes: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return GraphBuilder.build_masks(sizes)

def build_knn_edges(X_all: np.ndarray, k: int) -> torch.Tensor:
    return GraphBuilder.build_knn_edges(X_all, k)

def build_graph_data(
    data_dir: str, k_neighbors: int = 15, normalize: str | None = None,
) -> Data:
    return GraphBuilder(k_neighbors=k_neighbors, normalize=normalize).build_graph_data(data_dir)

def build_eval_graph(
    X: np.ndarray, y: np.ndarray, k_neighbors: int = 15,
) -> Data:
    return GraphBuilder.build_eval_graph(X, y, k_neighbors)

def masked_class_weights(
    y: torch.Tensor, mask: torch.Tensor, n_classes: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    return GraphBuilder.masked_class_weights(y, mask, n_classes, device)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class ResidualSAGEBlock(nn.Module):
    """SAGEConv with LayerNorm, GELU, and a residual projection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv    = SAGEConv(in_dim, out_dim)
        self.norm    = nn.LayerNorm(out_dim)
        self.dropout = dropout
        self.proj    = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h + self.proj(x)


class CellTypeGNN(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float = 0.3,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        blocks = [ResidualSAGEBlock(hidden_dim, hidden_dim, dropout)]
        for _ in range(1, n_layers):
            blocks.append(ResidualSAGEBlock(hidden_dim, hidden_dim, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.conv_out = SAGEConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x, edge_index)
        x = F.gelu(self.conv_out(x, edge_index))
        return self.classifier(x)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Pre-classifier embedding (for UMAP / transfer)."""
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x, edge_index)
        return F.gelu(self.conv_out(x, edge_index))


# ---------------------------------------------------------------------------
# Training configuration (used by registry in __init__.py)
# ---------------------------------------------------------------------------

from typing import Any

import optuna

from allen_brain.models.config import (
    TrainConfig,
    GNNHParams,
    GNNModelKwargs,
)


class GNNTrainConfig(TrainConfig):

    def suggest_hparams(self, trial: optuna.trial.Trial) -> GNNHParams:
        lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
        wd = trial.suggest_float('weight_decay', 1e-7, 5e-6, log=True)
        dropout = trial.suggest_float('dropout', 0.25, 0.5)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.15)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
        loss = trial.suggest_categorical('loss', ['focal', 'cross_entropy'])
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256])
        k_neighbors = trial.suggest_int('k_neighbors', 7, 11)
        focal_gamma = (trial.suggest_float('focal_gamma', 1.0, 3.0)
                       if loss == 'focal' else 2.0)
        normalize = trial.suggest_categorical(
            'normalize', ['none', 'log', 'standard', 'log+standard'])
        return GNNHParams(
            lr=lr, weight_decay=wd, dropout=dropout,
            label_smoothing=label_smoothing, optimizer=optimizer,
            loss=loss, focal_gamma=focal_gamma, normalize=normalize,
            n_layers=n_layers, hidden_dim=hidden_dim,
            k_neighbors=k_neighbors,
        )

    def model_kwargs_from_params(self, params: GNNHParams) -> GNNModelKwargs:
        return GNNModelKwargs(
            dropout=params.dropout,
            n_layers=params.n_layers,
            hidden_dim=params.hidden_dim,
        )

    def infer_model_kwargs(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        n_layers = sum(1 for k in state_dict
                       if k.startswith('blocks.') and 'conv.lin_l.weight' in k)
        if n_layers > 0:
            kw['n_layers'] = n_layers
        if 'blocks.0.conv.lin_l.weight' in state_dict:
            kw['hidden_dim'] = state_dict['blocks.0.conv.lin_l.weight'].shape[0]
        elif 'encoder.0.weight' in state_dict:
            kw['hidden_dim'] = state_dict['encoder.0.weight'].shape[0]
        return kw


TRAIN_CONFIG = GNNTrainConfig()
