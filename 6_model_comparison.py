"""Model comparison: AUC curves, accuracy bar charts, confusion matrices, per-class F1."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel

console: Console = Console()
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix

from allen_brain.models import train as T
from allen_brain.models.config import ModelPredictions
from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.cell_data.cell_vis import ModelComparisonVisualizer
from allen_brain.models.CellTypeGNN import GraphBuilder
from allen_brain.models.gnn_train import _collect_graph_probabilities

DATA_DIR: str = 'data/10x'
SAVE_DIR: str = 'figures'
BATCH_SIZE: int = 1024

MODELS: dict[str, tuple[str, bool, bool]] = {
    'MLP': ('CellTypeMLP', True, False),
    'CNN': ('CellTypeCNN', False, False),
    'Transformer': ('CellTypeTOSICA', True, False),
    'GNN': ('CellTypeGNN', False, True),
}


def _load_and_predict(
    model_cls_name: str,
    squeeze_channel: bool,
    is_graph: bool,
) -> ModelPredictions | None:
    """Load best checkpoint and collect probabilities + predictions on test set."""
    ckpt: str | None = T.find_best_ckpt(model_cls_name)
    if ckpt is None:
        return None
    saved_kw: dict[str, Any] = T._load_model_kwargs(ckpt, model_name=model_cls_name)

    if is_graph:
        kw_path: str = os.path.join(os.path.dirname(ckpt), 'model_kwargs.json')
        k: int = 15
        if os.path.exists(kw_path):
            import json
            with open(kw_path) as f:
                kw_data: dict[str, Any] = json.load(f)
            k = kw_data.get('k_neighbors', 15)
        data = GraphBuilder(k_neighbors=k).build_graph_data(DATA_DIR).to(T.DEVICE)
        n_features: int = data.x.shape[1]
        n_classes: int = int(data.y.max().item()) + 1
        model: torch.nn.Module = T.build_model(model_cls_name, n_features, n_classes, **saved_kw)
        model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
        y_probs: np.ndarray
        y_true: np.ndarray
        y_probs, y_true = _collect_graph_probabilities(model, data, data.test_mask)
        y_pred: np.ndarray = y_probs.argmax(1)
        class_names: list[str] = list(np.load(f'{DATA_DIR}/class_names.npy', allow_pickle=True))
    else:
        ds_test = make_dataset(DATA_DIR, split='test')
        hvg_path: str = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
        if os.path.exists(hvg_path):
            hvg_idx: np.ndarray = np.load(hvg_path)
            ds_test.X = np.asarray(ds_test.X[:, hvg_idx])
            ds_test.gene_names = ds_test.gene_names[hvg_idx]
        import pickle as _pickle
        _normalize: str | None = None
        _norm_path: str = os.path.join(os.path.dirname(ckpt), 'normalize.txt')
        if os.path.exists(_norm_path):
            with open(_norm_path) as _f:
                _normalize = _f.read().strip() or None
        _scaler: Any = None
        _scaler_path: str = os.path.join(os.path.dirname(ckpt), 'scaler.pkl')
        if os.path.exists(_scaler_path):
            with open(_scaler_path, 'rb') as _f:
                _scaler = _pickle.load(_f)
        if _normalize:
            ds_test.X = T._apply_normalization_test(
                np.asarray(ds_test.X, dtype=np.float32), _normalize, _scaler)
        n_features = len(ds_test.gene_names)
        model = T.build_model(model_cls_name, n_features, ds_test.n_classes, **saved_kw)
        model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
        pin: bool = T.DEVICE.type == 'cuda'
        loader: DataLoader[Any] = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
        y_probs, y_true = T._collect_probabilities(model, loader, squeeze_channel)
        y_pred = y_probs.argmax(1)
        class_names = list(ds_test.class_names)

    return ModelPredictions(
        y_true=y_true, y_pred=y_pred, y_probs=y_probs,
        class_names=class_names, n_classes=len(class_names),
    )


def _collect_all() -> dict[str, ModelPredictions]:
    results: dict[str, ModelPredictions] = {}
    for name, (cls_name, squeeze, is_graph) in MODELS.items():
        console.print(f'Loading [bold]{name}[/bold]...')
        r: ModelPredictions | None = _load_and_predict(cls_name, squeeze, is_graph)
        if r is not None:
            results[name] = r
            console.print(f'  [green]{name}[/green]: {len(r.y_true)} test samples')
        else:
            console.print(f'  [yellow]{name}[/yellow]: no checkpoint found, skipping')
    return results


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    console.print(Panel(
        '[bold]MODEL COMPARISON[/bold]',
        border_style='cyan', expand=False,
    ))

    results: dict[str, ModelPredictions] = _collect_all()
    if not results:
        console.print('[bold red]No trained models found.[/bold red] Run 4_*.py scripts first.')
        return

    vis = ModelComparisonVisualizer(results, fig_dir=SAVE_DIR)
    vis.plot_roc_per_model()
    vis.plot_roc_comparison()
    vis.plot_accuracy_comparison(SAVE_DIR)
    vis.plot_confusion_matrices()
    vis.plot_per_class_f1()
    vis.plot_metrics_table(SAVE_DIR)

    console.print(Panel(
        f'All figures saved to [bold]{SAVE_DIR}/[/bold]',
        border_style='green', expand=False,
    ))


if __name__ == '__main__':
    main()