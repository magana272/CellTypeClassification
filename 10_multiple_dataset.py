"""
10_multiple_dataset.py
Train all models on all benchmark datasets, evaluate with full metrics,
generate ROC curves, confusion matrices, and comparative heatmaps.
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score, f1_score,
)
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data

from allen_brain.models import train as T
from allen_brain.models.config import EvalMetrics
from allen_brain.cell_data.cell_load import ALL_DATASETS
from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.models.CellTypeAttention import build_pathway_mask
from allen_brain.models.CellTypeGNN import build_graph_data, masked_class_weights

console: Console = Console()

SEED: int = 42
GMT_PATH: str = 'data/reactome.gmt'
MAX_PATHWAYS: int = 300
MIN_PATHWAY_OVERLAP: int = 5
MAX_GENE_SET_SIZE: int = 300
SAVE_DIR: str = 'figures/multi_dataset'

# ---------------------------------------------------------------------------
# Model configs — thin config shells
# ---------------------------------------------------------------------------
MODELS: dict[str, dict[str, Any]] = {
    'MLP': {
        'model': 'CellTypeMLP', 'squeeze_channel': True, 'is_graph': False,
        'batch_size': 8192, 'epochs': 20, 'n_hvg': 0,
        'normalize': 'log+standard',
    },
    'CNN': {
        'model': 'CellTypeCNN', 'squeeze_channel': False, 'is_graph': False,
        'batch_size': 16384, 'epochs': 30, 'n_hvg': 2000,
        'normalize': 'log+standard',
    },
    'Transformer': {
        'model': 'CellTypeTOSICA', 'squeeze_channel': True, 'is_graph': False,
        'batch_size': 4096, 'epochs': 20, 'n_hvg': 0,
        'normalize': None,
    },
    'GNN': {
        'model': 'CellTypeGNN', 'squeeze_channel': False, 'is_graph': True,
        'batch_size': 256, 'epochs': 200, 'n_hvg': 0,
        'normalize': 'log+standard',
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pathway_kwargs(data_dir: str) -> dict[str, Any]:
    ds = make_dataset(data_dir, split='train')
    mask: torch.Tensor
    n_pathways: int
    mask, n_pathways = build_pathway_mask(
        [str(g) for g in ds.gene_names],
        gmt_path=GMT_PATH, min_overlap=MIN_PATHWAY_OVERLAP,
        max_pathways=MAX_PATHWAYS, max_gene_set_size=MAX_GENE_SET_SIZE)
    return dict(mask=mask, n_pathways=n_pathways)


def _make_cfg(model_cfg: dict[str, Any], data_tag: str) -> dict[str, Any]:
    """Build a training config dict from a model config entry."""
    return {
        'model': model_cfg['model'],
        'seed': SEED,
        'batch_size': model_cfg['batch_size'],
        'accumulation_steps': model_cfg.get('accumulation_steps', 1),
        'n_hvg': model_cfg.get('n_hvg', 0),
        'device': str(T.DEVICE),
        'optimizer': 'adamw',
        'lr': 3e-4,
        'weight_decay': 1e-6,
        'epochs': model_cfg['epochs'],
        'loss': 'cross_entropy',
        'label_smoothing': 0.1,
        'normalize': model_cfg.get('normalize'),
    }


def train_and_eval_model(
    model_label: str,
    model_cfg: dict[str, Any],
    data_dir: str,
    data_tag: str,
) -> EvalMetrics | None:
    """Train one model on one dataset and evaluate. Returns metrics dict."""
    cfg: dict[str, Any] = _make_cfg(model_cfg, data_tag)
    extra_kw: dict[str, Any] | None = None

    if model_cfg['is_graph']:
        data: Data = build_graph_data(data_dir, k_neighbors=10,
                                      normalize=cfg['normalize']).to(T.DEVICE)
        n_classes: int = int(data.y.max().item()) + 1
        weights: torch.Tensor = masked_class_weights(data.y, data.train_mask, n_classes)
        n_features: int = data.x.shape[1]
        class_names: list[str] = list(np.load(f'{data_dir}/class_names.npy', allow_pickle=True))
        del data
        best_acc: float
        ckpt: str
        bp: dict[str, Any]
        best_acc, ckpt, bp = T.train_graph_single(
            cfg, data_dir, n_features, n_classes, weights)
        best_k: int = int(bp.get('k_neighbors', 10))
        best_norm: str | None = bp.get('normalize', cfg['normalize'])
        if best_norm == 'none':
            best_norm = None
        eval_data: Data = build_graph_data(data_dir, k_neighbors=best_k,
                                           normalize=best_norm).to(T.DEVICE)
        metrics: EvalMetrics = T.evaluate_graph(cfg, eval_data, ckpt, n_features, n_classes,
                                                class_names=class_names)
    else:
        if model_cfg['model'] == 'CellTypeTOSICA':
            try:
                extra_kw = _build_pathway_kwargs(data_dir)
            except Exception as e:
                console.print(f'  [yellow]Pathway mask failed: {e}[/yellow]')
                return None
        best_acc, ckpt, bp = T.train_single(
            cfg, data_dir, squeeze_channel=model_cfg['squeeze_channel'],
            extra_model_kwargs=extra_kw)
        metrics = T.evaluate(cfg, data_dir, ckpt,
                             squeeze_channel=model_cfg['squeeze_channel'])

    return metrics


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _plot_accuracy_heatmap(
    all_results: dict[str, dict[str, EvalMetrics | None]],
    save_dir: str,
) -> None:
    """Datasets x Models accuracy heatmap."""
    datasets: list[str] = list(all_results.keys())
    models: list[str] = list(MODELS.keys())
    matrix: np.ndarray = np.full((len(datasets), len(models)), np.nan)
    for i, ds in enumerate(datasets):
        for j, m in enumerate(models):
            r: EvalMetrics | None = all_results[ds].get(m)
            if r is not None:
                matrix[i, j] = r.get('accuracy', np.nan)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), max(4, len(datasets))))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGn',
                xticklabels=models, yticklabels=datasets, ax=ax,
                vmin=0, vmax=1)
    ax.set_title('Model Accuracy Across Datasets')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_heatmap.png'), dpi=150)
    plt.close()
    console.print(f'[green]Saved[/green] {save_dir}/accuracy_heatmap.png')


def _plot_f1_heatmap(
    all_results: dict[str, dict[str, EvalMetrics | None]],
    save_dir: str,
) -> None:
    """Datasets x Models F1-macro heatmap."""
    datasets: list[str] = list(all_results.keys())
    models: list[str] = list(MODELS.keys())
    matrix: np.ndarray = np.full((len(datasets), len(models)), np.nan)
    for i, ds in enumerate(datasets):
        for j, m in enumerate(models):
            r: EvalMetrics | None = all_results[ds].get(m)
            if r is not None:
                matrix[i, j] = r.get('f1_macro', np.nan)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), max(4, len(datasets))))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=models, yticklabels=datasets, ax=ax,
                vmin=0, vmax=1)
    ax.set_title('Model F1-Macro Across Datasets')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_macro_heatmap.png'), dpi=150)
    plt.close()
    console.print(f'[green]Saved[/green] {save_dir}/f1_macro_heatmap.png')


def _plot_confusion_matrices(
    all_results: dict[str, dict[str, EvalMetrics | None]],
    save_dir: str,
) -> None:
    """Per-dataset confusion matrices for each model (if confusion_matrix stored)."""
    for ds_tag, model_results in all_results.items():
        n_models: int = sum(1 for v in model_results.values() if v is not None
                            and 'confusion_matrix' in v)
        if n_models == 0:
            continue
        ncols: int = min(n_models, 4)
        nrows: int = (n_models + ncols - 1) // ncols
        fig: Figure
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes_flat = np.atleast_1d(axes).flat
        idx: int = 0
        for model_label, m in model_results.items():
            if m is None or 'confusion_matrix' not in m:
                continue
            ax: Axes = axes_flat[idx]
            cm: np.ndarray = m['confusion_matrix']
            cm_norm: np.ndarray = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
            sns.heatmap(cm_norm, annot=False, cmap='Blues', ax=ax,
                        vmin=0, vmax=1)
            ax.set_title(f'{model_label}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            idx += 1
        for j in range(idx, nrows * ncols):
            fig.delaxes(list(np.atleast_1d(fig.axes))[j])
        fig.suptitle(f'{ds_tag}: Confusion Matrices', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{ds_tag}_confusion.png'), dpi=150)
        plt.close()


def _print_summary_tables(
    all_results: dict[str, dict[str, EvalMetrics | None]],
) -> None:
    """Print Rich tables summarizing all results."""
    for ds_tag, model_results in all_results.items():
        table: Table = Table(title=f'{ds_tag} Results', show_lines=True)
        table.add_column('Model', style='bold')
        table.add_column('Accuracy', justify='right')
        table.add_column('F1-M', justify='right')
        table.add_column('F1-W', justify='right')
        table.add_column('Prec-W', justify='right')
        table.add_column('Rec-W', justify='right')
        for name, m in model_results.items():
            if m is None:
                table.add_row(name, 'N/A', '', '', '', '')
            else:
                table.add_row(
                    name,
                    f'{m.get("accuracy", 0):.4f}',
                    f'{m.get("f1_macro", 0):.4f}',
                    f'{m.get("f1_weighted", 0):.4f}',
                    f'{m.get("precision_weighted", 0):.4f}',
                    f'{m.get("recall_weighted", 0):.4f}',
                )
        console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_results: dict[str, dict[str, EvalMetrics | None]] = {}

    for ds_name, ds_info in ALL_DATASETS.items():
        data_dir: str = ds_info['dir']

        if not os.path.exists(os.path.join(data_dir, 'X_train.npy')):
            console.print(f'[yellow]Skipping {ds_name}: no data found[/yellow]')
            continue

        console.print(Panel(f'[bold]Dataset: {ds_name}[/bold]',
                            border_style='cyan'))
        all_results[ds_name] = {}

        for model_label, model_cfg in MODELS.items():
            console.print(f'\n  Training {model_label} on {ds_name}...')
            try:
                metrics: EvalMetrics | None = train_and_eval_model(
                    model_label, model_cfg, data_dir, ds_name)
                all_results[ds_name][model_label] = metrics
                if metrics:
                    T.append_results_csv(
                        f'{model_label}_{ds_name}', metrics,
                        csv_path='results_multi_dataset.csv')
            except Exception as e:
                console.print(f'  [red]{model_label} failed: {e}[/red]')
                all_results[ds_name][model_label] = None

    # Comparative visualisations
    _plot_accuracy_heatmap(all_results, SAVE_DIR)
    _plot_f1_heatmap(all_results, SAVE_DIR)
    _plot_confusion_matrices(all_results, SAVE_DIR)
    _print_summary_tables(all_results)


if __name__ == '__main__':
    main()
