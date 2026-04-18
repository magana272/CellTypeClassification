"""Cross-dataset evaluation: N x N — every (train, eval) pair for all models."""

from __future__ import annotations

import os
import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader, TensorDataset

from allen_brain.models import train as T
from allen_brain.models.config import EvalMetrics
from allen_brain.cell_data.cell_load import ALL_DATASETS
from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.cell_data.cell_preprocess import align_genes
from allen_brain.models.CellTypeAttention import build_pathway_mask
from allen_brain.models.CellTypeGNN import build_eval_graph

K_NEIGHBORS = 15
BATCH_SIZE = 1024

# GMT config for Transformer pathway mask
GMT_PATH = 'data/reactome.gmt'
MAX_PATHWAYS = 300
MIN_PATHWAY_OVERLAP = 5
MAX_GENE_SET_SIZE = 300

SAVE_DIR = 'figures'

# Models: (registry_name, squeeze_channel, is_graph)
MODELS = {
    'MLP':         ('CellTypeMLP',    True,  False),
    'CNN':         ('CellTypeCNN',    False, False),
    'Transformer': ('CellTypeTOSICA', True,  False),
    'GNN':         ('CellTypeGNN',    False, True),
}

console = Console()


# ---------------------------------------------------------------------------
# Data helpers (generalised from the old single-pair version)
# ---------------------------------------------------------------------------

def _load_eval_data(
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all splits combined for any dataset directory."""
    parts_x: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    for split in ('train', 'val', 'test'):
        xp: str = os.path.join(data_dir, f'X_{split}.npy')
        yp: str = os.path.join(data_dir, f'y_{split}.npy')
        if not os.path.exists(xp):
            continue
        parts_x.append(np.asarray(np.load(xp, mmap_mode='r')))
        parts_y.append(np.load(yp))
    X_eval: np.ndarray = np.concatenate(parts_x, axis=0)
    y_eval: np.ndarray = np.concatenate(parts_y, axis=0).astype(np.int64)
    gene_names: np.ndarray = np.load(os.path.join(data_dir, 'gene_names.npy'))
    class_names: np.ndarray = np.load(os.path.join(data_dir, 'class_names.npy'),
                                      allow_pickle=True)
    return X_eval, y_eval, gene_names, class_names


def _align_to_training(
    X_eval: np.ndarray,
    gene_names_eval: np.ndarray,
    ckpt_path: str,
    train_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align eval-data genes to match training gene order + HVG."""
    gene_names_train: np.ndarray = np.load(os.path.join(train_dir, 'gene_names.npy'))
    X_aligned: np.ndarray
    n_matched: int
    X_aligned, n_matched = align_genes(X_eval, gene_names_eval, gene_names_train)

    hvg_path: str = os.path.join(os.path.dirname(ckpt_path), 'hvg_indices.npy')
    if os.path.exists(hvg_path):
        hvg_idx: np.ndarray = np.load(hvg_path)
        X_aligned = X_aligned[:, hvg_idx]
        gene_names_train = gene_names_train[hvg_idx]

    return X_aligned, gene_names_train


def _apply_saved_normalization(X_al: np.ndarray, ckpt_path: str) -> np.ndarray:
    ckpt_dir: str = os.path.dirname(ckpt_path)
    normalize: str | None = None
    norm_path: str = os.path.join(ckpt_dir, 'normalize.txt')
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            normalize = f.read().strip()
    scaler: Any = None
    scaler_path: str = os.path.join(ckpt_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    if normalize:
        X_al = T._apply_normalization_test(
            np.asarray(X_al, dtype=np.float32), normalize, scaler)
    return X_al


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _eval_standard_model(
    model_name: str,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    gene_names_eval: np.ndarray,
    class_names_eval: np.ndarray,
    squeeze_channel: bool,
    train_dir: str,
    extra_model_kwargs: dict[str, Any] | None = None,
) -> EvalMetrics | None:
    ckpt_path: str | None = T.find_best_ckpt(model_name)
    if ckpt_path is None:
        return None

    X_al: np.ndarray
    gnames: np.ndarray
    X_al, gnames = _align_to_training(X_eval, gene_names_eval, ckpt_path, train_dir)
    X_al = _apply_saved_normalization(X_al, ckpt_path)
    n_features: int = X_al.shape[1]
    n_classes_train: int = len(np.load(os.path.join(train_dir, 'class_names.npy'),
                                       allow_pickle=True))

    saved_kw: dict[str, Any] = T._load_model_kwargs(ckpt_path, model_name=model_name)
    if extra_model_kwargs:
        saved_kw.update(extra_model_kwargs)
    model: torch.nn.Module = T.build_model(model_name, n_features, n_classes_train, **saved_kw)
    model.load_state_dict(torch.load(ckpt_path, map_location=T.DEVICE,
                                     weights_only=True))

    X_t: torch.Tensor = torch.from_numpy(X_al).unsqueeze(1)
    y_t: torch.Tensor = torch.from_numpy(y_eval)
    loader: DataLoader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE,
                                    shuffle=False)
    y_pred: np.ndarray
    y_true: np.ndarray
    y_pred, y_true = T._collect_predictions(model, loader, squeeze_channel)
    return T._compute_metrics(y_true, y_pred, list(class_names_eval))


def _eval_gnn_model(
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    gene_names_eval: np.ndarray,
    class_names_eval: np.ndarray,
    train_dir: str,
) -> EvalMetrics | None:
    ckpt_path: str | None = T.find_best_ckpt('CellTypeGNN')
    if ckpt_path is None:
        return None

    X_al: np.ndarray
    X_al, _ = _align_to_training(X_eval, gene_names_eval, ckpt_path, train_dir)
    n_features: int = X_al.shape[1]
    n_classes_train: int = len(np.load(os.path.join(train_dir, 'class_names.npy'),
                                       allow_pickle=True))

    data = build_eval_graph(X_al, y_eval, k_neighbors=K_NEIGHBORS).to(T.DEVICE)
    saved_kw: dict[str, Any] = T._load_model_kwargs(ckpt_path, model_name='CellTypeGNN')
    model: torch.nn.Module = T.build_model('CellTypeGNN', n_features, n_classes_train, **saved_kw)
    model.load_state_dict(torch.load(ckpt_path, map_location=T.DEVICE,
                                     weights_only=True))
    model.eval()
    with torch.no_grad():
        logits: torch.Tensor = model(data.x, data.edge_index)
    y_pred: np.ndarray = logits[data.test_mask].argmax(1).cpu().numpy()
    y_true: np.ndarray = data.y[data.test_mask].cpu().numpy()
    return T._compute_metrics(y_true, y_pred, list(class_names_eval))


def _transformer_extra_kwargs(gene_names: np.ndarray) -> dict[str, Any]:
    mask: torch.Tensor
    n_pathways: int
    mask, n_pathways = build_pathway_mask(
        [str(g) for g in gene_names],
        gmt_path=GMT_PATH, min_overlap=MIN_PATHWAY_OVERLAP,
        max_pathways=MAX_PATHWAYS, max_gene_set_size=MAX_GENE_SET_SIZE)
    return dict(mask=mask, n_pathways=n_pathways)


# ---------------------------------------------------------------------------
# N x N evaluation
# ---------------------------------------------------------------------------

def _cross_eval_pair(
    train_dir: str,
    train_tag: str,
    eval_dir: str,
    eval_tag: str,
) -> dict[str, EvalMetrics | None]:
    """Evaluate all models trained on train_dir against eval_dir."""
    X_eval: np.ndarray
    y_eval: np.ndarray
    gene_names_eval: np.ndarray
    class_names_eval: np.ndarray
    X_eval, y_eval, gene_names_eval, class_names_eval = _load_eval_data(eval_dir)
    results: dict[str, EvalMetrics | None] = {}

    for label, (cls_name, squeeze, is_graph) in MODELS.items():
        m: EvalMetrics | None
        try:
            if is_graph:
                m = _eval_gnn_model(X_eval, y_eval, gene_names_eval,
                                    class_names_eval, train_dir)
            elif cls_name == 'CellTypeTOSICA':
                ckpt: str | None = T.find_best_ckpt(cls_name)
                if ckpt is not None:
                    X_al: np.ndarray
                    gnames: np.ndarray
                    X_al, gnames = _align_to_training(
                        X_eval, gene_names_eval, ckpt, train_dir)
                    tosica_kw: dict[str, Any] | None = _transformer_extra_kwargs(gnames)
                else:
                    tosica_kw = None
                m = _eval_standard_model(
                    cls_name, X_eval, y_eval, gene_names_eval,
                    class_names_eval, squeeze, train_dir,
                    extra_model_kwargs=tosica_kw)
            else:
                m = _eval_standard_model(
                    cls_name, X_eval, y_eval, gene_names_eval,
                    class_names_eval, squeeze, train_dir)
        except Exception as e:
            console.print(f'  [red]{label} failed: {e}[/red]')
            m = None
        results[label] = m

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_cross_eval_heatmaps(
    all_results: dict[tuple[str, str], dict[str, EvalMetrics | None]],
    dataset_tags: list[str],
    metric: str = 'accuracy',
) -> None:
    """N x N heatmap per model for one metric."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_labels: list[str] = list(MODELS.keys())
    n: int = len(dataset_tags)
    ncols: int = min(len(model_labels), 4)
    nrows: int = (len(model_labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flat

    for ax, model_label in zip(axes, model_labels):
        matrix: np.ndarray = np.full((n, n), np.nan)
        for i, t_tag in enumerate(dataset_tags):
            for j, e_tag in enumerate(dataset_tags):
                if t_tag == e_tag:
                    continue
                m: EvalMetrics | None = all_results.get((t_tag, e_tag), {}).get(model_label)
                if m is not None:
                    matrix[i, j] = m.get(metric, np.nan)
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=dataset_tags, yticklabels=dataset_tags,
                    ax=ax, vmin=0, vmax=1)
        ax.set_xlabel('Eval Dataset')
        ax.set_ylabel('Train Dataset')
        ax.set_title(f'{model_label}: {metric}')

    # Hide unused axes
    for idx in range(len(model_labels), nrows * ncols):
        fig.delaxes(list(np.atleast_1d(fig.axes))[idx])

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'cross_eval_{metric}.png'), dpi=150)
    plt.close()
    console.print(f'[green]Saved[/green] {SAVE_DIR}/cross_eval_{metric}.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.print(Panel(
        '[bold]N x N CROSS-DATASET EVALUATION[/bold]',
        border_style='cyan', expand=False,
    ))

    # Only evaluate datasets that have processed splits
    available: list[tuple[str, str]] = []
    for name, info in ALL_DATASETS.items():
        if os.path.exists(os.path.join(info['dir'], 'X_train.npy')):
            available.append((name, info['dir']))
        else:
            console.print(f'[yellow]Skipping {name}: no splits found[/yellow]')

    dataset_tags: list[str] = [t for t, _ in available]
    all_results: dict[tuple[str, str], dict[str, EvalMetrics | None]] = {}

    for train_tag, train_dir in available:
        for eval_tag, eval_dir in available:
            if train_tag == eval_tag:
                continue
            console.print(Panel(
                f'[bold]{train_tag} -> {eval_tag}[/bold]',
                border_style='dim', expand=False,
            ))
            pair_results: dict[str, EvalMetrics | None] = _cross_eval_pair(
                train_dir, train_tag, eval_dir, eval_tag)
            all_results[(train_tag, eval_tag)] = pair_results

    # Heatmaps
    for metric in ('accuracy', 'f1_macro', 'f1_weighted'):
        _plot_cross_eval_heatmaps(all_results, dataset_tags, metric=metric)

    # Summary table
    for model_label in MODELS:
        table: Table = Table(title=f'{model_label}: Cross-Dataset Accuracy',
                             show_lines=True)
        table.add_column('Train \\ Eval', style='bold')
        for tag in dataset_tags:
            table.add_column(tag, justify='right')
        for t_tag in dataset_tags:
            row: list[str] = [t_tag]
            for e_tag in dataset_tags:
                if t_tag == e_tag:
                    row.append('-')
                else:
                    m: EvalMetrics | None = all_results.get((t_tag, e_tag), {}).get(model_label)
                    row.append(f'{m["accuracy"]:.4f}' if m else 'N/A')
            table.add_row(*row)
        console.print(table)


if __name__ == '__main__':
    main()
