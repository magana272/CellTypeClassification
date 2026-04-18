"""
10_multiple_dataset.py
Train all models on all benchmark datasets, evaluate with full metrics,
generate confusion matrices and comparative heatmaps.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch_geometric.data import Data

from allen_brain.models import train as T
from allen_brain.models.train import Trainer, DEVICE
from allen_brain.models.gnn_train import GraphTrainer
from allen_brain.models.config import EvalMetrics, ExperimentConfig
from allen_brain.cell_data.cell_load import ALL_DATASETS
from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.cell_data.cell_vis import ModelComparisonVisualizer
from allen_brain.models.CellTypeAttention import PathwayMaskBuilder
from allen_brain.models.CellTypeGNN import GraphBuilder

console: Console = Console()

SEED: int = 42
GMT_PATH: str = 'data/reactome.gmt'
MAX_PATHWAYS: int = 300
MIN_PATHWAY_OVERLAP: int = 5
MAX_GENE_SET_SIZE: int = 300
SAVE_DIR: str = 'figures/multi_dataset'

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



def _build_pathway_kwargs(data_dir: str) -> dict[str, Any]:
    ds = make_dataset(data_dir, split='train')
    mask: torch.Tensor
    n_pathways: int
    mask, n_pathways = PathwayMaskBuilder(
        gmt_path=GMT_PATH, min_overlap=MIN_PATHWAY_OVERLAP,
        max_pathways=MAX_PATHWAYS, max_gene_set_size=MAX_GENE_SET_SIZE,
    ).build_mask([str(g) for g in ds.gene_names])
    return dict(mask=mask, n_pathways=n_pathways)


def _make_cfg(model_cfg: dict[str, Any], data_tag: str) -> dict[str, Any]:
    """Build a training config dict from a model config entry."""
    return {
        'model': model_cfg['model'],
        'seed': SEED,
        'batch_size': model_cfg['batch_size'],
        'accumulation_steps': model_cfg.get('accumulation_steps', 1),
        'n_hvg': model_cfg.get('n_hvg', 0),
        'device': str(DEVICE),
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
    """Train one model on one dataset and evaluate. Returns metrics."""
    cfg: dict[str, Any] = _make_cfg(model_cfg, data_tag)
    extra_kw: dict[str, Any] | None = None

    exp_cfg: ExperimentConfig = ExperimentConfig(**cfg)

    if model_cfg['is_graph']:
        data: Data = GraphBuilder(k_neighbors=10,
                                  normalize=cfg['normalize']).build_graph_data(data_dir).to(DEVICE)
        n_classes: int = int(data.y.max().item()) + 1
        weights: torch.Tensor = GraphBuilder.masked_class_weights(data.y, data.train_mask, n_classes)
        n_features: int = data.x.shape[1]
        class_names: list[str] = list(np.load(f'{data_dir}/class_names.npy', allow_pickle=True))
        del data
        best_acc: float
        ckpt: str
        bp: dict[str, Any]
        gt = GraphTrainer(exp_cfg)
        best_acc, ckpt, bp = gt.train_single(
            data_dir, n_features, n_classes, weights)
        best_k: int = int(bp.get('k_neighbors', 10))
        best_norm: str | None = bp.get('normalize', cfg['normalize'])
        if best_norm == 'none':
            best_norm = None
        eval_data: Data = GraphBuilder(k_neighbors=best_k,
                                       normalize=best_norm).build_graph_data(data_dir).to(DEVICE)
        metrics: EvalMetrics = gt.evaluate(eval_data, ckpt, n_features, n_classes,
                                           class_names=class_names)
    else:
        if model_cfg['model'] == 'CellTypeTOSICA':
            try:
                extra_kw = _build_pathway_kwargs(data_dir)
            except Exception as e:
                console.print(f'  [yellow]Pathway mask failed: {e}[/yellow]')
                return None
        trainer = Trainer(exp_cfg)
        best_acc, ckpt, bp = trainer.train_single(
            data_dir, squeeze_channel=model_cfg['squeeze_channel'],
            extra_model_kwargs=extra_kw)
        metrics = trainer.evaluate(data_dir, ckpt,
                                   squeeze_channel=model_cfg['squeeze_channel'])

    return metrics


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
                    f'{m.accuracy:.4f}',
                    f'{m.f1_macro:.4f}',
                    f'{m.f1_weighted:.4f}',
                    f'{m.precision_weighted:.4f}',
                    f'{m.recall_weighted:.4f}',
                )
        console.print(table)



def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_names: list[str] = list(MODELS.keys())

    all_results: dict[str, dict[str, EvalMetrics | None]] = {}

    for ds_name, ds_info in ALL_DATASETS.items():
        data_dir: str = ds_info['dir']

        if not (os.path.exists(os.path.join(data_dir, 'X_train.npy'))
                or os.path.exists(os.path.join(data_dir, 'X_train.npz'))):
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

    # Comparative visualisations via shared class
    ModelComparisonVisualizer.plot_metric_heatmap(
        all_results, 'accuracy', model_names, SAVE_DIR,
        title='Model Accuracy Across Datasets', cmap='YlGn',
        filename='accuracy_heatmap.png')
    ModelComparisonVisualizer.plot_metric_heatmap(
        all_results, 'f1_macro', model_names, SAVE_DIR,
        title='Model F1-Macro Across Datasets', cmap='YlOrRd',
        filename='f1_macro_heatmap.png')

    for ds_tag, model_results in all_results.items():
        ModelComparisonVisualizer.plot_eval_confusion_matrices(
            model_results, SAVE_DIR, suptitle=ds_tag)

    _print_summary_tables(all_results)


if __name__ == '__main__':
    main()