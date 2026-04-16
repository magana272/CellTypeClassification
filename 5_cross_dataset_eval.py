"""Cross-dataset evaluation: models trained on 10x, evaluated on SmartSeq."""

import os
import pickle

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader, TensorDataset

from allen_brain.models import train as T
from allen_brain.cell_data.cell_load import load_smartseq
from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.cell_data.cell_preprocess import align_genes
from allen_brain.models.CellTypeAttention import build_pathway_mask
from allen_brain.models.CellTypeGNN import build_eval_graph

TRAIN_DIR = 'data/10x'
EVAL_DIR = 'data/smartseq'
K_NEIGHBORS = 15
BATCH_SIZE = 1024

# GMT config for Transformer pathway mask
GMT_PATH = 'data/reactome.gmt'
MAX_PATHWAYS = 300
MIN_PATHWAY_OVERLAP = 5
MAX_GENE_SET_SIZE = 300


_find_best_ckpt = T.find_best_ckpt
console = Console()


def _load_eval_data():
    """Load and return SmartSeq evaluation data (all splits combined)."""
    # Ensure SmartSeq data is processed
    if not os.path.exists(os.path.join(EVAL_DIR, 'X_train.npy')):
        console.print('Processing SmartSeq dataset...')
        load_smartseq()

    # Combine all SmartSeq splits for evaluation
    parts_x, parts_y = [], []
    for split in ('train', 'val', 'test'):
        X = np.load(os.path.join(EVAL_DIR, f'X_{split}.npy'), mmap_mode='r')
        y = np.load(os.path.join(EVAL_DIR, f'y_{split}.npy'))
        parts_x.append(np.asarray(X))
        parts_y.append(y)
    X_eval = np.concatenate(parts_x, axis=0)
    y_eval = np.concatenate(parts_y, axis=0).astype(np.int64)
    gene_names_eval = np.load(os.path.join(EVAL_DIR, 'gene_names.npy'))
    class_names_eval = np.load(os.path.join(EVAL_DIR, 'class_names.npy'),
                               allow_pickle=True)
    return X_eval, y_eval, gene_names_eval, class_names_eval


def _align_to_training(X_eval, gene_names_eval, ckpt_path):
    """Align eval data genes to match training gene order, including HVG."""
    gene_names_train = np.load(os.path.join(TRAIN_DIR, 'gene_names.npy'))
    X_aligned, n_matched = align_genes(X_eval, gene_names_eval, gene_names_train)

    # Apply HVG indices if used during training
    hvg_path = os.path.join(os.path.dirname(ckpt_path), 'hvg_indices.npy')
    if os.path.exists(hvg_path):
        hvg_idx = np.load(hvg_path)
        X_aligned = X_aligned[:, hvg_idx]
        gene_names_train = gene_names_train[hvg_idx]
        console.print(f'Applied HVG selection: {len(hvg_idx)} genes')

    return X_aligned, gene_names_train


def _apply_saved_normalization(X_al, ckpt_path):
    """Load and apply saved normalization from checkpoint directory to eval data."""
    ckpt_dir = os.path.dirname(ckpt_path)
    normalize = None
    norm_path = os.path.join(ckpt_dir, 'normalize.txt')
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            normalize = f.read().strip()
    scaler = None
    scaler_path = os.path.join(ckpt_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    if normalize:
        console.print(f'Applying saved normalization: {normalize}')
        X_al = T._apply_normalization_test(np.asarray(X_al, dtype=np.float32), normalize, scaler)
    return X_al


def _eval_standard_model(model_name, X_aligned, y_eval, gene_names,
                         class_names, squeeze_channel, extra_model_kwargs=None):
    """Evaluate a standard (non-graph) model on aligned data."""
    ckpt_path = _find_best_ckpt(model_name)
    if ckpt_path is None:
        console.print(f'  [yellow]No checkpoint found for {model_name}[/yellow], skipping.')
        return None

    X_al, gnames = _align_to_training(X_aligned, gene_names, ckpt_path)
    X_al = _apply_saved_normalization(X_al, ckpt_path)
    n_features = X_al.shape[1]
    n_classes_train = len(np.load(os.path.join(TRAIN_DIR, 'class_names.npy'),
                                  allow_pickle=True))

    saved_kw = T._load_model_kwargs(ckpt_path, model_name=model_name)
    extra_kw = extra_model_kwargs or {}
    saved_kw.update(extra_kw)
    model = T.build_model(model_name, n_features, n_classes_train, **saved_kw)

    model.load_state_dict(torch.load(ckpt_path, map_location=T.DEVICE,
                                     weights_only=True))
    console.print(f'Loaded checkpoint: {ckpt_path}')

    # Build DataLoader from aligned numpy arrays
    X_t = torch.from_numpy(X_al).unsqueeze(1)  # add channel dim for dataset compat
    y_t = torch.from_numpy(y_eval)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE,
                        shuffle=False)

    y_pred, y_true = T._collect_predictions(model, loader, squeeze_channel)

    save_dir = os.path.join(os.path.dirname(ckpt_path), 'cross_dataset')
    return T._compute_metrics(y_true, y_pred, list(class_names), save_dir)


def _eval_gnn_model(X_aligned, y_eval, gene_names, class_names):
    """Evaluate GNN model on aligned SmartSeq data with new graph."""
    ckpt_path = _find_best_ckpt('CellTypeGNN')
    if ckpt_path is None:
        console.print('  [yellow]No checkpoint found for CellTypeGNN[/yellow], skipping.')
        return None

    X_al, gnames = _align_to_training(X_aligned, gene_names, ckpt_path)
    n_features = X_al.shape[1]
    n_classes_train = len(np.load(os.path.join(TRAIN_DIR, 'class_names.npy'),
                                  allow_pickle=True))

    # Build new graph on SmartSeq data
    console.print('Building evaluation graph on SmartSeq data...')
    data = build_eval_graph(X_al, y_eval, k_neighbors=K_NEIGHBORS).to(T.DEVICE)

    saved_kw = T._load_model_kwargs(ckpt_path, model_name='CellTypeGNN')
    model = T.build_model('CellTypeGNN', n_features, n_classes_train, **saved_kw)
    model.load_state_dict(torch.load(ckpt_path, map_location=T.DEVICE,
                                     weights_only=True))
    console.print(f'Loaded checkpoint: {ckpt_path}')

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    y_pred = logits[data.test_mask].argmax(1).cpu().numpy()
    y_true = data.y[data.test_mask].cpu().numpy()

    save_dir = os.path.join(os.path.dirname(ckpt_path), 'cross_dataset')
    return T._compute_metrics(y_true, y_pred, list(class_names), save_dir)


def _transformer_extra_kwargs(gene_names):
    """Build extra_model_kwargs for TOSICA from gene names."""
    mask, n_pathways = build_pathway_mask(
        [str(g) for g in gene_names],
        gmt_path=GMT_PATH, min_overlap=MIN_PATHWAY_OVERLAP,
        max_pathways=MAX_PATHWAYS, max_gene_set_size=MAX_GENE_SET_SIZE)
    return dict(mask=mask, n_pathways=n_pathways)


def main():
    console.print(Panel(
        '[bold]CROSS-DATASET EVALUATION[/bold]  ·  10x → SmartSeq',
        border_style='cyan', expand=False,
    ))

    X_eval, y_eval, gene_names_eval, class_names_eval = _load_eval_data()
    console.print(f'\nSmartSeq data: {X_eval.shape[0]:,} cells, '
          f'{X_eval.shape[1]:,} genes, {len(class_names_eval)} classes')
    console.print(f'Classes: {list(class_names_eval)}\n')

    results = {}

    # MLP
    console.print(Panel('[bold]Evaluating CellTypeMLP on SmartSeq[/bold]',
                        border_style='dim', expand=False))
    results['MLP'] = _eval_standard_model(
        'CellTypeMLP', X_eval, y_eval, gene_names_eval,
        class_names_eval, squeeze_channel=True)

    # CNN
    console.print(Panel('[bold]Evaluating CellTypeCNN on SmartSeq[/bold]',
                        border_style='dim', expand=False))
    results['CNN'] = _eval_standard_model(
        'CellTypeCNN', X_eval, y_eval, gene_names_eval,
        class_names_eval, squeeze_channel=False)

    # Transformer
    console.print(Panel('[bold]Evaluating CellTypeTOSICA on SmartSeq[/bold]',
                        border_style='dim', expand=False))
    # Build pathway mask from the aligned gene names
    ckpt_t = _find_best_ckpt('CellTypeTOSICA')
    if ckpt_t is not None:
        X_al_t, gnames_t = _align_to_training(X_eval, gene_names_eval, ckpt_t)
        tosica_kw = _transformer_extra_kwargs(gnames_t)
    else:
        tosica_kw = None
    results['Transformer'] = _eval_standard_model(
        'CellTypeTOSICA', X_eval, y_eval, gene_names_eval,
        class_names_eval, squeeze_channel=True,
        extra_model_kwargs=tosica_kw)

    # GNN
    console.print(Panel('[bold]Evaluating CellTypeGNN on SmartSeq[/bold]',
                        border_style='dim', expand=False))
    results['GNN'] = _eval_gnn_model(
        X_eval, y_eval, gene_names_eval, class_names_eval)

    # Summary table
    table = Table(title='Cross-Dataset Evaluation Summary', show_lines=True)
    table.add_column('Model', style='bold')
    table.add_column('Acc', justify='right')
    table.add_column('F1-M', justify='right')
    table.add_column('F1-W', justify='right')
    table.add_column('Prec-M', justify='right')
    table.add_column('Rec-M', justify='right')
    for name, m in results.items():
        if m is None:
            table.add_row(name, 'N/A', '', '', '', '')
        else:
            table.add_row(name,
                          f'{m["accuracy"]:.4f}',
                          f'{m["f1_macro"]:.4f}',
                          f'{m["f1_weighted"]:.4f}',
                          f'{m["precision_macro"]:.4f}',
                          f'{m["recall_macro"]:.4f}')
    console.print(table)


if __name__ == '__main__':
    main()
