"""Cross-dataset evaluation: models trained on 10x, evaluated on SmartSeq."""

import os

import numpy as np
import torch
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


def _load_eval_data():
    """Load and return SmartSeq evaluation data (all splits combined)."""
    # Ensure SmartSeq data is processed
    if not os.path.exists(os.path.join(EVAL_DIR, 'X_train.npy')):
        print('Processing SmartSeq dataset...')
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
        print(f'Applied HVG selection: {len(hvg_idx)} genes')

    return X_aligned, gene_names_train


def _eval_standard_model(model_name, X_aligned, y_eval, gene_names,
                         class_names, squeeze_channel, extra_model_kwargs=None):
    """Evaluate a standard (non-graph) model on aligned data."""
    ckpt_path = _find_best_ckpt(model_name)
    if ckpt_path is None:
        print(f'  No checkpoint found for {model_name}, skipping.')
        return None

    X_al, gnames = _align_to_training(X_aligned, gene_names, ckpt_path)
    n_features = X_al.shape[1]
    n_classes_train = len(np.load(os.path.join(TRAIN_DIR, 'class_names.npy'),
                                  allow_pickle=True))

    saved_kw = T._load_model_kwargs(ckpt_path, model_name=model_name)
    extra_kw = extra_model_kwargs or {}
    saved_kw.update(extra_kw)
    model = T.build_model(model_name, n_features, n_classes_train, **saved_kw)

    model.load_state_dict(torch.load(ckpt_path, map_location=T.DEVICE,
                                     weights_only=True))
    print(f'Loaded checkpoint: {ckpt_path}')

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
        print('  No checkpoint found for CellTypeGNN, skipping.')
        return None

    X_al, gnames = _align_to_training(X_aligned, gene_names, ckpt_path)
    n_features = X_al.shape[1]
    n_classes_train = len(np.load(os.path.join(TRAIN_DIR, 'class_names.npy'),
                                  allow_pickle=True))

    # Build new graph on SmartSeq data
    print(f'Building evaluation graph on SmartSeq data...')
    data = build_eval_graph(X_al, y_eval, k_neighbors=K_NEIGHBORS).to(T.DEVICE)

    saved_kw = T._load_model_kwargs(ckpt_path, model_name='CellTypeGNN')
    model = T.build_model('CellTypeGNN', n_features, n_classes_train, **saved_kw)
    model.load_state_dict(torch.load(ckpt_path, map_location=T.DEVICE,
                                     weights_only=True))
    print(f'Loaded checkpoint: {ckpt_path}')

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
    print('=' * 60)
    print('CROSS-DATASET EVALUATION: 10x -> SmartSeq')
    print('=' * 60)

    X_eval, y_eval, gene_names_eval, class_names_eval = _load_eval_data()
    print(f'\nSmartSeq data: {X_eval.shape[0]:,} cells, '
          f'{X_eval.shape[1]:,} genes, {len(class_names_eval)} classes')
    print(f'Classes: {list(class_names_eval)}\n')

    results = {}

    # MLP
    print('\n' + '-' * 40)
    print('Evaluating CellTypeMLP on SmartSeq')
    print('-' * 40)
    results['MLP'] = _eval_standard_model(
        'CellTypeMLP', X_eval, y_eval, gene_names_eval,
        class_names_eval, squeeze_channel=True)

    # CNN
    print('\n' + '-' * 40)
    print('Evaluating CellTypeCNN on SmartSeq')
    print('-' * 40)
    results['CNN'] = _eval_standard_model(
        'CellTypeCNN', X_eval, y_eval, gene_names_eval,
        class_names_eval, squeeze_channel=False)

    # Transformer
    print('\n' + '-' * 40)
    print('Evaluating CellTypeTOSICA on SmartSeq')
    print('-' * 40)
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
    print('\n' + '-' * 40)
    print('Evaluating CellTypeGNN on SmartSeq')
    print('-' * 40)
    results['GNN'] = _eval_gnn_model(
        X_eval, y_eval, gene_names_eval, class_names_eval)

    # Summary table
    print('\n' + '=' * 60)
    print('CROSS-DATASET EVALUATION SUMMARY')
    print('=' * 60)
    header = f'{"Model":<15} {"Acc":>6} {"F1-M":>6} {"F1-W":>6} {"Prec-M":>7} {"Rec-M":>6}'
    print(header)
    print('-' * len(header))
    for name, m in results.items():
        if m is None:
            print(f'{name:<15} {"N/A":>6}')
        else:
            print(f'{name:<15} {m["accuracy"]:>6.4f} {m["f1_macro"]:>6.4f} '
                  f'{m["f1_weighted"]:>6.4f} {m["precision_macro"]:>7.4f} '
                  f'{m["recall_macro"]:>6.4f}')


if __name__ == '__main__':
    main()
