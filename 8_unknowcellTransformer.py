"""Train TOSICA on a subset of cell types, detect held-out types as Unknown.

Replicates the TOSICA paper's unknown cell type discovery experiment:
  1. Randomly sample 5000 genes
  2. Hold out one cell type from training
  3. Train TOSICA with pathway mask
  4. Predict on full test set — held-out cells should be flagged as 'Unknown'
  5. Run the paper's attention embedding pipeline: normalize -> PCA -> kNN -> UMAP
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from rich.console import Console
from rich.panel import Panel
from torch.utils.data import DataLoader

from allen_brain.cell_data.cell_dataset import make_dataset, GeneExpressionDataset
from allen_brain.models import train as T
from allen_brain.models.CellTypeAttention import build_pathway_mask
from allen_brain.models.CellTypeAttentionUMAP import (
    select_random_genes, collect_attention, attention_umap,
)

console = Console()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = 'data/10x'
SAVE_DIR = 'figures'
GMT_PATH = 'data/reactome.gmt'

N_RANDOM_GENES = 5000
SEED = 42
BATCH_SIZE = 4096
EPOCHS = 20
UNKNOWN_THRESHOLD = 0.95

MAX_PATHWAYS = 300
MIN_PATHWAY_OVERLAP = 5
MAX_GENE_SET_SIZE = 300

CFG = {
    'model': 'CellTypeTOSICA',
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'optimizer': 'adam',
    'lr': 3.2e-3,
    'weight_decay': 7e-4,
    'loss': 'focal',
    'label_smoothing': 0.06,
    'focal_gamma': 0.47,
    'normalize': None,
    'n_hvg': 0,
}

MODEL_KW = dict(embed_dim=64, n_heads=4, n_layers=1, dropout=0.41)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subset_genes(ds, gene_idx):
    """Subset dataset columns to selected gene indices."""
    ds.X = np.asarray(ds.X[:, gene_idx])
    ds.gene_names = ds.gene_names[gene_idx]
    return ds


def _hold_out_class(ds, class_idx):
    """Remove all cells of class_idx from the dataset. Returns new dataset arrays."""
    mask = ds.y != class_idx
    X_kept = np.asarray(ds.X)[mask]
    y_kept = np.asarray(ds.y)[mask]
    return X_kept, y_kept, mask


def _remap_labels(y, held_out_idx, n_original_classes):
    """Remap labels to fill the gap left by the held-out class."""
    mapping = {}
    new_label = 0
    for old in range(n_original_classes):
        if old == held_out_idx:
            continue
        mapping[old] = new_label
        new_label += 1
    return np.array([mapping[int(yi)] for yi in y], dtype=np.int64), mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    console.print(Panel('[bold]Unknown Cell Type Discovery — TOSICA[/bold]',
                        border_style='cyan', expand=False))

    # ------------------------------------------------------------------
    # 1. Load data and select random genes
    # ------------------------------------------------------------------
    ds_train = make_dataset(DATA_DIR, split='train')
    ds_val   = make_dataset(DATA_DIR, split='val')
    ds_test  = make_dataset(DATA_DIR, split='test')

    n_genes_total = len(ds_train.gene_names)
    console.print(f'Total genes: {n_genes_total}')

    gene_idx = select_random_genes(n_genes_total, N_RANDOM_GENES, seed=SEED)
    console.print(f'Randomly selected {len(gene_idx)} genes')

    for ds in (ds_train, ds_val, ds_test):
        _subset_genes(ds, gene_idx)

    gene_names = [str(g) for g in ds_train.gene_names]
    all_class_names = list(ds_train.class_names)
    n_original_classes = ds_train.n_classes

    # ------------------------------------------------------------------
    # 2. Pick held-out class (largest class for clear signal)
    # ------------------------------------------------------------------
    counts = np.bincount(np.asarray(ds_train.y), minlength=n_original_classes)
    held_out_idx = int(np.argmax(counts))
    held_out_name = all_class_names[held_out_idx]
    console.print(f'Holding out class [bold]{held_out_name}[/bold] '
                  f'({counts[held_out_idx]} train cells)')

    # ------------------------------------------------------------------
    # 3. Build reduced training / validation sets
    # ------------------------------------------------------------------
    X_train, y_train, _ = _hold_out_class(ds_train, held_out_idx)
    X_val,   y_val,   _ = _hold_out_class(ds_val,   held_out_idx)

    y_train, label_map = _remap_labels(y_train, held_out_idx, n_original_classes)
    y_val, _           = _remap_labels(y_val,   held_out_idx, n_original_classes)

    reduced_class_names = [all_class_names[i] for i in range(n_original_classes)
                           if i != held_out_idx]
    n_reduced = len(reduced_class_names)
    console.print(f'Training classes: {n_reduced} (held out: {held_out_name})')

    # Wrap into simple datasets
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long())
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long())

    pin = T.DEVICE.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=pin)
    loaders = (train_loader, val_loader)

    # ------------------------------------------------------------------
    # 4. Build pathway mask and model
    # ------------------------------------------------------------------
    mask, n_pathways = build_pathway_mask(
        gene_names, gmt_path=GMT_PATH, min_overlap=MIN_PATHWAY_OVERLAP,
        max_pathways=MAX_PATHWAYS, max_gene_set_size=MAX_GENE_SET_SIZE)

    model_kw = dict(MODEL_KW, mask=mask, n_pathways=n_pathways)
    model = T.build_model('CellTypeTOSICA', len(gene_names), n_reduced, **model_kw)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    w = torch.tensor(
        1.0 / (np.bincount(y_train, minlength=n_reduced).astype(np.float32) + 1e-6),
        dtype=torch.float32).to(T.DEVICE)
    w = w / w.sum() * n_reduced

    criterion = T.build_criterion(CFG['loss'], weight=w,
                                  label_smoothing=CFG['label_smoothing'],
                                  gamma=CFG['focal_gamma'])
    optimizer, scheduler = T.build_optimizer(
        model, CFG['lr'], CFG['weight_decay'], EPOCHS, opt_cls=CFG['optimizer'])
    writer, ckpt = T.make_writer_and_ckpt(CFG, len(gene_names))

    console.print(f'Training {EPOCHS} epochs on {T.DEVICE}...')
    T.print_header()
    T.train(model, loaders, criterion, optimizer, scheduler,
            EPOCHS, writer, ckpt, squeeze_channel=False)
    writer.close()

    # Reload best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))

    # ------------------------------------------------------------------
    # 6. Predict on full test set (including held-out class)
    # ------------------------------------------------------------------
    X_test = np.asarray(ds_test.X).astype(np.float32)
    y_test = np.asarray(ds_test.y)
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=pin)

    attn_matrix, test_labels, _, max_probs = collect_attention(
        model, test_loader, T.DEVICE, squeeze_channel=False)

    # Flag unknowns
    unknown_mask = max_probs < UNKNOWN_THRESHOLD

    # Stats for held-out class
    is_held = test_labels == held_out_idx
    n_held = is_held.sum()
    n_held_unknown = (is_held & unknown_mask).sum()
    pct = n_held_unknown / max(n_held, 1) * 100

    console.print(Panel(
        f'Held-out class: [bold]{held_out_name}[/bold]\n'
        f'  Test cells: {n_held}\n'
        f'  Flagged as Unknown (threshold={UNKNOWN_THRESHOLD}): '
        f'[bold green]{n_held_unknown}[/bold green] ({pct:.1f}%)\n'
        f'  Other cells flagged Unknown: {(~is_held & unknown_mask).sum()}',
        border_style='green', expand=False))

    # ------------------------------------------------------------------
    # 7. Attention embedding UMAP (paper pipeline)
    # ------------------------------------------------------------------
    console.print('Running attention embedding UMAP (normalize -> PCA -> kNN -> UMAP)...')

    # Build display labels: original names, with Unknown overlay
    display_labels = np.array(test_labels, copy=True)
    display_names = all_class_names + ['Unknown']
    unknown_label = len(all_class_names)
    display_labels[unknown_mask] = unknown_label

    adata = attention_umap(attn_matrix, display_labels, display_names)
    umap_coords = adata.obsm['X_umap']

    # ------------------------------------------------------------------
    # 8. Plot: UMAP colored by cell type, Unknown highlighted
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: all original labels
    ax = axes[0]
    for c in range(n_original_classes):
        m = test_labels == c
        ax.scatter(umap_coords[m, 0], umap_coords[m, 1],
                   s=4, alpha=0.5, label=all_class_names[c])
    ax.set_title('Attention UMAP — Original Labels')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(markerscale=4, fontsize=6, loc='best', ncol=2)

    # Right: known vs unknown prediction
    ax = axes[1]
    known = ~unknown_mask
    ax.scatter(umap_coords[known & ~is_held, 0],
               umap_coords[known & ~is_held, 1],
               s=4, alpha=0.3, c='tab:blue', label='Known (correct)')
    ax.scatter(umap_coords[known & is_held, 0],
               umap_coords[known & is_held, 1],
               s=6, alpha=0.6, c='tab:orange', label=f'{held_out_name} (missed)')
    ax.scatter(umap_coords[unknown_mask & is_held, 0],
               umap_coords[unknown_mask & is_held, 1],
               s=8, alpha=0.8, c='tab:red', label=f'{held_out_name} → Unknown')
    ax.scatter(umap_coords[unknown_mask & ~is_held, 0],
               umap_coords[unknown_mask & ~is_held, 1],
               s=6, alpha=0.6, c='tab:purple', label='Other → Unknown')
    ax.set_title(f'Unknown Detection (threshold={UNKNOWN_THRESHOLD})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(markerscale=4, fontsize=8)

    plt.suptitle(f'Held-out: {held_out_name} — '
                 f'{pct:.1f}% detected as Unknown', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'unknown_cell_attention_umap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'[green]Saved[/green] {save_path}')

    # ------------------------------------------------------------------
    # 9. Confidence histogram
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_probs[is_held], bins=50, alpha=0.7,
            label=f'{held_out_name} (held out)', color='tab:red')
    ax.hist(max_probs[~is_held], bins=50, alpha=0.5,
            label='Known classes', color='tab:blue')
    ax.axvline(UNKNOWN_THRESHOLD, color='black', ls='--', label=f'Threshold={UNKNOWN_THRESHOLD}')
    ax.set_xlabel('Max Softmax Probability')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution: Held-out vs Known')
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'unknown_cell_confidence_hist.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    console.print(f'[green]Saved[/green] {save_path}')

    console.print(Panel('[bold green]Done[/bold green]', expand=False))


if __name__ == '__main__':
    main()
