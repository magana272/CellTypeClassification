"""Interpretability analysis for all 4 model architectures."""

import os

import numpy as np
import torch
import torch.nn.functional as F_torch
from rich.console import Console
from rich.panel import Panel
from torch.utils.data import DataLoader

console = Console()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from allen_brain.models import train as T
from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.models.CellTypeAttention import (
    build_pathway_mask, _parse_gmt, _select_pathways,
)
from allen_brain.models.CellTypeGNN import build_graph_data

DATA_DIR = 'data/10x'
SAVE_DIR = 'figures'
BATCH_SIZE = 512

# GMT config (must match 4_Transformer.py)
GMT_PATH = 'data/reactome.gmt'
MAX_PATHWAYS = 300
MIN_PATHWAY_OVERLAP = 5
MAX_GENE_SET_SIZE = 300


# ---------------------------------------------------------------------------
# TOSICA — Pathway Attention
# ---------------------------------------------------------------------------

def interpret_tosica(save_dir):
    """Extract and visualize pathway attention scores from TOSICA."""
    console.print(Panel('[bold]TOSICA Pathway Attention[/bold]',
                        border_style='dim', expand=False))
    ckpt = T.find_best_ckpt('CellTypeTOSICA')
    if ckpt is None:
        console.print('[yellow]No TOSICA checkpoint found, skipping.[/yellow]')
        return

    ds_test = make_dataset(DATA_DIR, split='test')
    hvg_path = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
    if os.path.exists(hvg_path):
        hvg_idx = np.load(hvg_path)
        ds_test.X = np.asarray(ds_test.X[:, hvg_idx])
        ds_test.gene_names = ds_test.gene_names[hvg_idx]

    gene_names = [str(g) for g in ds_test.gene_names]
    mask, n_pathways = build_pathway_mask(
        gene_names, gmt_path=GMT_PATH, min_overlap=MIN_PATHWAY_OVERLAP,
        max_pathways=MAX_PATHWAYS, max_gene_set_size=MAX_GENE_SET_SIZE)

    # Recover pathway names
    gmt = _parse_gmt(GMT_PATH, max_gene_set_size=MAX_GENE_SET_SIZE)
    gene_set = set(gene_names)
    kept = _select_pathways(gmt, gene_set, MIN_PATHWAY_OVERLAP, MAX_PATHWAYS)
    pathway_names = [name.replace('REACTOME_', '').replace('_', ' ')[:40]
                     for name, _ in kept]

    saved_kw = T._load_model_kwargs(ckpt, model_name='CellTypeTOSICA')
    saved_kw.update(mask=mask, n_pathways=n_pathways)
    model = T.build_model('CellTypeTOSICA', len(gene_names),
                          ds_test.n_classes, **saved_kw)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
    model.eval()

    loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=T.DEVICE.type == 'cuda')
    class_names = list(ds_test.class_names)
    n_classes = ds_test.n_classes

    # Collect attention per class
    attn_by_class = {c: [] for c in range(n_classes)}
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(T.DEVICE)
            yb = yb.to(T.DEVICE)
            if xb.dim() == 3:
                xb = xb.squeeze(1)
            _, attn = model(xb, return_attention=True)
            attn_np = attn.cpu().numpy()
            yb_np = yb.cpu().numpy()
            for i, label in enumerate(yb_np):
                attn_by_class[label].append(attn_np[i])

    mean_attn = np.zeros((n_classes, len(pathway_names)))
    for c in range(n_classes):
        if attn_by_class[c]:
            mean_attn[c] = np.mean(attn_by_class[c], axis=0)

    # Top K pathways by overall attention
    top_k = min(30, len(pathway_names))
    overall_importance = mean_attn.mean(axis=0)
    top_idx = np.argsort(overall_importance)[::-1][:top_k]

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(mean_attn[:, top_idx], annot=False, cmap='YlOrRd',
                xticklabels=[pathway_names[i] for i in top_idx],
                yticklabels=class_names, ax=ax)
    ax.set_title('TOSICA: Mean Pathway Attention by Cell Type')
    ax.set_xlabel('Pathway')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tosica_pathway_attention_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print('[green]Saved[/green] tosica_pathway_attention_heatmap.png')

    # Top pathways per top 3 classes
    class_counts = [len(attn_by_class[c]) for c in range(n_classes)]
    top_classes = np.argsort(class_counts)[::-1][:3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, c in enumerate(top_classes):
        ax = axes[i]
        class_attn = mean_attn[c]
        top_pw = np.argsort(class_attn)[::-1][:10]
        ax.barh(range(10), class_attn[top_pw])
        ax.set_yticks(range(10))
        ax.set_yticklabels([pathway_names[j] for j in top_pw], fontsize=7)
        ax.set_title(f'{class_names[c]}')
        ax.set_xlabel('Mean Attention')
        ax.invert_yaxis()
    plt.suptitle('TOSICA: Top 10 Pathways per Cell Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tosica_top_pathways_per_class.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print('[green]Saved[/green] tosica_top_pathways_per_class.png')


# ---------------------------------------------------------------------------
# GNN — Embedding UMAP
# ---------------------------------------------------------------------------

def interpret_gnn(save_dir):
    """Visualize GNN node embeddings via UMAP."""
    console.print(Panel('[bold]GNN Embedding UMAP[/bold]',
                        border_style='dim', expand=False))
    ckpt = T.find_best_ckpt('CellTypeGNN')
    if ckpt is None:
        console.print('[yellow]No GNN checkpoint found, skipping.[/yellow]')
        return

    try:
        import umap
    except ImportError:
        console.print('[yellow]umap-learn not installed, skipping GNN interpretability.[/yellow]')
        return

    saved_kw = T._load_model_kwargs(ckpt, model_name='CellTypeGNN')
    k = saved_kw.get('k_neighbors', 15)
    data = build_graph_data(DATA_DIR, k_neighbors=k).to(T.DEVICE)
    n_features = data.x.shape[1]
    n_classes = int(data.y.max().item()) + 1
    model = T.build_model('CellTypeGNN', n_features, n_classes, **saved_kw)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
    model.eval()

    class_names = list(np.load(f'{DATA_DIR}/class_names.npy', allow_pickle=True))

    with torch.no_grad():
        embeddings = model.embed(data.x, data.edge_index).cpu().numpy()
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(1).cpu().numpy()

    labels = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    # Subsample for speed
    n_sample = min(10000, embeddings.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(embeddings.shape[0], n_sample, replace=False)
    emb_sub = embeddings[idx]
    labels_sub = labels[idx]
    preds_sub = preds[idx]

    console.print(f'Running UMAP on {n_sample} nodes...')
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_emb = reducer.fit_transform(emb_sub)

    # UMAP by class
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(n_classes):
        mask = labels_sub == c
        ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1], s=3, alpha=0.5,
                   label=class_names[c])
    ax.legend(markerscale=4, fontsize=7)
    ax.set_title('GNN Embeddings — UMAP by Cell Type')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gnn_embedding_umap.png'), dpi=150)
    plt.close(fig)
    console.print('[green]Saved[/green] gnn_embedding_umap.png')

    # UMAP by correctness
    correct = labels_sub == preds_sub
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(umap_emb[correct, 0], umap_emb[correct, 1], s=3, alpha=0.3,
               c='tab:blue', label='Correct')
    ax.scatter(umap_emb[~correct, 0], umap_emb[~correct, 1], s=5, alpha=0.6,
               c='tab:red', label='Incorrect')
    ax.legend(markerscale=4)
    ax.set_title('GNN Embeddings — Prediction Correctness')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gnn_embedding_correctness.png'), dpi=150)
    plt.close(fig)
    console.print('[green]Saved[/green] gnn_embedding_correctness.png')


# ---------------------------------------------------------------------------
# MLP — Gene Saliency
# ---------------------------------------------------------------------------

def _compute_saliency(model, loader, squeeze_channel, n_genes, n_classes):
    """Compute input-gradient saliency for a non-graph model."""
    model.eval()
    saliency_sum = np.zeros((n_classes, n_genes))
    class_counts = np.zeros(n_classes)

    for xb, yb in loader:
        xb = xb.to(T.DEVICE)
        yb = yb.to(T.DEVICE)
        if squeeze_channel and xb.dim() == 3:
            xb = xb.squeeze(1)
        xb.requires_grad_(True)
        logits = model(xb)
        # Backward on predicted class logit for each sample
        pred_logits = logits.gather(1, logits.argmax(1, keepdim=True)).sum()
        pred_logits.backward()
        grad = xb.grad.abs().detach().cpu().numpy()
        yb_np = yb.cpu().numpy()
        for i, label in enumerate(yb_np):
            saliency_sum[label] += grad[i]
            class_counts[label] += 1
        model.zero_grad()

    # Mean saliency per class
    for c in range(n_classes):
        if class_counts[c] > 0:
            saliency_sum[c] /= class_counts[c]
    return saliency_sum


def _plot_saliency(saliency, gene_names, class_names, model_prefix, save_dir):
    """Plot top-20 genes and per-class heatmap from saliency matrix."""
    overall = saliency.mean(axis=0)
    top_20 = np.argsort(overall)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(20), overall[top_20])
    ax.set_yticks(range(20))
    ax.set_yticklabels([gene_names[i] for i in top_20], fontsize=8)
    ax.set_title(f'{model_prefix}: Top 20 Genes by Mean Saliency')
    ax.set_xlabel('Mean |∂L/∂x|')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_prefix.lower()}_gene_saliency_top20.png'),
                dpi=150)
    plt.close(fig)
    console.print(f'[green]Saved[/green] {model_prefix.lower()}_gene_saliency_top20.png')

    # Per-class heatmap (top 30 genes)
    top_30 = np.argsort(overall)[::-1][:30]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(saliency[:, top_30], cmap='YlOrRd',
                xticklabels=[gene_names[i] for i in top_30],
                yticklabels=class_names, ax=ax)
    ax.set_title(f'{model_prefix}: Per-Class Gene Saliency')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_prefix.lower()}_saliency_per_class.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'[green]Saved[/green] {model_prefix.lower()}_saliency_per_class.png')


def interpret_mlp(save_dir):
    """Compute and visualize gene-level saliency for MLP."""
    console.print(Panel('[bold]MLP Gene Saliency[/bold]',
                        border_style='dim', expand=False))
    ckpt = T.find_best_ckpt('CellTypeMLP')
    if ckpt is None:
        console.print('[yellow]No MLP checkpoint found, skipping.[/yellow]')
        return

    ds_test = make_dataset(DATA_DIR, split='test')
    hvg_path = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
    if os.path.exists(hvg_path):
        hvg_idx = np.load(hvg_path)
        ds_test.X = np.asarray(ds_test.X[:, hvg_idx])
        ds_test.gene_names = ds_test.gene_names[hvg_idx]

    saved_kw = T._load_model_kwargs(ckpt, model_name='CellTypeMLP')
    n_features = len(ds_test.gene_names)
    model = T.build_model('CellTypeMLP', n_features, ds_test.n_classes, **saved_kw)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))

    loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
    class_names = list(ds_test.class_names)
    gene_names = [str(g) for g in ds_test.gene_names]

    saliency = _compute_saliency(model, loader, squeeze_channel=True,
                                 n_genes=n_features, n_classes=ds_test.n_classes)
    _plot_saliency(saliency, gene_names, class_names, 'MLP', save_dir)


# ---------------------------------------------------------------------------
# CNN — Gene Saliency + Filters
# ---------------------------------------------------------------------------

def interpret_cnn(save_dir):
    """Compute gene saliency and visualize stem filters for CNN."""
    console.print(Panel('[bold]CNN Gene Saliency + Filters[/bold]',
                        border_style='dim', expand=False))
    ckpt = T.find_best_ckpt('CellTypeCNN')
    if ckpt is None:
        console.print('[yellow]No CNN checkpoint found, skipping.[/yellow]')
        return

    ds_test = make_dataset(DATA_DIR, split='test')
    hvg_path = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
    if os.path.exists(hvg_path):
        hvg_idx = np.load(hvg_path)
        ds_test.X = np.asarray(ds_test.X[:, hvg_idx])
        ds_test.gene_names = ds_test.gene_names[hvg_idx]

    saved_kw = T._load_model_kwargs(ckpt, model_name='CellTypeCNN')
    n_features = len(ds_test.gene_names)
    model = T.build_model('CellTypeCNN', n_features, ds_test.n_classes, **saved_kw)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))

    loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
    class_names = list(ds_test.class_names)
    gene_names = [str(g) for g in ds_test.gene_names]

    # Saliency — CNN input is (batch, 1, n_genes), gradient same shape
    model.eval()
    saliency_sum = np.zeros((ds_test.n_classes, n_features))
    class_counts = np.zeros(ds_test.n_classes)
    for xb, yb in loader:
        xb = xb.to(T.DEVICE)
        yb = yb.to(T.DEVICE)
        xb.requires_grad_(True)
        logits = model(xb)
        pred_logits = logits.gather(1, logits.argmax(1, keepdim=True)).sum()
        pred_logits.backward()
        grad = xb.grad.abs().squeeze(1).detach().cpu().numpy()  # squeeze channel
        yb_np = yb.cpu().numpy()
        for i, label in enumerate(yb_np):
            saliency_sum[label] += grad[i]
            class_counts[label] += 1
        model.zero_grad()

    for c in range(ds_test.n_classes):
        if class_counts[c] > 0:
            saliency_sum[c] /= class_counts[c]

    _plot_saliency(saliency_sum, gene_names, class_names, 'CNN', save_dir)

    # Stem filter visualization
    stem_weights = model.stem[0].weight.detach().cpu().numpy()  # (32, 1, 15)
    n_filters = stem_weights.shape[0]
    cols = 8
    rows = (n_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
    axes = axes.flatten()
    for i in range(n_filters):
        axes[i].plot(stem_weights[i, 0], linewidth=1.5)
        axes[i].set_title(f'Filter {i}', fontsize=7)
        axes[i].tick_params(labelsize=5)
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    plt.suptitle('CNN Stem Conv1d Filters (kernel=15)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cnn_stem_filters.png'), dpi=150)
    plt.close(fig)
    console.print('[green]Saved[/green] cnn_stem_filters.png')


# ---------------------------------------------------------------------------
# Cross-Model Gene Importance Comparison
# ---------------------------------------------------------------------------

def compare_gene_importance(save_dir):
    """Compare top important genes across models."""
    console.print(Panel('[bold]Cross-Model Gene Importance[/bold]',
                        border_style='dim', expand=False))
    top_n = 50
    importance = {}

    # MLP saliency
    ckpt = T.find_best_ckpt('CellTypeMLP')
    if ckpt is not None:
        ds = make_dataset(DATA_DIR, split='test')
        hvg_path = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
        if os.path.exists(hvg_path):
            hvg_idx = np.load(hvg_path)
            ds.X = np.asarray(ds.X[:, hvg_idx])
            ds.gene_names = ds.gene_names[hvg_idx]
        saved_kw = T._load_model_kwargs(ckpt, model_name='CellTypeMLP')
        model = T.build_model('CellTypeMLP', len(ds.gene_names), ds.n_classes, **saved_kw)
        model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        sal = _compute_saliency(model, loader, True, len(ds.gene_names), ds.n_classes)
        overall = sal.mean(axis=0)
        top_idx = np.argsort(overall)[::-1][:top_n]
        importance['MLP'] = set(str(ds.gene_names[i]) for i in top_idx)

    # CNN saliency
    ckpt = T.find_best_ckpt('CellTypeCNN')
    if ckpt is not None:
        ds = make_dataset(DATA_DIR, split='test')
        hvg_path = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
        if os.path.exists(hvg_path):
            hvg_idx = np.load(hvg_path)
            ds.X = np.asarray(ds.X[:, hvg_idx])
            ds.gene_names = ds.gene_names[hvg_idx]
        saved_kw = T._load_model_kwargs(ckpt, model_name='CellTypeCNN')
        model = T.build_model('CellTypeCNN', len(ds.gene_names), ds.n_classes, **saved_kw)
        model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        # CNN saliency with channel dim
        model.eval()
        sal_sum = np.zeros(len(ds.gene_names))
        n = 0
        for xb, yb in loader:
            xb = xb.to(T.DEVICE).requires_grad_(True)
            logits = model(xb)
            logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
            sal_sum += xb.grad.abs().squeeze(1).sum(0).detach().cpu().numpy()
            n += len(yb)
            model.zero_grad()
        sal_sum /= n
        top_idx = np.argsort(sal_sum)[::-1][:top_n]
        importance['CNN'] = set(str(ds.gene_names[i]) for i in top_idx)

    # TOSICA — top pathways mapped to genes
    ckpt = T.find_best_ckpt('CellTypeTOSICA')
    if ckpt is not None and os.path.exists(GMT_PATH):
        ds = make_dataset(DATA_DIR, split='test')
        gene_names = [str(g) for g in ds.gene_names]
        gmt = _parse_gmt(GMT_PATH, max_gene_set_size=MAX_GENE_SET_SIZE)
        kept = _select_pathways(gmt, set(gene_names), MIN_PATHWAY_OVERLAP, MAX_PATHWAYS)
        # Collect mean attention
        mask_t, n_pw = build_pathway_mask(gene_names, gmt_path=GMT_PATH,
                                          min_overlap=MIN_PATHWAY_OVERLAP,
                                          max_pathways=MAX_PATHWAYS,
                                          max_gene_set_size=MAX_GENE_SET_SIZE)
        # Top pathways by attention → genes in those pathways
        all_genes = set()
        for _, genes in kept[:top_n]:
            all_genes.update(genes)
        importance['Transformer'] = all_genes

    if len(importance) < 2:
        console.print('[yellow]Need at least 2 models for comparison, skipping.[/yellow]')
        return

    # Overlap bar chart
    model_names = list(importance.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    pairs = []
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            overlap = len(importance[m1] & importance[m2])
            pairs.append((f'{m1} ∩ {m2}', overlap))
    # All-model intersection
    if len(model_names) >= 3:
        common = importance[model_names[0]]
        for m in model_names[1:]:
            common = common & importance[m]
        pairs.append((f'All ({len(model_names)})', len(common)))

    labels, values = zip(*pairs)
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f'Number of Shared Genes (top {top_n})')
    ax.set_title('Cross-Model Gene Importance Overlap')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gene_importance_comparison.png'), dpi=150)
    plt.close(fig)
    console.print('[green]Saved[/green] gene_importance_comparison.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    console.print(Panel(
        '[bold]INTERPRETABILITY ANALYSIS[/bold]',
        border_style='cyan', expand=False,
    ))

    for fn in (interpret_tosica, interpret_gnn, interpret_mlp,
               interpret_cnn, compare_gene_importance):
        try:
            fn(SAVE_DIR)
        except Exception as e:
            console.print(f'[bold red]Error in {fn.__name__}[/bold red]: {e}')
            import traceback
            traceback.print_exc()

    console.print(Panel(
        f'All figures saved to [bold]{SAVE_DIR}/[/bold]',
        border_style='green', expand=False,
    ))


if __name__ == '__main__':
    main()
