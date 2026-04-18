"""Interpretability analysis for all 4 model architectures."""
from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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

DATA_DIR: str = 'data/10x'
SAVE_DIR: str = 'figures'
BATCH_SIZE: int = 512

# GMT config (must match 4_Transformer.py)
GMT_PATH: str = 'data/reactome.gmt'
MAX_PATHWAYS: int = 300
MIN_PATHWAY_OVERLAP: int = 5
MAX_GENE_SET_SIZE: int = 300


# ---------------------------------------------------------------------------
# TOSICA — Pathway Attention
# ---------------------------------------------------------------------------

def interpret_tosica(save_dir: str) -> None:
    """Extract and visualize pathway attention scores from TOSICA."""
    console.print(Panel('[bold]TOSICA Pathway Attention[/bold]',
                        border_style='dim', expand=False))
    ckpt: str | None = T.find_best_ckpt('CellTypeTOSICA')
    if ckpt is None:
        console.print('[yellow]No TOSICA checkpoint found, skipping.[/yellow]')
        return

    ds_test = make_dataset(DATA_DIR, split='test')
    hvg_path: str = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
    if os.path.exists(hvg_path):
        hvg_idx: np.ndarray = np.load(hvg_path)
        ds_test.X = np.asarray(ds_test.X[:, hvg_idx])
        ds_test.gene_names = ds_test.gene_names[hvg_idx]

    gene_names: list[str] = [str(g) for g in ds_test.gene_names]
    mask: torch.Tensor
    n_pathways: int
    mask, n_pathways = build_pathway_mask(
        gene_names, gmt_path=GMT_PATH, min_overlap=MIN_PATHWAY_OVERLAP,
        max_pathways=MAX_PATHWAYS, max_gene_set_size=MAX_GENE_SET_SIZE)

    # Recover pathway names
    gmt: dict[str, list[str]] = _parse_gmt(GMT_PATH, max_gene_set_size=MAX_GENE_SET_SIZE)
    gene_set: set[str] = set(gene_names)
    kept: list[tuple[str, list[str]]] = _select_pathways(gmt, gene_set, MIN_PATHWAY_OVERLAP, MAX_PATHWAYS)
    pathway_names: list[str] = [name.replace('REACTOME_', '').replace('_', ' ')[:40]
                     for name, _ in kept]

    saved_kw: dict[str, Any] = T._load_model_kwargs(ckpt, model_name='CellTypeTOSICA')
    saved_kw.update(mask=mask, n_pathways=n_pathways)
    model: nn.Module = T.build_model('CellTypeTOSICA', len(gene_names),
                          ds_test.n_classes, **saved_kw)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
    model.eval()

    loader: DataLoader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=T.DEVICE.type == 'cuda')
    class_names: list[str] = list(ds_test.class_names)
    n_classes: int = ds_test.n_classes

    # Collect attention per class
    attn_by_class: dict[int, list[np.ndarray]] = {c: [] for c in range(n_classes)}
    with torch.no_grad():
        for xb, yb in loader:
            xb: torch.Tensor = xb.to(T.DEVICE)
            yb: torch.Tensor = yb.to(T.DEVICE)
            if xb.dim() == 3:
                xb = xb.squeeze(1)
            _: torch.Tensor
            attn: torch.Tensor
            _, attn = model(xb, return_attention=True)
            attn_np: np.ndarray = attn.cpu().numpy()
            yb_np: np.ndarray = yb.cpu().numpy()
            for i, label in enumerate(yb_np):
                attn_by_class[label].append(attn_np[i])

    mean_attn: np.ndarray = np.zeros((n_classes, len(pathway_names)))
    for c in range(n_classes):
        if attn_by_class[c]:
            mean_attn[c] = np.mean(attn_by_class[c], axis=0)

    # Top K pathways by overall attention
    top_k: int = min(30, len(pathway_names))
    overall_importance: np.ndarray = mean_attn.mean(axis=0)
    top_idx: np.ndarray = np.argsort(overall_importance)[::-1][:top_k]

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
    class_counts: list[int] = [len(attn_by_class[c]) for c in range(n_classes)]
    top_classes: np.ndarray = np.argsort(class_counts)[::-1][:3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, c in enumerate(top_classes):
        ax = axes[i]
        class_attn: np.ndarray = mean_attn[c]
        top_pw: np.ndarray = np.argsort(class_attn)[::-1][:10]
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

def interpret_gnn(save_dir: str) -> None:
    """Visualize GNN node embeddings via UMAP."""
    console.print(Panel('[bold]GNN Embedding UMAP[/bold]',
                        border_style='dim', expand=False))
    ckpt: str | None = T.find_best_ckpt('CellTypeGNN')
    if ckpt is None:
        console.print('[yellow]No GNN checkpoint found, skipping.[/yellow]')
        return

    try:
        import umap
    except ImportError:
        console.print('[yellow]umap-learn not installed, skipping GNN interpretability.[/yellow]')
        return

    saved_kw: dict[str, Any] = T._load_model_kwargs(ckpt, model_name='CellTypeGNN')
    k: int = saved_kw.get('k_neighbors', 15)
    data = build_graph_data(DATA_DIR, k_neighbors=k).to(T.DEVICE)
    n_features: int = data.x.shape[1]
    n_classes: int = int(data.y.max().item()) + 1
    model: nn.Module = T.build_model('CellTypeGNN', n_features, n_classes, **saved_kw)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
    model.eval()

    class_names: list[str] = list(np.load(f'{DATA_DIR}/class_names.npy', allow_pickle=True))

    with torch.no_grad():
        embeddings: np.ndarray = model.embed(data.x, data.edge_index).cpu().numpy()
        logits: torch.Tensor = model(data.x, data.edge_index)
        preds: np.ndarray = logits.argmax(1).cpu().numpy()

    labels: np.ndarray = data.y.cpu().numpy()
    test_mask: np.ndarray = data.test_mask.cpu().numpy()

    # Subsample for speed
    n_sample: int = min(10000, embeddings.shape[0])
    rng: np.random.RandomState = np.random.RandomState(42)
    idx: np.ndarray = rng.choice(embeddings.shape[0], n_sample, replace=False)
    emb_sub: np.ndarray = embeddings[idx]
    labels_sub: np.ndarray = labels[idx]
    preds_sub: np.ndarray = preds[idx]

    console.print(f'Running UMAP on {n_sample} nodes...')
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_emb: np.ndarray = reducer.fit_transform(emb_sub)

    # UMAP by class
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(n_classes):
        mask: np.ndarray = labels_sub == c
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
    correct: np.ndarray = labels_sub == preds_sub
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

def _compute_saliency(
    model: nn.Module,
    loader: DataLoader,
    squeeze_channel: bool,
    n_genes: int,
    n_classes: int,
) -> np.ndarray:
    """Compute input-gradient saliency for a non-graph model."""
    model.eval()
    saliency_sum: np.ndarray = np.zeros((n_classes, n_genes))
    class_counts: np.ndarray = np.zeros(n_classes)

    for xb, yb in loader:
        xb: torch.Tensor = xb.to(T.DEVICE)
        yb: torch.Tensor = yb.to(T.DEVICE)
        if squeeze_channel and xb.dim() == 3:
            xb = xb.squeeze(1)
        xb.requires_grad_(True)
        logits: torch.Tensor = model(xb)
        # Backward on predicted class logit for each sample
        pred_logits: torch.Tensor = logits.gather(1, logits.argmax(1, keepdim=True)).sum()
        pred_logits.backward()
        grad: np.ndarray = xb.grad.abs().detach().cpu().numpy()
        yb_np: np.ndarray = yb.cpu().numpy()
        for i, label in enumerate(yb_np):
            saliency_sum[label] += grad[i]
            class_counts[label] += 1
        model.zero_grad()

    # Mean saliency per class
    for c in range(n_classes):
        if class_counts[c] > 0:
            saliency_sum[c] /= class_counts[c]
    return saliency_sum


def _plot_saliency(
    saliency: np.ndarray,
    gene_names: list[str],
    class_names: list[str],
    model_prefix: str,
    save_dir: str,
) -> None:
    """Plot top-20 genes and per-class heatmap from saliency matrix."""
    overall: np.ndarray = saliency.mean(axis=0)
    top_20: np.ndarray = np.argsort(overall)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(20), overall[top_20])
    ax.set_yticks(range(20))
    ax.set_yticklabels([gene_names[i] for i in top_20], fontsize=8)
    ax.set_title(f'{model_prefix}: Top 20 Genes by Mean Saliency')
    ax.set_xlabel('Mean |dL/dx|')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_prefix.lower()}_gene_saliency_top20.png'),
                dpi=150)
    plt.close(fig)
    console.print(f'[green]Saved[/green] {model_prefix.lower()}_gene_saliency_top20.png')

    # Per-class heatmap (top 30 genes)
    top_30: np.ndarray = np.argsort(overall)[::-1][:30]
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


def interpret_mlp(save_dir: str) -> None:
    """Compute and visualize gene-level saliency for MLP."""
    console.print(Panel('[bold]MLP Gene Saliency[/bold]',
                        border_style='dim', expand=False))
    ckpt: str | None = T.find_best_ckpt('CellTypeMLP')
    if ckpt is None:
        console.print('[yellow]No MLP checkpoint found, skipping.[/yellow]')
        return

    ds_test = make_dataset(DATA_DIR, split='test')
    hvg_path: str = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
    if os.path.exists(hvg_path):
        hvg_idx: np.ndarray = np.load(hvg_path)
        ds_test.X = np.asarray(ds_test.X[:, hvg_idx])
        ds_test.gene_names = ds_test.gene_names[hvg_idx]

    saved_kw: dict[str, Any] = T._load_model_kwargs(ckpt, model_name='CellTypeMLP')
    n_features: int = len(ds_test.gene_names)
    model: nn.Module = T.build_model('CellTypeMLP', n_features, ds_test.n_classes, **saved_kw)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))

    loader: DataLoader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
    class_names: list[str] = list(ds_test.class_names)
    gene_names: list[str] = [str(g) for g in ds_test.gene_names]

    saliency: np.ndarray = _compute_saliency(model, loader, squeeze_channel=True,
                                 n_genes=n_features, n_classes=ds_test.n_classes)
    _plot_saliency(saliency, gene_names, class_names, 'MLP', save_dir)


# ---------------------------------------------------------------------------
# CNN — Gene Saliency + Filters
# ---------------------------------------------------------------------------

def interpret_cnn(save_dir: str) -> None:
    """Compute gene saliency and visualize stem filters for CNN."""
    console.print(Panel('[bold]CNN Gene Saliency + Filters[/bold]',
                        border_style='dim', expand=False))
    ckpt: str | None = T.find_best_ckpt('CellTypeCNN')
    if ckpt is None:
        console.print('[yellow]No CNN checkpoint found, skipping.[/yellow]')
        return

    ds_test = make_dataset(DATA_DIR, split='test')
    hvg_path: str = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
    if os.path.exists(hvg_path):
        hvg_idx: np.ndarray = np.load(hvg_path)
        ds_test.X = np.asarray(ds_test.X[:, hvg_idx])
        ds_test.gene_names = ds_test.gene_names[hvg_idx]

    saved_kw: dict[str, Any] = T._load_model_kwargs(ckpt, model_name='CellTypeCNN')
    n_features: int = len(ds_test.gene_names)
    model: nn.Module = T.build_model('CellTypeCNN', n_features, ds_test.n_classes, **saved_kw)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))

    loader: DataLoader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
    class_names: list[str] = list(ds_test.class_names)
    gene_names: list[str] = [str(g) for g in ds_test.gene_names]

    # Saliency — CNN input is (batch, 1, n_genes), gradient same shape
    model.eval()
    saliency_sum: np.ndarray = np.zeros((ds_test.n_classes, n_features))
    class_counts: np.ndarray = np.zeros(ds_test.n_classes)
    for xb, yb in loader:
        xb: torch.Tensor = xb.to(T.DEVICE)
        yb: torch.Tensor = yb.to(T.DEVICE)
        xb.requires_grad_(True)
        logits: torch.Tensor = model(xb)
        pred_logits: torch.Tensor = logits.gather(1, logits.argmax(1, keepdim=True)).sum()
        pred_logits.backward()
        grad: np.ndarray = xb.grad.abs().squeeze(1).detach().cpu().numpy()  # squeeze channel
        yb_np: np.ndarray = yb.cpu().numpy()
        for i, label in enumerate(yb_np):
            saliency_sum[label] += grad[i]
            class_counts[label] += 1
        model.zero_grad()

    for c in range(ds_test.n_classes):
        if class_counts[c] > 0:
            saliency_sum[c] /= class_counts[c]

    _plot_saliency(saliency_sum, gene_names, class_names, 'CNN', save_dir)

    # Stem filter visualization
    stem_weights: np.ndarray = model.stem[0].weight.detach().cpu().numpy()  # (32, 1, 15)
    n_filters: int = stem_weights.shape[0]
    cols: int = 8
    rows: int = (n_filters + cols - 1) // cols
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

def compare_gene_importance(save_dir: str) -> None:
    """Compare top important genes across models."""
    console.print(Panel('[bold]Cross-Model Gene Importance[/bold]',
                        border_style='dim', expand=False))
    top_n: int = 50
    importance: dict[str, set[str]] = {}

    # MLP saliency
    ckpt: str | None = T.find_best_ckpt('CellTypeMLP')
    if ckpt is not None:
        ds = make_dataset(DATA_DIR, split='test')
        hvg_path: str = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
        if os.path.exists(hvg_path):
            hvg_idx: np.ndarray = np.load(hvg_path)
            ds.X = np.asarray(ds.X[:, hvg_idx])
            ds.gene_names = ds.gene_names[hvg_idx]
        saved_kw: dict[str, Any] = T._load_model_kwargs(ckpt, model_name='CellTypeMLP')
        model: nn.Module = T.build_model('CellTypeMLP', len(ds.gene_names), ds.n_classes, **saved_kw)
        model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
        loader: DataLoader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        sal: np.ndarray = _compute_saliency(model, loader, True, len(ds.gene_names), ds.n_classes)
        overall: np.ndarray = sal.mean(axis=0)
        top_idx: np.ndarray = np.argsort(overall)[::-1][:top_n]
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
        sal_sum: np.ndarray = np.zeros(len(ds.gene_names))
        n: int = 0
        for xb, yb in loader:
            xb: torch.Tensor = xb.to(T.DEVICE).requires_grad_(True)
            logits: torch.Tensor = model(xb)
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
        gene_names: list[str] = [str(g) for g in ds.gene_names]
        gmt: dict[str, list[str]] = _parse_gmt(GMT_PATH, max_gene_set_size=MAX_GENE_SET_SIZE)
        kept: list[tuple[str, list[str]]] = _select_pathways(gmt, set(gene_names), MIN_PATHWAY_OVERLAP, MAX_PATHWAYS)
        # Collect mean attention
        mask_t: torch.Tensor
        n_pw: int
        mask_t, n_pw = build_pathway_mask(gene_names, gmt_path=GMT_PATH,
                                          min_overlap=MIN_PATHWAY_OVERLAP,
                                          max_pathways=MAX_PATHWAYS,
                                          max_gene_set_size=MAX_GENE_SET_SIZE)
        # Top pathways by attention -> genes in those pathways
        all_genes: set[str] = set()
        for _, genes in kept[:top_n]:
            all_genes.update(genes)
        importance['Transformer'] = all_genes

    if len(importance) < 2:
        console.print('[yellow]Need at least 2 models for comparison, skipping.[/yellow]')
        return

    # Overlap bar chart
    model_names: list[str] = list(importance.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    pairs: list[tuple[str, int]] = []
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            overlap: int = len(importance[m1] & importance[m2])
            pairs.append((f'{m1} \u2229 {m2}', overlap))
    # All-model intersection
    if len(model_names) >= 3:
        common: set[str] = importance[model_names[0]]
        for m in model_names[1:]:
            common = common & importance[m]
        pairs.append((f'All ({len(model_names)})', len(common)))

    labels: tuple[str, ...]
    values: tuple[int, ...]
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

def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    console.print(Panel(
        '[bold]INTERPRETABILITY ANALYSIS[/bold]',
        border_style='cyan', expand=False,
    ))

    fn: Callable[[str], None]
    for fn in (interpret_tosica, interpret_gnn, interpret_mlp,
               interpret_cnn, compare_gene_importance):
        try:
            fn(SAVE_DIR)
        except Exception as e:
            console.print(f'[bold red]Error in {fn.__name__}[/bold red]: {e}')
            traceback.print_exc()

    console.print(Panel(
        f'All figures saved to [bold]{SAVE_DIR}/[/bold]',
        border_style='green', expand=False,
    ))


if __name__ == '__main__':
    main()
