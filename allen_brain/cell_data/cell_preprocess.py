from __future__ import annotations

import numpy as np
import scipy.sparse
import torch
from rich.console import Console
from sklearn.preprocessing import StandardScaler

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
console = Console()


def _gene_filter(X_val: np.ndarray | scipy.sparse.spmatrix, min_gene_frac: float) -> np.ndarray:
    """Keep genes expressed (>0) in at least max(3, frac * n_val) val cells.

    Runs on GPU when available; returns a numpy index array.
    """
    min_cells = max(3, int(min_gene_frac * X_val.shape[0]))
    if scipy.sparse.issparse(X_val):
        X_val = np.asarray(X_val.todense())
    X_t = torch.as_tensor(np.asarray(X_val), device=_DEVICE)
    nonzero = (X_t > 0).sum(dim=0)
    keep = torch.nonzero(nonzero >= min_cells, as_tuple=False).squeeze(1)
    console.print(f'Gene filter: {keep.numel():,} / {X_val.shape[1]:,} genes '
          f'(>= {min_cells} val cells)')
    return keep.cpu().numpy()


def select_hvg(X_val_filtered: np.ndarray | scipy.sparse.spmatrix, n_hvg: int) -> np.ndarray:
    """Top-variance genes from log-normalized data, sorted by descending variance."""
    if scipy.sparse.issparse(X_val_filtered):
        X_val_filtered = np.asarray(X_val_filtered.todense())
    X_t = torch.as_tensor(X_val_filtered, dtype=torch.float32)
    lib = X_t.sum(dim=1, keepdim=True).clamp(min=1.0)
    normed = torch.log1p(X_t / lib * 1e4)
    del X_t
    var = normed.var(dim=0)
    del normed
    n = min(n_hvg, var.numel())
    top_vals, top_idx = torch.topk(var, n)
    sorted_order = torch.argsort(top_vals, descending=True)
    return top_idx[sorted_order].numpy()


def _normalize(X: np.ndarray, filtered_idx: np.ndarray,
               hvg_local: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Library-size normalize + log1p, keeping only HVG columns.

    Library size is summed over filtered genes; only HVG columns are returned.
    Runs on GPU when available; returns a numpy array.
    """
    filt_t = torch.as_tensor(filtered_idx, dtype=torch.long, device=_DEVICE)
    hvg_t = torch.as_tensor(hvg_local, dtype=torch.long, device=_DEVICE)
    out = np.empty((len(X), hvg_local.size), dtype=np.float32)
    for start in range(0, len(X), chunk):
        end = min(start + chunk, len(X))
        block = torch.as_tensor(
            np.asarray(X[start:end]), dtype=torch.float64, device=_DEVICE
        )
        block_filt = block[:, filt_t]
        lib = block_filt.sum(dim=1, keepdim=True).clamp(min=1.0)
        hvg_block = block_filt[:, hvg_t].float()
        out[start:end] = torch.log1p(hvg_block / lib.float() * 1e4).cpu().numpy()
    return out


def preprocess_hvg(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    gene_names: np.ndarray,
    n_hvg: int = 2000,
    min_gene_frac: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Gene filter -> HVG selection -> log-norm -> StandardScaler.

    Returns (X_train, X_val, X_test, hvg_gene_names, scaler).
    """
    filtered_idx = _gene_filter(X_val, min_gene_frac)
    hvg_local = select_hvg(X_val[:, filtered_idx], n_hvg)
    hvg_gene_idx = filtered_idx[hvg_local]
    console.print(f'HVG: {hvg_local.size} genes selected')

    X_train = _normalize(X_train, filtered_idx, hvg_local)
    X_val = _normalize(X_val, filtered_idx, hvg_local)
    X_test = _normalize(X_test, filtered_idx, hvg_local)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_val, X_test, gene_names[hvg_gene_idx], scaler


def align_genes(
    X_source: np.ndarray,
    gene_names_source: np.ndarray,
    gene_names_target: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Reindex X_source columns to match gene_names_target order.

    Genes in target but missing from source get zero-filled.
    Genes in source but missing from target are dropped.
    Runs on GPU when available; returns a numpy array.

    Returns X_aligned (n_samples, len(gene_names_target)) and overlap count.
    """
    source_map = {g: i for i, g in enumerate(gene_names_source)}
    n_samples = X_source.shape[0]
    n_target = len(gene_names_target)

    # Build mapping: target col j -> source col, or -1 if missing
    src_indices = []
    matched = 0
    for gene in gene_names_target:
        idx = source_map.get(gene, -1)
        src_indices.append(idx)
        if idx >= 0:
            matched += 1

    src_idx_t = torch.tensor(src_indices, dtype=torch.long, device=_DEVICE)
    has_match = src_idx_t >= 0

    X_t = torch.as_tensor(np.asarray(X_source), dtype=torch.float32, device=_DEVICE)
    X_aligned = torch.zeros(n_samples, n_target, dtype=torch.float32, device=_DEVICE)
    X_aligned[:, has_match] = X_t[:, src_idx_t[has_match]]

    console.print(f'Gene alignment: {matched}/{n_target} target genes matched '
          f'({n_target - matched} zero-filled)')
    return X_aligned.cpu().numpy(), matched
