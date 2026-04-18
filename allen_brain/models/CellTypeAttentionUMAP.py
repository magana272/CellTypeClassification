"""TOSICA variant with scanpy-style attention embedding pipeline.

Adds the paper's preprocessing of attention scores:
    library-size normalize -> PCA -> k-NN graph -> UMAP
and random gene subsampling (default 5000 genes).
"""

import numpy as np
import scanpy as sc
import anndata as ad
import torch
import torch.nn.functional as F
from rich.console import Console

from .CellTypeAttention import TOSICA, build_pathway_mask

console = Console()



def select_random_cells(n_cells, n_select=5000, seed=42):
    """Return sorted indices of a random subset of cells."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_cells, size=min(n_select, n_cells), replace=False)
    return np.sort(idx)



@torch.no_grad()
def collect_attention(model, loader, device, squeeze_channel=True):
    """Run forward pass and return (attention_matrix, labels, predictions, max_probs).

    attention_matrix : (N, n_pathways)  raw attention scores
    labels           : (N,)            ground-truth int labels
    predictions      : (N,)            argmax predictions
    max_probs        : (N,)            max softmax probability per cell
    """
    model.eval()
    all_attn, all_labels, all_preds, all_probs = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        if squeeze_channel and xb.dim() == 3:
            xb = xb.squeeze(1)
        logits, attn = model(xb, return_attention=True)
        probs = F.softmax(logits, dim=-1)
        max_p, preds = probs.max(dim=-1)
        all_attn.append(attn.cpu().numpy())
        all_labels.append(yb.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(max_p.cpu().numpy())
    return (np.concatenate(all_attn),
            np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs))



def attention_umap(attn_matrix, labels, class_names,
                   n_pcs=30, n_neighbors=15, min_dist=0.3):
    """Paper pipeline: normalize -> PCA -> k-NN -> UMAP on attention scores.

    Parameters
    ----------
    attn_matrix : (N, n_pathways) raw attention scores
    labels      : (N,) integer labels
    class_names : list of str
    n_pcs       : PCA components
    n_neighbors : k-NN neighbors for UMAP
    min_dist    : UMAP min_dist

    Returns
    -------
    adata : AnnData with .obsm['X_umap'], .obs['cell_type']
    """
    adata = ad.AnnData(X=attn_matrix.astype(np.float32))
    adata.obs['cell_type'] = [class_names[l] for l in labels]
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')

    # Step 1: library-size normalize (scale to 10,000 per cell, no log)
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Step 2: PCA
    n_pcs = min(n_pcs, attn_matrix.shape[1] - 1, attn_matrix.shape[0] - 1)
    sc.tl.pca(adata, n_comps=n_pcs)

    # Step 3: k-NN graph from PCA
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # Step 4: UMAP
    sc.tl.umap(adata, min_dist=min_dist)

    return adata
