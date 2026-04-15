import numpy as np
from sklearn.preprocessing import StandardScaler


def _gene_filter(X_val: np.ndarray, min_gene_frac: float) -> np.ndarray:
    """Keep genes expressed (>0) in at least max(3, frac * n_val) val cells."""
    min_cells = max(3, int(min_gene_frac * len(X_val)))
    nonzero = (X_val > 0).sum(axis=0)
    keep = np.flatnonzero(nonzero >= min_cells)
    print(f'Gene filter: {keep.size:,} / {X_val.shape[1]:,} genes '
          f'(>= {min_cells} val cells)')
    return keep


def select_hvg(X_val_filtered: np.ndarray, n_hvg: int) -> np.ndarray:
    """Top-variance genes from log-normalized data, sorted by descending variance."""
    lib = np.maximum(X_val_filtered.sum(axis=1, keepdims=True, dtype=np.float64), 1.0)
    normed = np.log1p(X_val_filtered / lib * 1e4).astype(np.float32)
    var = normed.var(axis=0)
    n = min(n_hvg, var.size)
    top = np.argpartition(-var, n - 1)[:n]
    return top[np.argsort(-var[top])]


def _normalize(X: np.ndarray, filtered_idx: np.ndarray,
               hvg_local: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Library-size normalize + log1p, keeping only HVG columns.

    Library size is summed over filtered genes; only HVG columns are returned.
    """
    out = np.empty((len(X), hvg_local.size), dtype=np.float32)
    for start in range(0, len(X), chunk):
        end = min(start + chunk, len(X))
        block = X[start:end][:, filtered_idx]
        lib = np.maximum(block.sum(axis=1, keepdims=True, dtype=np.float64), 1.0)
        hvg = block[:, hvg_local].astype(np.float32, copy=False)
        out[start:end] = np.log1p(hvg / lib.astype(np.float32) * 1e4)
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
    print(f'HVG: {hvg_local.size} genes selected')

    X_train = _normalize(X_train, filtered_idx, hvg_local)
    X_val = _normalize(X_val, filtered_idx, hvg_local)
    X_test = _normalize(X_test, filtered_idx, hvg_local)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_val, X_test, gene_names[hvg_gene_idx], scaler


def align_genes(X_source, gene_names_source, gene_names_target):
    """Reindex X_source columns to match gene_names_target order.

    Genes in target but missing from source get zero-filled.
    Genes in source but missing from target are dropped.

    Returns X_aligned (n_samples, len(gene_names_target)) and overlap count.
    """
    source_map = {g: i for i, g in enumerate(gene_names_source)}
    n_samples = X_source.shape[0]
    n_target = len(gene_names_target)
    X_aligned = np.zeros((n_samples, n_target), dtype=np.float32)
    matched = 0
    for j, gene in enumerate(gene_names_target):
        src_idx = source_map.get(gene)
        if src_idx is not None:
            X_aligned[:, j] = X_source[:, src_idx]
            matched += 1
    print(f'Gene alignment: {matched}/{n_target} target genes matched '
          f'({n_target - matched} zero-filled)')
    return X_aligned, matched
