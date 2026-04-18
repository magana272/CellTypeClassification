from __future__ import annotations

import gc
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from rich.console import Console
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

console = Console()


# ---------------------------------------------------------------------------
# Typed dataset configuration (replaces raw dicts)
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Per-dataset metadata for the registry."""

    dir: str
    loader: str  # 'csv' or 'h5ad'
    label_col: str = 'subclass_label'
    min_cells: int = 200

    def __getitem__(self, key: str) -> str | int:
        """Dict-style access for backward compatibility."""
        return getattr(self, key)

    def get(self, key: str, default: str | int | None = None) -> str | int | None:
        return getattr(self, key, default)


DEFAULT_10X_PATHS = {
    'matrix': 'data/10x/matrix.csv',
    'metadata': 'data/10x/metadata.csv',
    'dir': 'data/10x',
}
DEFAULT_SMARTSEQ_PATHS = {
    'matrix': 'data/smartseq/smartseq_data.csv',
    'metadata': 'data/smartseq/smartseq_meta.csv',
    'dir': 'data/smartseq',
}

MIN_CELLS_PER_CLASS = 200
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10

# ---------------------------------------------------------------------------
# Dataset registry — every dataset the project knows about
# ---------------------------------------------------------------------------
ALL_DATASETS: dict[str, DatasetConfig] = {
    '10x':          DatasetConfig(dir='data/10x',          loader='csv'),
    'smartseq':     DatasetConfig(dir='data/smartseq',     loader='csv'),
    'pbmc':         DatasetConfig(dir='data/pbmc',         loader='h5ad',
                                  label_col='cell_type',          min_cells=30),
    'pancreas':     DatasetConfig(dir='data/pancreas',     loader='h5ad',
                                  label_col='cell_type',          min_cells=50),
    'tabula_muris': DatasetConfig(dir='data/tabula_muris', loader='h5ad',
                                  label_col='cell_ontology_class', min_cells=100),
    'lung':         DatasetConfig(dir='data/lung',         loader='h5ad',
                                  label_col='cell_type',     min_cells=50),
}

CANONICAL_LABEL_MAP: dict[str, str] = {
    'Astro': 'Astro', 'Astrocyte': 'Astro',
    'Oligo': 'Oligo', 'Oligodendrocyte': 'Oligo',
    'OPC': 'OPC',
    'Lamp5': 'LAMP5', 'LAMP5': 'LAMP5',
    'Pvalb': 'PVALB', 'PVALB': 'PVALB',
    'Sst': 'SST',     'SST': 'SST',
    'Vip': 'VIP',     'VIP': 'VIP',
    'L2/3 IT': 'IT', 'L5 IT': 'IT', 'L6 IT': 'IT',
    'IT': 'IT',      'L4 IT': 'IT',
    'L6 IT Car3': 'IT Car3', 'L5/6 IT Car3': 'IT Car3',
    'L5/6 NP': 'L5/6 NP',
    'L6 CT': 'L6 CT',
    'L6b': 'L6b',
}


def load_metadata(path: str) -> pd.DataFrame:
    meta = pd.read_csv(path, usecols=['sample_name', 'subclass_label'])
    meta = meta.dropna(subset=['subclass_label'])
    mapped = meta['subclass_label'].map(CANONICAL_LABEL_MAP)
    meta = meta.loc[mapped.notna()].copy()
    meta['subclass_label'] = mapped.dropna().values
    counts = meta['subclass_label'].value_counts()
    keep = counts[counts >= MIN_CELLS_PER_CLASS].index
    meta = meta[meta['subclass_label'].isin(keep)].reset_index(drop=True)
    console.print(f'Metadata: {len(meta):,} cells, {meta["subclass_label"].nunique()} classes')
    return meta


def split_indices(
    y: np.ndarray, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=VAL_FRAC + TEST_FRAC, stratify=y, random_state=seed,
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=y_temp, random_state=seed,
    )
    return idx_train, idx_val, idx_test, y_train, y_val, y_test


def cache_matrix(csv_path: str) -> None:
    root, _ = os.path.splitext(csv_path)
    mat_path, names_path, genes_path = root + '.npy', root + '_cells.npy', root + '_genes.npy'
    if os.path.exists(mat_path) and os.path.exists(names_path) and os.path.exists(genes_path):
        return
    console.print(f'Caching {csv_path} -> {mat_path} (first run only)')
    header = pl.read_csv(csv_path, n_rows=0).columns
    name_col, gene_cols = header[0], header[1:]
    df = pl.read_csv(csv_path, schema_overrides={name_col: pl.Utf8, **{c: pl.Float32 for c in gene_cols}}, rechunk=False)
    np.save(names_path, df[name_col].to_numpy().astype(str))
    np.save(genes_path, np.asarray(gene_cols, dtype=str))
    mat = df.drop(name_col).to_numpy().astype(np.float32, copy=False)
    np.save(mat_path, mat)
    del df, mat; gc.collect()


def load_matrix(csv_path: str, sample_names: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_matrix(csv_path)
    root, _ = os.path.splitext(csv_path)
    X_mmap = np.load(root + '.npy', mmap_mode='r')
    names_full = np.load(root + '_cells.npy')
    gene_names = np.load(root + '_genes.npy')
    order = np.argsort(names_full)
    row_map = order[np.searchsorted(names_full, sample_names, sorter=order)]
    console.print(f'Expression matrix: ({len(sample_names)}, {X_mmap.shape[1]})')
    return X_mmap, row_map, gene_names


def load_dataset(paths: dict[str, str], seed: int = 42) -> str:
    out_dir = paths['dir']
    if os.path.exists(os.path.join(out_dir, 'X_train.npy')):
        console.print(f'Splits already exist in {out_dir}')
        return out_dir

    meta = load_metadata(paths['metadata'])
    le = LabelEncoder()
    y = le.fit_transform(meta['subclass_label'].values)
    idx_train, idx_val, idx_test, y_train, y_val, y_test = split_indices(y, seed)
    X_mmap, row_map, gene_names = load_matrix(paths['matrix'], meta['sample_name'].values)

    os.makedirs(out_dir, exist_ok=True)
    for name, idx, y_split in [('train', idx_train, y_train), ('val', idx_val, y_val), ('test', idx_test, y_test)]:
        mmap_rows = row_map[idx]
        sort_order = np.argsort(mmap_rows)
        sorted_rows = mmap_rows[sort_order]
        mm = np.lib.format.open_memmap(
            os.path.join(out_dir, f'X_{name}.npy'),
            mode='w+', dtype=X_mmap.dtype, shape=(len(sorted_rows), X_mmap.shape[1]),
        )
        np.take(X_mmap, sorted_rows, axis=0, out=mm)
        del mm
        np.save(os.path.join(out_dir, f'y_{name}.npy'), y_split[sort_order])
    del X_mmap; gc.collect()

    np.save(os.path.join(out_dir, 'gene_names.npy'), gene_names)
    np.save(os.path.join(out_dir, 'class_names.npy'), le.classes_)
    with open(os.path.join(out_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    console.print(f'[green]Saved[/green] splits to {out_dir}')
    return out_dir


def load_10x(seed: int = 42) -> str:
    return load_dataset(DEFAULT_10X_PATHS, seed)


def load_smartseq(seed: int = 42) -> str:
    return load_dataset(DEFAULT_SMARTSEQ_PATHS, seed)


# ---------------------------------------------------------------------------
# h5ad-based dataset loading (benchmark datasets)
# ---------------------------------------------------------------------------

def load_h5ad_dataset(
    h5ad_path: str,
    out_dir: str,
    label_column: str = 'cell_type',
    min_cells: int = 50,
    seed: int = 42,
) -> str:
    """Load an h5ad file -> filter by min cells -> stratified split -> save.

    Saves sparse (.npz) when the source h5ad has a sparse matrix, or
    dense (.npy) otherwise.  Downstream ``GeneExpressionDataset`` handles both.
    """
    # Check for both sparse (.npz) and dense (.npy) formats
    if (os.path.exists(os.path.join(out_dir, 'X_train.npy'))
            or os.path.exists(os.path.join(out_dir, 'X_train.npz'))):
        console.print(f'Splits already exist in {out_dir}')
        return out_dir

    import anndata as ad
    import scipy.sparse

    adata = ad.read_h5ad(h5ad_path)
    is_sparse = scipy.sparse.issparse(adata.X)

    # Labels
    if label_column not in adata.obs.columns:
        available = list(adata.obs.columns)
        raise KeyError(
            f'label_column {label_column!r} not in obs. '
            f'Available: {available}')
    labels = adata.obs[label_column].astype(str).values
    gene_names = np.array(adata.var_names)

    # Filter classes with too few cells
    counts = pd.Series(labels).value_counts()
    keep_classes = counts[counts >= min_cells].index
    cell_mask = np.isin(labels, keep_classes)

    if is_sparse:
        X_sp = adata.X[cell_mask]
        if not scipy.sparse.issparse(X_sp):
            X_sp = scipy.sparse.csr_matrix(X_sp)
        elif not scipy.sparse.isspmatrix_csr(X_sp):
            X_sp = X_sp.tocsr()
        n_cells, n_genes = X_sp.shape
    else:
        X = np.asarray(adata.X[cell_mask], dtype=np.float32)
        n_cells, n_genes = X.shape
    labels = labels[cell_mask]

    console.print(f'h5ad loaded: {n_cells:,} cells, {n_genes:,} genes, '
                  f'{len(keep_classes)} classes (min_cells={min_cells})'
                  f'{" [sparse]" if is_sparse else ""}')

    le = LabelEncoder()
    y = le.fit_transform(labels)

    idx_train, idx_val, idx_test, y_train, y_val, y_test = split_indices(y, seed)

    os.makedirs(out_dir, exist_ok=True)
    for name, idx, y_split in [('train', idx_train, y_train),
                                ('val', idx_val, y_val),
                                ('test', idx_test, y_test)]:
        if is_sparse:
            scipy.sparse.save_npz(
                os.path.join(out_dir, f'X_{name}.npz'),
                X_sp[idx].tocsr(),
            )
        else:
            np.save(os.path.join(out_dir, f'X_{name}.npy'),
                    X[idx].astype(np.float32))
        np.save(os.path.join(out_dir, f'y_{name}.npy'), y_split)

    np.save(os.path.join(out_dir, 'gene_names.npy'), gene_names)
    np.save(os.path.join(out_dir, 'class_names.npy'), le.classes_)
    with open(os.path.join(out_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    console.print(f'[green]Saved[/green] splits to {out_dir}'
                  f'{" (sparse .npz)" if is_sparse else ""}')
    return out_dir


def load_pbmc(seed: int = 42) -> str:
    """Load PBMC dataset. Falls back to scanpy built-in if h5ad not found."""
    h5ad = 'data/pbmc/pbmc.h5ad'
    if not os.path.exists(h5ad):
        os.makedirs('data/pbmc', exist_ok=True)
        console.print('Downloading PBMC via scanpy ...')
        import scanpy as sc
        adata = sc.datasets.pbmc3k_processed()
        # Rename louvain clusters to cell_type for consistency
        adata.obs['cell_type'] = adata.obs['louvain'].astype(str)
        adata.write_h5ad(h5ad)
    return load_h5ad_dataset(h5ad, 'data/pbmc',
                             label_column='cell_type', min_cells=30, seed=seed)


def load_pancreas(seed: int = 42) -> str:
    """Load human pancreas dataset (Baron et al.) from figshare."""
    from allen_brain.cell_data.cell_download import H5AD_SOURCES, download_h5ad
    cfg = ALL_DATASETS['pancreas']
    h5ad = os.path.join(cfg['dir'], 'pancreas.h5ad')
    if not os.path.exists(h5ad):
        console.print('Downloading Baron human pancreas from figshare ...')
        download_h5ad(H5AD_SOURCES['pancreas'], h5ad)
    return load_h5ad_dataset(h5ad, cfg['dir'],
                             label_column="celltype",
                             min_cells=cfg['min_cells'], seed=seed)


def load_tabula_muris(seed: int = 42) -> str:
    """Load Tabula Muris (mouse, droplet) from figshare."""
    from allen_brain.cell_data.cell_download import H5AD_SOURCES, download_h5ad
    cfg = ALL_DATASETS['tabula_muris']
    h5ad = os.path.join(cfg['dir'], 'tabula_muris.h5ad')
    if not os.path.exists(h5ad):
        console.print('Downloading Tabula Muris droplet from figshare ...')
        download_h5ad(H5AD_SOURCES['tabula_muris'], h5ad)
    return load_h5ad_dataset(h5ad, cfg['dir'],
                             label_column=cfg['label_col'],
                             min_cells=cfg['min_cells'], seed=seed)


def load_lung(seed: int = 42) -> str:
    """Load human lung dataset (Adams et al. IPF) from figshare."""
    from allen_brain.cell_data.cell_download import H5AD_SOURCES, download_h5ad
    cfg = ALL_DATASETS['lung']
    h5ad = os.path.join(cfg['dir'], 'lung.h5ad')
    if not os.path.exists(h5ad):
        console.print('Downloading human lung atlas from figshare ...')
        download_h5ad(H5AD_SOURCES['lung'], h5ad)
    return load_h5ad_dataset(h5ad, cfg['dir'],
                             label_column=cfg['label_col'],
                             min_cells=cfg['min_cells'], seed=seed)
