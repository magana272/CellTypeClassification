import gc
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import polars as pl
from tqdm.auto import tqdm


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

# Canonical subclass vocabulary shared between 10x (M1) and SMART-seq (MTG).
# Any raw label not present here is dropped.
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


def _load_metadata(path: str) -> pd.DataFrame:
    meta = pd.read_csv(path, usecols=['sample_name', 'subclass_label'])
    meta = meta.dropna(subset=['subclass_label'])
    print(f'Metadata: {len(meta):,} labelled cells')
    return meta


def _canonicalize_labels(meta: pd.DataFrame) -> pd.DataFrame:
    mapped = meta['subclass_label'].map(CANONICAL_LABEL_MAP)
    if mapped.isna().any():
        dropped = meta.loc[mapped.isna(), 'subclass_label'].value_counts()
        print(f'Dropping {int(mapped.isna().sum()):,} cells '
              f'from non-shared labels: {dropped.to_dict()}')
    meta = meta.loc[mapped.notna()].copy()
    meta['subclass_label'] = mapped.dropna().values
    return meta.reset_index(drop=True)


def _filter_rare_classes(meta: pd.DataFrame, min_cells: int) -> pd.DataFrame:
    counts = meta['subclass_label'].value_counts()
    keep = counts[counts >= min_cells].index
    return meta[meta['subclass_label'].isin(keep)].reset_index(drop=True)


def _split_indices(y: np.ndarray, seed: int):
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


def _load_matrix(csv_path: str, sample_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load expression matrix aligned to sample_names, with .npy caching."""
    root, _ = os.path.splitext(csv_path)
    mat_path = root + '.npy'
    names_path = root + '_cells.npy'
    genes_path = root + '_genes.npy'

    if not (os.path.exists(mat_path) and os.path.exists(names_path)):
        print(f'Caching {csv_path} -> {mat_path} (first run only)')
        header = pl.read_csv(csv_path, n_rows=0).columns
        name_col, gene_cols = header[0], header[1:]
        df = pl.read_csv(
            csv_path,
            schema_overrides={name_col: pl.Utf8,
                              **{c: pl.Float32 for c in gene_cols}},
        )
        np.save(names_path, df[name_col].to_numpy().astype(str))
        np.save(genes_path, np.asarray(gene_cols, dtype=str))

        mat = np.lib.format.open_memmap(
            mat_path, mode='w+', dtype=np.float32,
            shape=(df.height, len(gene_cols)),
        )
        chunk = 2048
        for ci in tqdm(range(0, len(gene_cols), chunk), desc='caching matrix'):
            j1 = min(ci + chunk, len(gene_cols))
            mat[:, ci:j1] = df.select(gene_cols[ci:j1]).to_numpy()
        mat.flush()
        del df, mat
        gc.collect()

    X_full = np.load(mat_path, mmap_mode='r')
    names_full = np.load(names_path)
    gene_names = np.load(genes_path)

    row_of = {n: i for i, n in enumerate(names_full)}
    rows = np.fromiter(
        (row_of[n] for n in sample_names), dtype=np.int64, count=len(sample_names),
    )
    X = np.ascontiguousarray(X_full[rows])
    print(f'Expression matrix: {X.shape}')
    return X, gene_names


def load_dataset(paths: dict, seed: int = 42) -> str:
    """Load CSV, split into train/val/test, save as raw .npy files.

    Saves to paths['dir']: X_train.npy, y_train.npy, X_val.npy, y_val.npy,
    X_test.npy, y_test.npy, gene_names.npy, class_names.npy.

    Returns the output directory path.
    """
    out_dir = paths['dir']

    if os.path.exists(os.path.join(out_dir, 'X_train.npy')):
        print(f'Splits already exist in {out_dir}')
        return out_dir

    meta = _load_metadata(paths['metadata'])
    meta = _canonicalize_labels(meta)
    meta = _filter_rare_classes(meta, MIN_CELLS_PER_CLASS)

    le = LabelEncoder()
    y = le.fit_transform(meta['subclass_label'].values)
    class_names = le.classes_
    print(f'After filtering: {len(meta):,} cells, {len(class_names)} classes')

    idx_train, idx_val, idx_test, y_train, y_val, y_test = _split_indices(y, seed)
    total = len(y)
    print(f'Train: {len(idx_train):>6,} ({len(idx_train)/total*100:.1f}%)')
    print(f'Val:   {len(idx_val):>6,} ({len(idx_val)/total*100:.1f}%)')
    print(f'Test:  {len(idx_test):>6,} ({len(idx_test)/total*100:.1f}%)')

    X, gene_names = _load_matrix(paths['matrix'], meta['sample_name'].values)

    os.makedirs(out_dir, exist_ok=True)
    for name, idx, y_split in [('train', idx_train, y_train),
                                ('val', idx_val, y_val),
                                ('test', idx_test, y_test)]:
        np.save(os.path.join(out_dir, f'X_{name}.npy'), X[idx])
        np.save(os.path.join(out_dir, f'y_{name}.npy'), y_split)
    np.save(os.path.join(out_dir, 'gene_names.npy'), gene_names)
    np.save(os.path.join(out_dir, 'class_names.npy'), class_names)

    print(f'Saved raw splits to {out_dir}')
    return out_dir


def load_10x(seed: int = 42) -> str:
    return load_dataset(DEFAULT_10X_PATHS, seed)


def load_smartseq(seed: int = 42) -> str:
    return load_dataset(DEFAULT_SMARTSEQ_PATHS, seed)
