import gc
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import polars as pl


DEFAULT_10X_PATHS = {
    'matrix': 'data/10x/matrix.csv',
    'metadata': 'data/10x/metadata.csv',
}
DEFAULT_SMARTSEQ_PATHS = {
    'matrix': 'data/smartseq/smartseq_data.csv',
    'metadata': 'data/smartseq/smartseq_meta.csv',
}

MIN_CELLS_PER_CLASS = 200
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10

# Canonical subclass vocabulary shared between 10x (M1) and SMART-seq (MTG).
# Any raw label not present here is dropped: it's either dataset-specific
# (no counterpart in the other dataset) or unknown.
CANONICAL_LABEL_MAP: dict[str, str] = {
    # Glia
    'Astro': 'Astro', 'Astrocyte': 'Astro',
    'Oligo': 'Oligo', 'Oligodendrocyte': 'Oligo',
    'OPC': 'OPC',
    # GABAergic
    'Lamp5': 'LAMP5', 'LAMP5': 'LAMP5',
    'Pvalb': 'PVALB', 'PVALB': 'PVALB',
    'Sst': 'SST',     'SST': 'SST',
    'Vip': 'VIP',     'VIP': 'VIP',
    # Glutamatergic IT — layer splits collapsed because SMART-seq can't resolve them
    'L2/3 IT': 'IT', 'L5 IT': 'IT', 'L6 IT': 'IT',
    'IT': 'IT',      'L4 IT': 'IT',
    # IT Car3 kept separate — both datasets resolve it
    'L6 IT Car3': 'IT Car3', 'L5/6 IT Car3': 'IT Car3',
    # Deep-layer projection neurons
    'L5/6 NP': 'L5/6 NP',
    'L6 CT': 'L6 CT',
    'L6b': 'L6b',
    # Intentionally absent (dropped as dataset-specific):
    #   10x only:      L5 ET, Sncg
    #   smartseq only: Microglia, PAX6
}


@dataclass
class CellSplit:
    class_names: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    sample_info: dict
    X: np.ndarray  # (n_cells, n_genes) float32, rows aligned to idx_*

    @property
    def n_classes(self) -> int:
        return len(self.class_names)
    


def _load_metadata(path: str) -> pd.DataFrame:
    meta = pd.read_csv(path, usecols=['sample_name', 'subclass_label'])
    meta = meta.dropna(subset=['subclass_label'])
    print(f'Metadata: {len(meta):,} labelled cells')
    return meta


def _canonicalize_labels(meta: pd.DataFrame) -> pd.DataFrame:
    mapped = meta['subclass_label'].map(CANONICAL_LABEL_MAP)
    if mapped.isna().any():
        dropped = meta.loc[mapped.isna(), 'subclass_label'].value_counts()
        print(f'Canonicalize: dropping {int(mapped.isna().sum()):,} cells '
              f'from non-shared labels: {dropped.to_dict()}')
    meta = meta.loc[mapped.notna()].copy()
    meta['subclass_label'] = mapped.dropna().values
    return meta.reset_index(drop=True)


def _filter_rare_classes(meta: pd.DataFrame, min_cells: int) -> pd.DataFrame:
    counts = meta['subclass_label'].value_counts()
    keep = counts[counts >= min_cells].index
    return meta[meta['subclass_label'].isin(keep)].reset_index(drop=True)


def _encode_labels(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    le = LabelEncoder()
    y = le.fit_transform(meta['subclass_label'].values)
    return y, le.classes_


def _split_indices(y: np.ndarray, seed: int):
    indices = np.arange(len(y))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y,
        test_size=VAL_FRAC + TEST_FRAC,
        stratify=y,
        random_state=seed,
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=y_temp,
        random_state=seed,
    )
    return idx_train, idx_val, idx_test, y_train, y_val, y_test


def _build_sample_info(
    meta: pd.DataFrame,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> dict:
    names = meta['sample_name'].values
    sample_info: dict = {}
    for split_name, idxs in (('train', idx_train), ('val', idx_val), ('test', idx_test)):
        for pos, i in enumerate(idxs):
            sample_info[names[i]] = (split_name, pos)
    return sample_info


def _print_split_sizes(idx_train, idx_val, idx_test, total: int) -> None:
    print(f'\nTrain: {len(idx_train):>6,} ({len(idx_train) / total * 100:.1f}%)')
    print(f'Val:   {len(idx_val):>6,} ({len(idx_val) / total * 100:.1f}%)')
    print(f'Test:  {len(idx_test):>6,} ({len(idx_test) / total * 100:.1f}%)')


def _load_matrix(csv_path: str, sample_names: np.ndarray) -> np.ndarray:
    """Load the expression matrix, aligned row-wise to ``sample_names``.

    On first call, reads the CSV once and caches two sibling ``.npy`` files
    next to it: ``<root>.npy`` (full (N, G) float32 matrix) and
    ``<root>_cells.npy`` (the sample_name order of its rows). Subsequent
    calls memmap the cache and fancy-index only the requested rows.
    """
    root, _ = os.path.splitext(csv_path)
    mat_path = root + '.npy'
    names_path = root + '_cells.npy'

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

        n_genes = len(gene_cols)
        mat = np.lib.format.open_memmap(
            mat_path, mode='w+', dtype=np.float32,
            shape=(df.height, n_genes),
        )
        step = max(1, n_genes // 20)
        for j, col in enumerate(gene_cols):
            mat[:, j] = df[col].to_numpy()
            if (j + 1) % step == 0 or (j + 1) == n_genes:
                print(f'  caching matrix: {(j + 1) / n_genes * 100:5.1f}% '
                      f'({j + 1:,}/{n_genes:,} genes)', flush=True)
        mat.flush()
        del df, mat
        gc.collect()

    X_full = np.load(mat_path, mmap_mode='r')
    names_full = np.load(names_path)

    row_of = {n: i for i, n in enumerate(names_full)}
    try:
        rows = np.fromiter(
            (row_of[n] for n in sample_names),
            dtype=np.int64,
            count=len(sample_names),
        )
    except KeyError as e:
        raise KeyError(f'sample {e} missing from matrix cache {mat_path}') from e

    X = np.ascontiguousarray(X_full[rows])
    print(f'Expression matrix: {X.shape}')
    return X



def _build_split(paths: dict, seed: int) -> CellSplit:
    meta = _load_metadata(paths['metadata'])
    meta = _canonicalize_labels(meta)
    meta = _filter_rare_classes(meta, MIN_CELLS_PER_CLASS)
    y_all, class_names = _encode_labels(meta)
    print(f'After class filter: {len(meta):,} cells, {len(class_names)} classes')

    idx_train, idx_val, idx_test, y_train, y_val, y_test = _split_indices(y_all, seed)
    sample_info = _build_sample_info(meta, idx_train, idx_val, idx_test)
    _print_split_sizes(idx_train, idx_val, idx_test, len(y_all))

    X = _load_matrix(paths['matrix'], meta['sample_name'].values)

    del meta
    gc.collect()

    return CellSplit(
        class_names=class_names,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        sample_info=sample_info,
        X=X,
    )


def get_10x_dataset(paths: dict = DEFAULT_10X_PATHS, seed: int = 42) -> CellSplit:
    return _build_split(paths, seed)


def get_smartseq_dataset(paths: dict = DEFAULT_SMARTSEQ_PATHS, seed: int = 42) -> CellSplit:
    return _build_split(paths, seed)
