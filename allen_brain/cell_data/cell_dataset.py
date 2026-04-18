from __future__ import annotations

import os

import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset

from .cell_preprocess import preprocess_hvg
from sklearn.preprocessing import LabelEncoder


class GeneExpressionDataset(Dataset):
    """Lazy row-wise view over an (N, G) expression matrix.

    Supports both dense (.npy, memory-mapped) and sparse (.npz, CSR) storage.
    ``__getitem__`` always returns a dense tensor regardless of backing format.
    """

    X: np.ndarray | scipy.sparse.csr_matrix

    def __init__(
        self,
        X_path: str,
        y_path: str,
        labelencoder: LabelEncoder | None = None,
        split: str | None = None,
        gene_names: np.ndarray | None = None,
        class_names: np.ndarray | None = None,
    ) -> None:
        if X_path.endswith('.npz'):
            self.X = scipy.sparse.load_npz(X_path)
            self._sparse = True
        else:
            self.X = np.load(X_path, mmap_mode='r')
            self._sparse = False

        self.y: np.ndarray = np.load(y_path, mmap_mode='r')
        self.split = split
        self.labelencoder = labelencoder
        self.n_classes: int = (
            len(labelencoder.classes_) if labelencoder
            else int(self.y.max()) + 1
        )
        self.class_names: np.ndarray = (
            class_names if class_names is not None
            else (labelencoder.classes_ if labelencoder
                  else np.array([str(i) for i in range(self.n_classes)]))
        )
        self.gene_names: np.ndarray | None = gene_names

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self._sparse:
            row = np.asarray(self.X[idx].todense(), dtype=np.float32).squeeze()
        else:
            row = np.array(self.X[idx], dtype=np.float32, copy=True)
        return torch.from_numpy(row).unsqueeze(0), int(self.y[idx])

    def get_y_labels(self) -> list[str]:
        y_arr = self.y if isinstance(self.y, np.ndarray) else self.y.cpu().numpy()
        if self.labelencoder:
            return list(self.labelencoder.inverse_transform(y_arr))
        return [str(v) for v in y_arr]


def load_label_encoder(le_path: str) -> LabelEncoder | None:
    if os.path.exists(le_path):
        import pickle
        with open(le_path, 'rb') as f:
            return pickle.load(f)
    return None


def make_split_dataset(data_dir: str, split: str = 'train') -> GeneExpressionDataset:
    """Load .npy or .npz splits and return a GeneExpressionDataset."""
    # Prefer sparse .npz if available, fall back to dense .npy
    npz_path = os.path.join(data_dir, f'X_{split}.npz')
    npy_path = os.path.join(data_dir, f'X_{split}.npy')
    X_path = npz_path if os.path.exists(npz_path) else npy_path

    y_path = os.path.join(data_dir, f'y_{split}.npy')
    gene_names = np.load(os.path.join(data_dir, 'gene_names.npy'))
    class_names = np.load(os.path.join(data_dir, 'class_names.npy'), allow_pickle=True)

    le_path = os.path.join(data_dir, 'label_encoder.pkl')
    labelencoder = load_label_encoder(le_path)

    return GeneExpressionDataset(
        X_path=X_path, y_path=y_path, labelencoder=labelencoder,
        split=split, gene_names=gene_names, class_names=class_names,
    )


def make_dataset(data_dir: str, split: str = 'train') -> GeneExpressionDataset:
    """Load a specific split and return a GeneExpressionDataset."""
    if split not in ('train', 'val', 'test'):
        raise ValueError(f"Invalid split '{split}', expected 'train', 'val', 'test'")
    return make_split_dataset(data_dir, split=split)
