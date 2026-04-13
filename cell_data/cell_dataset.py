import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import preprocess_hvg


class GeneExpressionDataset(Dataset):
    """Wraps an (N, G) expression matrix and integer labels as a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_datasets(data_dir: str, n_hvg: int = 2000, min_gene_frac: float = 0.01) -> dict:
    """Load raw .npy splits, preprocess, return datasets + metadata.

    Returns a dict with keys:
        'train', 'val', 'test'  — GeneExpressionDataset instances
        'class_names'           — np.ndarray of class label strings
        'gene_names'            — np.ndarray of HVG gene name strings
        'scaler'                — fitted StandardScaler
        'n_classes'             — int
    """
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'), mmap_mode='r')
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'), mmap_mode='r')
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'), mmap_mode='r')
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    gene_names = np.load(os.path.join(data_dir, 'gene_names.npy'))
    class_names = np.load(os.path.join(data_dir, 'class_names.npy'), allow_pickle=True)

    X_train, X_val, X_test, hvg_gene_names, scaler = preprocess_hvg(
        X_train, X_val, X_test, gene_names,
        n_hvg=n_hvg, min_gene_frac=min_gene_frac,
    )

    return {
        'train': GeneExpressionDataset(X_train, y_train),
        'val': GeneExpressionDataset(X_val, y_val),
        'test': GeneExpressionDataset(X_test, y_test),
        'class_names': class_names,
        'gene_names': hvg_gene_names,
        'scaler': scaler,
        'n_classes': len(class_names),
    }
