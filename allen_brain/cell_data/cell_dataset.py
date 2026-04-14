import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .cell_preprocess import preprocess_hvg
from sklearn.preprocessing import LabelEncoder


def _as_float32_contig(X):
    arr = np.asarray(X)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr


def _as_int64_contig(y):
    arr = np.asarray(y)
    if arr.dtype != np.int64:
        arr = arr.astype(np.int64, copy=False)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr


class GeneExpressionDataset(Dataset):
    """Wraps an (N, G) expression matrix and integer labels as a PyTorch Dataset."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, labelencoder: LabelEncoder=None, split=None, gene_names=None, class_names=None):

        self.X = torch.from_numpy(_as_float32_contig(X))
        self.y = torch.from_numpy(_as_int64_contig(y))
        self.split = split
        self.labelencoder: LabelEncoder = labelencoder
        self.n_classes = len(labelencoder.classes_) if labelencoder else int(self.y.max()) + 1
        self.class_names = class_names if class_names is not None else (labelencoder.classes_ if labelencoder else np.array([str(i) for i in range(self.n_classes)]))
        self.gene_names = gene_names

    def to(self, device):
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        return self

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]
        
    def get_y_labels(self):
        if self.labelencoder:
            return list(self.labelencoder.inverse_transform(self.y.cpu().numpy()))
        else:
            return list(self.y.cpu().numpy())
    
    
def load_label_encoder(le_path)-> [LabelEncoder|None]:
    if os.path.exists(le_path):
        import pickle
        with open(le_path, 'rb') as f:
            return pickle.load(f)
    return None


def make_split_dataset(data_dir: str, split='train') -> GeneExpressionDataset:
    """Load raw .npy splits, preprocess, return datasets + metadata.

    Returns a dict with keys:
        'train'                 — GeneExpressionDataset instance
        'class_names'           — np.ndarray of class label strings
        'gene_names'            — np.ndarray of HVG gene name strings
        'scaler'                — fitted StandardScaler
        'n_classes'             — int
        
    """
    
    
    X = np.load(os.path.join(data_dir, f'X_{split}.npy'))
    y = np.load(os.path.join(data_dir, f'y_{split}.npy'))
    gene_names = np.load(os.path.join(data_dir, 'gene_names.npy'))
    class_names = np.load(os.path.join(data_dir, 'class_names.npy'), allow_pickle=True)
        
    le_path = os.path.join(data_dir, 'label_encoder.pkl')
    labelencoder = load_label_encoder(le_path)

    return GeneExpressionDataset(X, y, labelencoder=labelencoder, split=split, gene_names=gene_names, class_names=class_names)


def make_dataset(data_dir: str, split='train', min_gene_frac: float = 0.01) -> GeneExpressionDataset:
    """Load raw .npy splits, preprocess, return datasets + metadata.

    Returns a dict with keys:
        'train', 'val', 'test'  — GeneExpressionDataset instances
        'split'                 — which split was loaded ('train', 'val', 'test', or 'all')
        'class_names'           — np.ndarray of class label strings
        'gene_names'            — np.ndarray of HVG gene name strings
        'scaler'                — fitted StandardScaler
        'n_classes'             — int
        
    """
    ds_val, ds_test, ds_train = None, None, None
    
    if split not in ('train', 'val', 'test'):
        raise ValueError(f"Invalid split '{split}', expected 'train', 'val', 'test'")
    if split == 'test':
        return make_split_dataset(data_dir, split='test')
    elif split == 'val':
        return make_split_dataset(data_dir, split='val')
    else: 
        return make_split_dataset(data_dir, split='train')
    
    