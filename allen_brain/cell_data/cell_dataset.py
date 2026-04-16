import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .cell_preprocess import preprocess_hvg
from sklearn.preprocessing import LabelEncoder



class GeneExpressionDataset(Dataset):
    """Lazy row-wise view over an (N, G) expression matrix stored as .npy.

    X is memory-mapped (mmap_mode='r'), so the full matrix never lives in
    RAM; __getitem__ copies one row at a time. DataLoader workers inherit
    the mmap via the OS page cache without per-worker copies.
    """

    def __init__(self,
                 X_path: str,
                 y_path: str,
                 labelencoder: LabelEncoder=None,
                 split=None,
                 gene_names=None,
                 class_names=None):

        self.X : np.ndarray = np.load(X_path, mmap_mode='r')
        self.y : np.ndarray = np.load(y_path, mmap_mode='r')
        self.split: str = split
        self.labelencoder: LabelEncoder = labelencoder
        self.n_classes: int = len(labelencoder.classes_) if labelencoder else int(self.y.max()) + 1
        self.class_names: np.ndarray = class_names if class_names is not None else (labelencoder.classes_ if labelencoder else np.array([str(i) for i in range(self.n_classes)]))
        self.gene_names: np.ndarray = gene_names


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = np.array(self.X[idx], dtype=np.float32, copy=True)
        return torch.from_numpy(row).unsqueeze(0), self.y[idx]
        
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
    
    
    X_path = os.path.join(data_dir, f'X_{split}.npy')
    y_path = os.path.join(data_dir, f'y_{split}.npy')
    gene_names = np.load(os.path.join(data_dir, 'gene_names.npy'))
    class_names = np.load(os.path.join(data_dir, 'class_names.npy'), allow_pickle=True)
        
    le_path = os.path.join(data_dir, 'label_encoder.pkl')
    labelencoder = load_label_encoder(le_path)

    return GeneExpressionDataset(X_path=X_path, y_path=y_path, labelencoder=labelencoder, split=split, gene_names=gene_names, class_names=class_names)


def make_dataset(data_dir: str, split='train') -> GeneExpressionDataset:
    """Load raw .npy splits, preprocess, return datasets + metadata.

    Returns a dict with keys:
        'train', 'val', 'test'  — GeneExpressionDataset instances
        'split'                 — which split was loaded ('train', 'val', 'test', or 'all')
        'class_names'           — np.ndarray of class label strings
        'gene_names'            — np.ndarray of HVG gene name strings
        'scaler'                — fitted StandardScaler
        'n_classes'             — int
        
    """    
    if split not in ('train', 'val', 'test'):
        raise ValueError(f"Invalid split '{split}', expected 'train', 'val', 'test'")
    if split == 'test':
        return make_split_dataset(data_dir, split='test')
    elif split == 'val':
        return make_split_dataset(data_dir, split='val')
    else: 
        return make_split_dataset(data_dir, split='train')
    
    