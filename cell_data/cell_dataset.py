import numpy as np
import torch
from torch.utils.data import Dataset


class GeneExpressionDataset(Dataset):
    """Single-cell RNA-seq dataset holding a dense (N, G) expression matrix
    and integer class labels.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None, target_transform=None):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        expression = self.X[idx]
        label = self.y[idx]
        if self.transform:
            expression = self.transform(expression)
        if self.target_transform:
            label = self.target_transform(label)
        return expression, label
