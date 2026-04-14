import os

import numpy as np
import torch
from torch import nn, optim

from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from allen_brain.models import train as T

torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    import faiss

SEED = 42
BATCH_SIZE = 4096
N_HVG = 2000
DATA_DIR = 'data/smartseq'
K_NEIGHBORS = 15
N_TRIALS = 15
TUNE_EPOCHS = 30

COFIG = {
    'model': 'CellTypeGNN',
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'n_hvg': N_HVG,
    'device': str(T.DEVICE),
    'optimizer': optim.AdamW,
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'epochs': 200,
    'loss': nn.CrossEntropyLoss,
}


def _load_combined_xy():
    """Load all splits into a single pre-allocated X buffer (no vstack copy)."""
    sizes = {}
    y_parts = {}
    for split in ('train', 'val', 'test'):
        y_parts[split] = np.load(os.path.join(DATA_DIR, f'y_{split}.npy'))
        sizes[split] = len(y_parts[split])
    n_total = sum(sizes.values())
    X_train_mm = np.load(os.path.join(DATA_DIR, 'X_train.npy'), mmap_mode='r')
    n_features = X_train_mm.shape[1]

    X_all = np.empty((n_total, n_features), dtype=np.float32)
    offset = 0
    for split in ('train', 'val', 'test'):
        X_mm = (X_train_mm if split == 'train'
                else np.load(os.path.join(DATA_DIR, f'X_{split}.npy'), mmap_mode='r'))
        X_all[offset:offset + sizes[split]] = X_mm
        offset += sizes[split]
        del X_mm
    y_all = np.concatenate([y_parts['train'], y_parts['val'], y_parts['test']]).astype(np.int64, copy=False)
    return X_all, y_all, sizes


def _build_masks(sizes):
    n_total = sum(sizes.values())
    masks = {s: torch.zeros(n_total, dtype=torch.bool) for s in ('train', 'val', 'test')}
    offset = 0
    for s in ('train', 'val', 'test'):
        masks[s][offset:offset + sizes[s]] = True
        offset += sizes[s]
    return masks['train'], masks['val'], masks['test']


def _build_knn_edges(X_all, k):
    n_total = X_all.shape[0]
    print(f'Building k={k} cosine-NN graph on {n_total:,} cells...')
    if torch.cuda.is_available():
        X_norm = X_all / np.linalg.norm(X_all, axis=1, keepdims=True)
        X_norm = X_norm.astype(np.float32)
        index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), X_norm.shape[1])
        index.add(X_norm)
        _, indices = index.search(X_norm, k + 1)
    else:
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute').fit(X_all)
        _, indices = nbrs.kneighbors(X_all)
    src = np.repeat(np.arange(n_total), k)
    dst = indices[:, 1:].reshape(-1)
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    edge_index = torch.tensor(np.stack([src_sym, dst_sym], axis=0), dtype=torch.long)
    return torch.unique(edge_index, dim=1)


def data_set_up():
    X_all, y_all, sizes = _load_combined_xy()
    tr_m, vl_m, te_m = _build_masks(sizes)
    edge_index = _build_knn_edges(X_all, K_NEIGHBORS)
    n_total = X_all.shape[0]
    print(f'Graph: {n_total:,} nodes, {edge_index.shape[1]:,} edges, avg deg {edge_index.shape[1] / n_total:.1f}')
    return Data(x=torch.from_numpy(X_all), edge_index=edge_index,
                y=torch.from_numpy(y_all),
                train_mask=tr_m, val_mask=vl_m, test_mask=te_m)


def _masked_class_weights(y, mask, n_classes, device=T.DEVICE):
    counts = np.bincount(y[mask].cpu().numpy(), minlength=n_classes).astype(np.float32)
    w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    return w / w.sum() * n_classes


def main():
    data = data_set_up().to(T.DEVICE)
    n_classes = int(data.y.max().item()) + 1
    weights = _masked_class_weights(data.y, data.train_mask, n_classes)
    T.train_graph_with_tuning(COFIG, data, data.x.shape[1], n_classes, weights,
                              n_trials=N_TRIALS, tune_epochs=TUNE_EPOCHS)


if __name__ == '__main__':
    main()
