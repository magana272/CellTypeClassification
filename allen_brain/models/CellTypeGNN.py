import os

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

console = Console()


def load_combined_xy(data_dir):
    """Load all splits into a single pre-allocated X buffer (no vstack copy)."""
    sizes = {}
    y_parts = {}
    for split in ('train', 'val', 'test'):
        y_parts[split] = np.load(os.path.join(data_dir, f'y_{split}.npy'))
        sizes[split] = len(y_parts[split])
    n_total = sum(sizes.values())
    X_train_mm = np.load(os.path.join(data_dir, 'X_train.npy'), mmap_mode='r')
    n_features = X_train_mm.shape[1]

    X_all = np.empty((n_total, n_features), dtype=np.float32)
    offset = 0
    for split in ('train', 'val', 'test'):
        X_mm = (X_train_mm if split == 'train'
                else np.load(os.path.join(data_dir, f'X_{split}.npy'), mmap_mode='r'))
        X_all[offset:offset + sizes[split]] = X_mm
        offset += sizes[split]
        del X_mm
    y_all = np.concatenate([y_parts['train'], y_parts['val'],
                            y_parts['test']]).astype(np.int64, copy=False)
    return X_all, y_all, sizes


def build_masks(sizes):
    """Create boolean train/val/test masks from split sizes."""
    n_total = sum(sizes.values())
    masks = {s: torch.zeros(n_total, dtype=torch.bool) for s in ('train', 'val', 'test')}
    offset = 0
    for s in ('train', 'val', 'test'):
        masks[s][offset:offset + sizes[s]] = True
        offset += sizes[s]
    return masks['train'], masks['val'], masks['test']

def _torch_knn(X_all, k, batch_size=256):
    X = torch.from_numpy(X_all).to('cuda')
    X = X / X.norm(dim=1, keepdim=True)
    indices = torch.empty(X.shape[0], k + 1, dtype=torch.long)
    for i in range(0, X.shape[0], batch_size):
        sim = X[i:i+batch_size] @ X.T
        indices[i:i+batch_size] = sim.topk(k + 1, dim=1).indices
    return indices.cpu().numpy()


def build_knn_edges(X_all, k):
    """Build symmetric k-NN edge index using cosine distance."""
    n_total = X_all.shape[0]
    console.print(f'Building k={k} cosine-NN graph on {n_total:,} cells...')
    indices = _torch_knn(X_all, k)
    if indices is None:
        from sklearn.neighbors import NearestNeighbors
        console.print('  (sklearn brute-force — install faiss-cpu for 10x speed)')
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine',
                                algorithm='brute').fit(X_all)
        _, indices = nbrs.kneighbors(X_all)
    src = np.repeat(np.arange(n_total), k)
    dst = indices[:, 1:].reshape(-1)
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    edge_index = torch.tensor(np.stack([src_sym, dst_sym], axis=0), dtype=torch.long)
    return torch.unique(edge_index, dim=1)


def build_graph_data(data_dir, k_neighbors=15, normalize=None):
    """Load data and build a PyG Data object with k-NN edges and split masks.

    normalize: None, 'log', 'standard', or 'log+standard'.
    StandardScaler is fit on train split only, then applied to all.
    """
    X_all, y_all, sizes = load_combined_xy(data_dir)
    tr_m, vl_m, te_m = build_masks(sizes)

    if normalize and normalize != 'none':
        n_train = sizes['train']
        if normalize in ('log', 'log+standard'):
            console.print('GNN: applying log normalization...')
            lib = X_all.sum(axis=1, keepdims=True)
            lib = np.maximum(lib, 1.0)
            X_all = np.log1p(X_all / lib * 1e4)
        if normalize in ('standard', 'log+standard'):
            from sklearn.preprocessing import StandardScaler
            console.print('GNN: applying StandardScaler (fit on train)...')
            scaler = StandardScaler()
            X_all[:n_train] = scaler.fit_transform(X_all[:n_train])
            X_all[n_train:] = scaler.transform(X_all[n_train:])

    edge_index = build_knn_edges(X_all, k_neighbors)
    n_total = X_all.shape[0]
    console.print(f'Graph: {n_total:,} nodes, {edge_index.shape[1]:,} edges, '
          f'avg deg {edge_index.shape[1] / n_total:.1f}')
    return Data(x=torch.from_numpy(X_all), edge_index=edge_index,
                y=torch.from_numpy(y_all),
                train_mask=tr_m, val_mask=vl_m, test_mask=te_m)


def build_eval_graph(X, y, k_neighbors=15):
    """Build a PyG Data object for evaluation (all nodes are test nodes)."""
    edge_index = build_knn_edges(X, k_neighbors)
    n = X.shape[0]
    return Data(x=torch.from_numpy(X), edge_index=edge_index,
                y=torch.from_numpy(y),
                test_mask=torch.ones(n, dtype=torch.bool))


def masked_class_weights(y, mask, n_classes, device=None):
    """Compute balanced class weights using only masked (training) nodes."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    counts = np.bincount(y[mask].cpu().numpy(), minlength=n_classes).astype(np.float32)
    w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    return w / w.sum() * n_classes


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class ResidualSAGEBlock(nn.Module):
    """SAGEConv with LayerNorm, GELU, and a residual projection."""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.conv    = SAGEConv(in_dim, out_dim)
        self.norm    = nn.LayerNorm(out_dim)
        self.dropout = dropout
        self.proj    = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h + self.proj(x)

    
class CellTypeGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, dropout=0.3, n_layers=2):
        super().__init__()
        # Linear bottleneck to reduce 50k gene features before graph conv
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        blocks = [ResidualSAGEBlock(hidden_dim, hidden_dim, dropout)]
        for _ in range(1, n_layers):
            blocks.append(ResidualSAGEBlock(hidden_dim, hidden_dim, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.conv_out = SAGEConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x, edge_index):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x, edge_index)
        x = F.gelu(self.conv_out(x, edge_index))
        return self.classifier(x)

    def embed(self, x, edge_index):
        """Pre-classifier embedding (for UMAP / transfer)."""
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x, edge_index)
        return F.gelu(self.conv_out(x, edge_index))
