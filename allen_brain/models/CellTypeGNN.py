import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

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


def _faiss_knn(X_all, k):
    """Try FAISS (GPU then CPU) for fast cosine k-NN. Returns indices or None."""
    try:
        import faiss
    except ImportError:
        return None
    X_norm = (X_all / np.linalg.norm(X_all, axis=1, keepdims=True)).astype(np.float32)
    # Try GPU first
    if torch.cuda.is_available():
        try:
            index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), X_norm.shape[1])
            index.add(X_norm)
            _, indices = index.search(X_norm, k + 1)
            print('  (FAISS GPU)')
            return indices
        except Exception as e:
            print(f'  FAISS GPU failed ({e}), trying CPU...')
    # CPU fallback
    index = faiss.IndexFlatIP(X_norm.shape[1])
    index.add(X_norm)
    _, indices = index.search(X_norm, k + 1)
    print('  (FAISS CPU)')
    return indices


def build_knn_edges(X_all, k):
    """Build symmetric k-NN edge index using cosine distance."""
    n_total = X_all.shape[0]
    print(f'Building k={k} cosine-NN graph on {n_total:,} cells...')
    indices = _faiss_knn(X_all, k)
    if indices is None:
        from sklearn.neighbors import NearestNeighbors
        print('  (sklearn brute-force — install faiss-cpu for 10x speed)')
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine',
                                algorithm='brute').fit(X_all)
        _, indices = nbrs.kneighbors(X_all)
    src = np.repeat(np.arange(n_total), k)
    dst = indices[:, 1:].reshape(-1)
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    edge_index = torch.tensor(np.stack([src_sym, dst_sym], axis=0), dtype=torch.long)
    return torch.unique(edge_index, dim=1)


def build_graph_data(data_dir, k_neighbors=15):
    """Load data and build a PyG Data object with k-NN edges and split masks."""
    X_all, y_all, sizes = load_combined_xy(data_dir)
    tr_m, vl_m, te_m = build_masks(sizes)
    edge_index = build_knn_edges(X_all, k_neighbors)
    n_total = X_all.shape[0]
    print(f'Graph: {n_total:,} nodes, {edge_index.shape[1]:,} edges, '
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
        blocks = [ResidualSAGEBlock(in_dim, hidden_dim, dropout)]
        for _ in range(1, n_layers):
            blocks.append(ResidualSAGEBlock(hidden_dim, hidden_dim, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.conv_out = SAGEConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x, edge_index):
        for block in self.blocks:
            x = block(x, edge_index)
        x = F.gelu(self.conv_out(x, edge_index))
        return self.classifier(x)

    def embed(self, x, edge_index):
        """Pre-classifier embedding (for UMAP / transfer)."""
        for block in self.blocks:
            x = block(x, edge_index)
        return F.gelu(self.conv_out(x, edge_index))
