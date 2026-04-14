
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
torch.set_float32_matmul_precision('high')
 
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
    def __init__(self, in_dim, hidden_dim, n_classes, dropout=0.3):
        super().__init__()
        self.block1    = ResidualSAGEBlock(in_dim,     hidden_dim, dropout)
        self.block2    = ResidualSAGEBlock(hidden_dim, hidden_dim, dropout)
        self.conv_out  = SAGEConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = F.gelu(self.conv_out(x, edge_index))
        return self.classifier(x)

    def embed(self, x, edge_index):
        """Pre-classifier embedding (for UMAP / transfer)."""
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        return F.gelu(self.conv_out(x, edge_index))
