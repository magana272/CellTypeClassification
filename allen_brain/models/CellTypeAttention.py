import math
import os

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

console = Console()




_DEFAULT_GMT_URL = ('https://data.broadinstitute.org/gsea-msigdb/msigdb/'
                    'release/2023.2.Hs/c2.cp.reactome.v2023.2.Hs.symbols.gmt')


def _parse_gmt(path, max_gene_set_size=300):
    gmt = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            genes = [g for g in parts[2:] if g]
            if len(genes) <= max_gene_set_size:
                gmt[parts[0]] = genes
    return gmt


def _download_gmt(path, url):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if os.path.exists(path):
        return True
    try:
        console.print(f'Downloading Reactome GMT to {path}...')
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        return True
    except Exception as e:
        console.print(f'[yellow]GMT download failed[/yellow] ({e}); falling back to identity mask.')
        return False


def _select_pathways(gmt, gene_set, min_overlap, max_pathways):
    kept = []
    for name, genes in gmt.items():
        overlap = [g for g in genes if g in gene_set]
        if len(overlap) >= min_overlap:
            kept.append((name, overlap))
    kept.sort(key=lambda x: len(x[1]), reverse=True)
    return kept[:max_pathways]


def _pathways_to_mask(kept, gene_names):
    gene_to_row = {g: i for i, g in enumerate(gene_names)}
    mask = np.zeros((len(gene_names), len(kept)), dtype=np.float32)
    for j, (_, genes) in enumerate(kept):
        for g in genes:
            i = gene_to_row.get(g)
            if i is not None:
                mask[i, j] = 1.0
    return torch.from_numpy(mask)


def build_pathway_mask(gene_names, gmt_path='data/reactome.gmt',
                       gmt_url=_DEFAULT_GMT_URL, min_overlap=5,
                       max_pathways=300, max_gene_set_size=300):
    """Build a (n_genes, n_pathways) binary mask from Reactome pathways."""
    if not _download_gmt(gmt_path, gmt_url):
        return torch.eye(len(gene_names)), len(gene_names)
    gmt = _parse_gmt(gmt_path, max_gene_set_size)
    console.print(f'Gene sets loaded: {len(gmt):,}')
    kept = _select_pathways(gmt, set(gene_names), min_overlap, max_pathways)
    if not kept:
        console.print('[yellow]No pathways matched; using identity mask.[/yellow]')
        return torch.eye(len(gene_names)), len(gene_names)
    mask = _pathways_to_mask(kept, gene_names)
    console.print(f'Mask: {tuple(mask.shape)}, sparsity {1 - mask.mean().item():.2%}')
    return mask, len(kept)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class MaskedEmbedding(nn.Module):
    def __init__(self, n_genes: int, n_pathways: int, embed_dim: int, mask: torch.Tensor):
        super().__init__()
        self.n_genes     = n_genes
        self.n_pathways  = n_pathways
        self.embed_dim   = embed_dim
        self.register_buffer('mask', mask)
        self.weight = nn.Parameter(torch.randn(embed_dim, n_genes, n_pathways) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(embed_dim, n_pathways))

    def forward(self, x):
        W = self.weight * self.mask.unsqueeze(0)
        out = torch.einsum('bg,egp->bep', x, W) + self.bias
        return out


class TOSICA(nn.Module):

    def __init__(self, n_genes, n_pathways, n_classes, mask,
                 embed_dim=48, n_heads=4, n_layers=2, dropout=0.1,
                 unknown_threshold=0.95):
        super().__init__()
        self.n_pathways        = n_pathways
        self.embed_dim         = embed_dim
        self.unknown_threshold = unknown_threshold

        self.embedding = MaskedEmbedding(n_genes, n_pathways, embed_dim, mask)

        self.cls_token = nn.Parameter(torch.randn(1, embed_dim, 1))

        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, n_pathways + 1) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self._attn_weights = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, n_classes),
        )

    def forward(self, x, return_attention=False):

        batch = x.size(0)

        tokens = self.embedding(x)

        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=2)

        tokens = tokens + self.pos_embed

        tokens = tokens.permute(0, 2, 1)

        out = self.transformer(tokens)

        cls_out = out[:, 0, :]

        logits = self.classifier(cls_out)

        if return_attention:

            pathway_out = out[:, 1:, :]
            cls_rep     = cls_out.unsqueeze(1)
            attn_scores = torch.bmm(cls_rep, pathway_out.transpose(1, 2))
            attn_scores = F.softmax(attn_scores.squeeze(1) / math.sqrt(self.embed_dim), dim=-1)
            return logits, attn_scores

        return logits

    def predict_with_unknown(self, x, threshold=None):
        if threshold is None:
            threshold = self.unknown_threshold
        logits = self.forward(x)
        probs  = F.softmax(logits, dim=-1)
        max_p, preds = probs.max(dim=-1)
        preds[max_p < threshold] = -1  
        return preds, max_p
