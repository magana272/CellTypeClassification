import os
import requests

import numpy as np
import torch
from torch import nn, optim

from allen_brain.models import train as T

torch.set_float32_matmul_precision('high')
SEED = 42
BATCH_SIZE = 4096
N_HVG = 2000

DATA_DIR = 'data/smartseq'
GMT_PATH = 'data/reactome.gmt'
GMT_URL  = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/c2.cp.reactome.v2023.2.Hs.symbols.gmt'
MAX_PATHWAYS = 300
MIN_PATHWAY_OVERLAP = 5
MAX_GENE_SET_SIZE = 300
N_TRIALS = 10
TUNE_EPOCHS = 5

COFIG = {
    'model': 'CellTypeTOSICA',
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'n_hvg': N_HVG,
    'device': str(T.DEVICE),
    'optimizer': optim.AdamW,
    'lr': 3e-4,
    'weight_decay': 1e-6,
    'epochs': 20,
    'loss': nn.CrossEntropyLoss,
}


def _parse_gmt(path, max_gene_set_size=MAX_GENE_SET_SIZE):
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
        print(f'Downloading Reactome GMT to {path}...')
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f'GMT download failed ({e}); falling back to identity mask.')
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


def build_pathway_mask(gene_names):
    if not _download_gmt(GMT_PATH, GMT_URL):
        return torch.eye(len(gene_names)), len(gene_names)
    gmt = _parse_gmt(GMT_PATH)
    print(f'Gene sets loaded: {len(gmt):,}')
    kept = _select_pathways(gmt, set(gene_names), MIN_PATHWAY_OVERLAP, MAX_PATHWAYS)
    if not kept:
        print('No pathways matched; using identity mask.')
        return torch.eye(len(gene_names)), len(gene_names)
    mask = _pathways_to_mask(kept, gene_names)
    print(f'Mask: {tuple(mask.shape)}, sparsity {1 - mask.mean().item():.2%}')
    return mask, len(kept)


def _make_builder(ds):
    mask, n_pathways = build_pathway_mask([str(g) for g in ds.gene_names])
    return lambda: T.build_model(COFIG['model'], len(ds.gene_names), ds.n_classes,
                                 mask=mask, n_pathways=n_pathways)


def main():
    T.train_with_tuning(COFIG, DATA_DIR, squeeze_channel=True,
                        make_builder=_make_builder,
                        n_trials=N_TRIALS, tune_epochs=TUNE_EPOCHS)


if __name__ == '__main__':
    main()
