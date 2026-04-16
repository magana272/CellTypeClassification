"""Unknown cell type discovery using TOSICA."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TOSICA'))

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import TOSICA

from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.cell_data.cell_preprocess import select_hvg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = 'data/10x'
GMT_PATH = 'data/reactome.gmt'
PROJECT = 'tosica_unknown'
LABEL_COL = 'cell_type'
N_HVG = 10_000
EPOCHS = 20
BATCH_SIZE = 64
UNKNOWN_THRESHOLD = 0.95


def main():
    ds_train = make_dataset(DATA_DIR, split='train')
    ds_test  = make_dataset(DATA_DIR, split='test')

    if N_HVG and 0 < N_HVG < len(ds_train.gene_names):
        hvg_idx = np.sort(select_hvg(np.asarray(ds_train.X), N_HVG))
        for ds in (ds_train, ds_test):
            ds.X = np.asarray(ds.X[:, hvg_idx])
            ds.gene_names = ds.gene_names[hvg_idx]

    gene_names = [str(g) for g in ds_train.gene_names]
    all_class_names = list(ds_train.class_names)
    n_classes = ds_train.n_classes

    counts = np.bincount(np.asarray(ds_train.y), minlength=n_classes)
    ranked = np.argsort(counts)[::-1]
    held_out_idx = next(
        (int(i) for i in ranked if 'IT' not in all_class_names[i].upper()),
        int(ranked[0]),
    )
    held_out_name = all_class_names[held_out_idx]
    print(f'Holding out: {held_out_name} ({counts[held_out_idx]} train cells)')

    mask = ds_train.y != held_out_idx
    X_train = np.asarray(ds_train.X)[mask].astype(np.float32)
    y_train = np.asarray(ds_train.y)[mask]
    y_str = [all_class_names[int(yi)] for yi in y_train]

    train_adata = ad.AnnData(X=X_train, var=pd.DataFrame(index=gene_names))
    train_adata.obs[LABEL_COL] = y_str

    TOSICA.train(
        train_adata,
        gmt_path=GMT_PATH,
        project=PROJECT,
        label_name=LABEL_COL,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_gs=300,
        max_g=300,
    )

    # Predict on full test set
    X_test = np.asarray(ds_test.X).astype(np.float32)
    y_test = np.asarray(ds_test.y)

    test_adata = ad.AnnData(X=X_test, var=pd.DataFrame(index=gene_names))

    result = TOSICA.pre(
        test_adata,
        model_weight_path=f'./{PROJECT}/model-{EPOCHS - 1}.pth',
        project=PROJECT,
        cutoff=UNKNOWN_THRESHOLD,
        batch_size=BATCH_SIZE,
    )

    predictions = result.obs['Prediction'].values.astype(str)
    probabilities = result.obs['Probability'].values.astype(float)
    is_held = y_test == held_out_idx
    unknown_mask = predictions == 'Unknown'

    true_names = np.array([all_class_names[int(yi)] for yi in y_test])
    known_true = true_names[~is_held]
    known_pred = predictions[~is_held]
    acc = (known_pred == known_true).mean()

    tp = int((is_held & unknown_mask).sum())
    fp = int((~is_held & unknown_mask).sum())
    fn = int((is_held & ~unknown_mask).sum())

    print(f'Known-class accuracy: {acc:.4f}')
    print(f'Unknown detection — TP={tp}  FP={fp}  FN={fn}')
    print(f'  Precision: {tp / max(tp + fp, 1):.4f}  '
          f'Recall: {tp / max(tp + fn, 1):.4f}')

    umap_adata = result.copy()
    n_comps = min(30, umap_adata.shape[1] - 1)
    sc.tl.pca(umap_adata, n_comps=n_comps)
    sc.pp.neighbors(umap_adata, n_neighbors=15, n_pcs=n_comps)
    sc.tl.umap(umap_adata)

    sc.pl.umap(umap_adata, color='Prediction', show=False)
    sc.pl.umap(umap_adata, color='Probability', show=False)


if __name__ == '__main__':
    main()
