"""Unknown cell type discovery using TOSICA."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TOSICA'))

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import TOSICA

from allen_brain.cell_data.cell_dataset import make_dataset, GeneExpressionDataset
from allen_brain.cell_data.cell_preprocess import select_hvg

DATA_DIR: str = 'data/10x'
GMT_PATH: str = 'data/reactome.gmt'
PROJECT: str = 'tosica_unknown'
LABEL_COL: str = 'cell_type'
N_HVG: int = 10_000
EPOCHS: int = 20
BATCH_SIZE: int = 64
UNKNOWN_THRESHOLD: float = 0.95


def main() -> None:
    ds_train: GeneExpressionDataset = make_dataset(DATA_DIR, split='train')
    ds_test: GeneExpressionDataset = make_dataset(DATA_DIR, split='test')

    if N_HVG and 0 < N_HVG < len(ds_train.gene_names):
        hvg_idx: np.ndarray = np.sort(select_hvg(np.asarray(ds_train.X), N_HVG))
        for ds in (ds_train, ds_test):
            ds.X = np.asarray(ds.X[:, hvg_idx])
            ds.gene_names = ds.gene_names[hvg_idx]

    gene_names: list[str] = [str(g) for g in ds_train.gene_names]
    all_class_names: list[str] = list(ds_train.class_names)
    n_classes: int = ds_train.n_classes

    counts: np.ndarray = np.bincount(np.asarray(ds_train.y), minlength=n_classes)
    ranked: np.ndarray = np.argsort(counts)[::-1]
    held_out_idx: int = next(
        (int(i) for i in ranked if 'IT' not in all_class_names[i].upper()),
        int(ranked[0]),
    )
    held_out_name: str = all_class_names[held_out_idx]
    print(f'Holding out: {held_out_name} ({counts[held_out_idx]} train cells)')

    mask: np.ndarray = ds_train.y != held_out_idx
    X_train: np.ndarray = np.asarray(ds_train.X)[mask].astype(np.float32)
    y_train: np.ndarray = np.asarray(ds_train.y)[mask]
    y_str: list[str] = [all_class_names[int(yi)] for yi in y_train]

    train_adata: ad.AnnData = ad.AnnData(X=X_train, var=pd.DataFrame(index=gene_names))
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
    X_test: np.ndarray = np.asarray(ds_test.X).astype(np.float32)
    y_test: np.ndarray = np.asarray(ds_test.y)

    test_adata: ad.AnnData = ad.AnnData(X=X_test, var=pd.DataFrame(index=gene_names))

    result: ad.AnnData = TOSICA.pre(
        test_adata,
        model_weight_path=f'./{PROJECT}/model-{EPOCHS - 1}.pth',
        project=PROJECT,
        cutoff=UNKNOWN_THRESHOLD,
        batch_size=BATCH_SIZE,
    )

    predictions: np.ndarray = result.obs['Prediction'].values.astype(str)
    probabilities: np.ndarray = result.obs['Probability'].values.astype(float)
    is_held: np.ndarray = y_test == held_out_idx
    unknown_mask: np.ndarray = predictions == 'Unknown'

    true_names: np.ndarray = np.array([all_class_names[int(yi)] for yi in y_test])
    known_true: np.ndarray = true_names[~is_held]
    known_pred: np.ndarray = predictions[~is_held]
    acc: float = (known_pred == known_true).mean()

    tp: int = int((is_held & unknown_mask).sum())
    fp: int = int((~is_held & unknown_mask).sum())
    fn: int = int((is_held & ~unknown_mask).sum())

    print(f'Known-class accuracy: {acc:.4f}')
    print(f'Unknown detection — TP={tp}  FP={fp}  FN={fn}')
    print(f'  Precision: {tp / max(tp + fp, 1):.4f}  '
          f'Recall: {tp / max(tp + fn, 1):.4f}')

    umap_adata: ad.AnnData = result.copy()
    n_comps: int = min(30, umap_adata.shape[1] - 1)
    sc.tl.pca(umap_adata, n_comps=n_comps)
    sc.pp.neighbors(umap_adata, n_neighbors=15, n_pcs=n_comps)
    sc.tl.umap(umap_adata)

    sc.pl.umap(umap_adata, color='Prediction', show=True)
    sc.pl.umap(umap_adata, color='Probability', show=True)


if __name__ == '__main__':
    main()
