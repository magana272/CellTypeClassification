from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA

import cell_data.cell_dataset as cell_dataset


def _to_numpy(a):
    if hasattr(a, 'detach'):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def _class_names(ds: cell_dataset.GeneExpressionDataset):
    return np.asarray(ds.labelencoder.classes_)


def _string_labels(ds: cell_dataset.GeneExpressionDataset):
    return _class_names(ds)[_to_numpy(ds.y)]


def plot_class_distribution(ds: cell_dataset.GeneExpressionDataset):
    label_counter = Counter(ds.y.tolist())
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(label_counter)), label_counter.values(),
                color=sns.color_palette('tab20', len(label_counter)))
    ax.set_xticks(range(len(label_counter)))
    ax.set_xticklabels(label_counter.keys(), rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Training Set — Cell Type Distribution')
    for bar, count in zip(bars, label_counter.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig('fig_class_distribution.png', dpi=150)
    plt.show()
    print('Saved fig_class_distribution.png')


def plot_pca(ds: cell_dataset.GeneExpressionDataset,
             seed=42, n_components=50, save_path='fig_pca.png'):
    """Run PCA on ds.X and plot scree + PC1 vs PC2 colored by class.

    Returns
    -------
    pca : fitted sklearn PCA
    X_pca : (n_samples, n_components) transformed array
    """
    X = _to_numpy(ds.X)
    y = _to_numpy(ds.y)
    class_names = _class_names(ds)

    print('Running PCA on training set...')
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X)
    explained_var = np.cumsum(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(1, n_components + 1), pca.explained_variance_ratio_,
                 'o-', ms=3, label='Per-PC')
    axes[0].plot(range(1, n_components + 1), explained_var, 's-', ms=3, label='Cumulative')
    axes[0].axhline(0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('PCA — Scree Plot (Training Set)')
    axes[0].legend(fontsize=8)

    palette = sns.color_palette('tab20', len(class_names))
    for i, cls in enumerate(class_names):
        mask = y == i
        if not mask.any():
            continue
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=[palette[i]], s=2, alpha=0.5, label=cls)

    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title('PCA — PC1 vs PC2 (Training Set)')
    axes[1].legend(markerscale=4, fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved {save_path}')
    return pca, X_pca


def plot_umap(ds: cell_dataset.GeneExpressionDataset, X_pca,
              max_cells=1000, seed=42, n_neighbors=30, min_dist=0.3,
              save_path='fig_umap.png'):
    """Fit UMAP on X_pca (subsampled if large) and scatter colored by class."""
    X_pca = _to_numpy(X_pca)
    labels = _string_labels(ds)
    class_names = _class_names(ds)

    rng = np.random.default_rng(seed)
    if len(X_pca) > max_cells:
        umap_idx = rng.choice(len(X_pca), max_cells, replace=False)
        X_input = X_pca[umap_idx]
        umap_labels = labels[umap_idx]
        print(f'  Subsampled to {max_cells:,} cells for UMAP speed.')
    else:
        X_input = X_pca
        umap_labels = labels

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed, verbose=True)
    X_umap = reducer.fit_transform(X_input)

    palette = sns.color_palette('tab20', len(class_names))
    color_map = {cls: palette[i] for i, cls in enumerate(class_names)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in class_names:
        mask = umap_labels == cls
        if not mask.any():
            continue
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=[color_map[cls]], s=2, alpha=0.6, label=cls)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP of Training Set — Colored by Cell Type')
    ax.legend(markerscale=5, fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved {save_path}')
    return X_umap


def get_top_hvg_genes(X, gene_names, top_n=20):
    X = _to_numpy(X)
    gene_names = np.asarray(gene_names)
    gene_variances = np.var(X, axis=0)
    top_indices = np.argsort(gene_variances)[-top_n:][::-1]
    return gene_names[top_indices], top_indices


def plot_heatmap(ds: cell_dataset.GeneExpressionDataset, gene_names,
                 n_genes=10, n_cells_per_type=50, seed=42,
                 save_path='fig_heatmap.png'):
    """Clustermap of top-variance HVGs for a balanced sample of cells per class."""
    X = _to_numpy(ds.X)
    labels = _string_labels(ds)
    class_names = _class_names(ds)
    gene_names = np.asarray(gene_names)

    top_gene_names, top_indices = get_top_hvg_genes(X, gene_names, top_n=n_genes)

    rng = np.random.default_rng(seed)
    sampled_idx, sampled_types = [], []
    for cls in class_names:
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            continue
        chosen = rng.choice(cls_idx, size=min(n_cells_per_type, len(cls_idx)), replace=False)
        sampled_idx.extend(chosen)
        sampled_types.extend([cls] * len(chosen))

    sampled_idx = np.array(sampled_idx)
    sampled_types = np.array(sampled_types)
    sort_order = np.argsort(sampled_types)
    sampled_idx = sampled_idx[sort_order]
    sampled_types = sampled_types[sort_order]

    heat_data = X[sampled_idx][:, top_indices]

    palette = sns.color_palette('tab20', len(class_names))
    color_map = {cls: palette[i] for i, cls in enumerate(class_names)}
    row_colors = pd.Series(sampled_types, name='Cell Type').map(color_map)

    g = sns.clustermap(
        pd.DataFrame(heat_data, columns=top_gene_names),
        row_colors=row_colors,
        col_cluster=True, row_cluster=False,
        cmap='RdBu_r', center=0, vmin=-3, vmax=3,
        xticklabels=True, yticklabels=False,
        figsize=(14, 8),
    )
    g.ax_heatmap.set_title(f'Top {n_genes} HVGs — Training Set Sample', pad=20)
    g.ax_heatmap.set_xlabel('Gene')
    handles = [mpatches.Patch(color=color_map[cls], label=cls) for cls in class_names]
    g.ax_col_dendrogram.legend(handles=handles, fontsize=7,
                               bbox_to_anchor=(1.15, 1), loc='upper left')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved {save_path}')
    return g


def plot_violin(ds: cell_dataset.GeneExpressionDataset, gene_names,
                top_n=6, save_path='fig_violin.png'):
    """Violin plots of expression for the top-N highly variable genes per cell type."""
    print(f'Plotting violin plots for top {top_n} most variable genes...')
    X = _to_numpy(ds.X)
    labels = _string_labels(ds)
    top_gene_names, top_indices = get_top_hvg_genes(X, gene_names, top_n=top_n)

    rows = int(np.ceil(top_n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes_flat = np.atleast_1d(axes).flat

    for ax, gene_idx, gene_name in zip(axes_flat, top_indices, top_gene_names):
        df_gene = pd.DataFrame({'expression': X[:, gene_idx],
                                'cell_type':  labels})
        sns.violinplot(data=df_gene, x='cell_type', y='expression',
                       hue='cell_type', inner=None, ax=ax,
                       density_norm='width', cut=0)
        ax.set_title(f'{gene_name}', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('Standardized Expression')
        ax.tick_params(axis='x', rotation=45, labelsize=7)

    plt.suptitle(f'Top {top_n} Highly Variable Genes — Expression by Cell Type (Training Set)',
                 y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved {save_path}')
