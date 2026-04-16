from collections import Counter
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
import seaborn as sns
import umap
from sklearn.decomposition import PCA

import allen_brain.cell_data.cell_dataset as cell_dataset
import allen_brain.cell_data.cell_preprocess as cell_preprocess

FIG_DIR = 'figures'
console = Console()


def _to_numpy(a):
    if hasattr(a, 'detach'):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def _class_names(ds: cell_dataset.GeneExpressionDataset):
    return np.asarray(ds.labelencoder.classes_)


def _string_labels(ds: cell_dataset.GeneExpressionDataset):
    return _class_names(ds)[_to_numpy(ds.y)]


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_class_distribution(ds: cell_dataset.GeneExpressionDataset,
                            save_path=None):
    if save_path is None:
        save_path = os.path.join(FIG_DIR, 'fig_class_distribution.png')
    _ensure_dir(save_path)
    label_counter = Counter(_string_labels(ds))
    label_counter = sorted(label_counter.items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(label_counter)), [count for _, count in label_counter],
                color=sns.color_palette('tab20', len(label_counter)))
    ax.set_xticks(range(len(label_counter)))
    ax.set_xticklabels([label for label, _ in label_counter], rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Training Set : Cell Type Distribution')
    for bar, (_, count) in zip(bars, label_counter):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    console.print(f'[green]Saved[/green] {save_path}')


def plot_pca(ds: cell_dataset.GeneExpressionDataset,
             seed=42, n_components=50, save_path=None, file_name='fig_pca.png'):
    """Run PCA on ds.X and plot scree + PC1 vs PC2 colored by class.

    Returns
    -------
    pca : fitted sklearn PCA
    X_pca : (n_samples, n_components) transformed array
    """
    if save_path is None:
        save_path = FIG_DIR
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, file_name)

    X = _to_numpy(ds.X)
    y = _to_numpy(ds.y)
    class_names = _class_names(ds)

    console.print('Running PCA on training set...')
    pca = PCA(n_components=n_components, random_state=seed)
    X = X - X.mean(axis=0)
    X_pca = pca.fit_transform(X)
    explained_var = np.cumsum(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(1, n_components + 1), pca.explained_variance_ratio_,
                 'o-', ms=3, label='Per-PC')
    axes[0].plot(range(1, n_components + 1), explained_var, 's-', ms=3, label='Cumulative')
    axes[0].axhline(0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('PCA Scree Plot (Training Set)')
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
    axes[1].set_title('PCA vs PC1 vs PC2 (Training Set)')
    axes[1].legend(markerscale=4, fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    console.print(f'[green]Saved[/green] {save_path}')
    return pca, X_pca


def plot_umap(ds: cell_dataset.GeneExpressionDataset, X_pca=None,
              max_cells=1000, seed=42, n_neighbors=30, min_dist=0.3,
              save_path=None):
    """Fit UMAP on ds and scatter colored by class."""
    if save_path is None:
        save_path = os.path.join(FIG_DIR, 'fig_umap.png')
    _ensure_dir(save_path)
    X_pca = _to_numpy(ds.X) if X_pca is None else _to_numpy(X_pca)
    labels = _string_labels(ds)
    class_names = _class_names(ds)

    rng = np.random.default_rng(seed)
    if len(X_pca) > max_cells:
        umap_idx = rng.choice(len(X_pca), max_cells, replace=False)
        X_input = X_pca[umap_idx]
        umap_labels = labels[umap_idx]
        console.print(f'  Subsampled to {max_cells:,} cells for UMAP speed.')
    else:
        X_input = X_pca
        umap_labels = labels

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=None, verbose=True, n_jobs=os.cpu_count())
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
    ax.set_title('UMAP of Training Set : Colored by Cell Type')
    ax.legend(markerscale=5, fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    console.print(f'[green]Saved[/green] {save_path}')
    return X_umap


def get_top_hvg_genes(X, gene_names, top_n=20):
    """Top HVGs via variance of log-normalized expression (matches training pipeline)."""
    X = _to_numpy(X)
    gene_names = np.asarray(gene_names)
    top_indices = cell_preprocess.select_hvg(X, top_n)
    return gene_names[top_indices], top_indices


def plot_heatmap(ds: cell_dataset.GeneExpressionDataset, gene_names,
                 n_genes=10, n_cells_per_type=50, seed=42,
                 save_path=None):
    """Plot a clustered heatmap of the top-N highly variable genes."""
    if save_path is None:
        save_path = os.path.join(FIG_DIR, 'fig_heatmap.png')
    _ensure_dir(save_path)
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
    g.ax_heatmap.set_title(f'Top {n_genes} HVGs : Training Set Sample', pad=20)
    g.ax_heatmap.set_xlabel('Gene')
    handles = [mpatches.Patch(color=color_map[cls], label=cls) for cls in class_names]
    g.ax_col_dendrogram.legend(handles=handles, fontsize=7,
                               bbox_to_anchor=(1.15, 1), loc='upper left')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    console.print(f'[green]Saved[/green] {save_path}')
    return g


def plot_violin(ds: cell_dataset.GeneExpressionDataset, gene_names,
                top_n=6, save_path=None):
    """Violin plots of expression for the top-N highly variable genes per cell type."""
    if save_path is None:
        save_path = os.path.join(FIG_DIR, 'fig_violin.png')
    _ensure_dir(save_path)
    console.print(f'Plotting violin plots for top {top_n} most variable genes...')
    X = _to_numpy(ds.X)
    labels = _string_labels(ds)
    top_gene_names, top_indices = get_top_hvg_genes(X, gene_names, top_n=top_n)

    rows = int(np.ceil(top_n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes_flat = np.atleast_1d(axes).flat

    for ax, gene_idx, gene_name in zip(axes_flat, top_indices, top_gene_names):
        df_gene = pd.DataFrame({'expression': np.log1p(np.abs(X[:, gene_idx])),
                                'cell_type':  labels})
        sns.violinplot(data=df_gene, x='cell_type', y='expression',
                       hue='cell_type', inner=None, ax=ax,
                       density_norm='width', cut=0)
        ax.set_title(f'{gene_name}', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('log(1 + |Expression|)')
        ax.tick_params(axis='x', rotation=45, labelsize=7)

    plt.suptitle(f'Top {top_n} Highly Variable Genes: Expression by Cell Type (Training Set)',
                 y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    console.print(f'[green]Saved[/green] {save_path}')
    
def _compute_cv2_stats(X: np.ndarray):
    """Compute per-gene mean, variance, and CV^2 on library-normalized (non-log) data.

    Parameters
    ----------
    X : (n_cells, n_genes) raw count / expression matrix (NOT log-transformed).

    Returns
    -------
    mean, var, cv2 : 1-D arrays (genes with zero mean are excluded)
    valid_mask : boolean mask into the original gene axis
    """
    lib = np.maximum(X.sum(axis=1, keepdims=True, dtype=np.float64), 1.0)
    X_norm = (X / lib * 1e4).astype(np.float32)

    mean = X_norm.mean(axis=0).astype(np.float64)
    var  = X_norm.var(axis=0).astype(np.float64)

    valid = mean > 0
    mean_v = mean[valid]
    var_v  = var[valid]
    cv2_v  = var_v / (mean_v ** 2)

    return mean_v, var_v, cv2_v, valid


def _fit_cv2_trend(log_mean: np.ndarray, log_cv2: np.ndarray, frac: float = 0.3):
    """Fit a LOWESS trend to log10(CV^2) vs log10(mean).

    Returns trend_log_mean, trend_log_cv2 (sorted arrays for plotting) and
    an interpolation function trend_fn(log10_mean) -> log10_cv2.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    from scipy.interpolate import interp1d

    sort_idx = np.argsort(log_mean)
    smoothed = sm_lowess(log_cv2[sort_idx], log_mean[sort_idx], frac=frac,
                         return_sorted=True, it=3)
    trend_log_mean = smoothed[:, 0]
    trend_log_cv2  = smoothed[:, 1]

    trend_fn = interp1d(trend_log_mean, trend_log_cv2,
                        bounds_error=False, fill_value='extrapolate')
    return trend_log_mean, trend_log_cv2, trend_fn


def plot_cv2(ds: cell_dataset.GeneExpressionDataset, gene_names,
             n_top: int = 1000, save_path=None):
    """CV^2 vs mean plot for HVG identification.

    Implements the approach from Brennecke et al. (2013) /
    Bioconductor modelGeneCV2():

    1. Library-size normalize (no log).
    2. Compute per-gene mean and CV^2 = var / mean^2.
    3. Fit a LOWESS trend on the log-log scale.
    4. Rank genes by ratio = observed CV^2 / trend CV^2.
    5. Scatter-plot mean vs CV^2 highlighting the top *n_top* HVGs.
    """
    if save_path is None:
        save_path = os.path.join(FIG_DIR, 'fig_cv2.png')
    _ensure_dir(save_path)

    console.print(f'Computing CV^2 statistics for {ds.X.shape[1]:,} genes ...')
    X = _to_numpy(ds.X)
    gene_names = np.asarray(gene_names)

    mean_v, var_v, cv2_v, valid = _compute_cv2_stats(X)
    gene_names_v = gene_names[valid]

    log_mean = np.log10(mean_v)
    log_cv2  = np.log10(cv2_v)

    # -- fit trend ---------------------------------------------------------
    console.print('  Fitting LOWESS trend ...')
    trend_log_mean, trend_log_cv2, trend_fn = _fit_cv2_trend(log_mean, log_cv2)

    # -- ratio = observed / trend (linear scale) ---------------------------
    fitted_log_cv2 = trend_fn(log_mean)
    ratio = cv2_v / np.power(10.0, fitted_log_cv2)

    # -- HVG selection via select_hvg (matches training pipeline) ----------
    hvg_idx_full = cell_preprocess.select_hvg(X, n_top)   # indices into full gene axis
    # Map full-gene indices to valid-gene indices for the CV² arrays
    full_to_valid = np.full(X.shape[1], -1, dtype=int)
    full_to_valid[valid] = np.arange(valid.sum())
    hvg_in_valid = full_to_valid[hvg_idx_full]
    hvg_in_valid = hvg_in_valid[hvg_in_valid >= 0]  # drop any that had zero mean
    is_hvg = np.zeros(len(mean_v), dtype=bool)
    is_hvg[hvg_in_valid] = True
    n_actual = int(is_hvg.sum())

    # -- two-panel figure --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: CV^2 vs mean with trend
    ax = axes[0]
    ax.scatter(mean_v[~is_hvg], cv2_v[~is_hvg],
               s=1, alpha=0.15, c='grey', rasterized=True, label='Other genes')
    ax.scatter(mean_v[is_hvg], cv2_v[is_hvg],
               s=4, alpha=0.45, c='#e41a1c', rasterized=True,
               label=f'Top {n_actual} HVGs (select_hvg)')
    ax.plot(10 ** trend_log_mean, 10 ** trend_log_cv2,
            color='dodgerblue', lw=2.5, label='Fitted trend (LOWESS)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Mean expression (library-normalized)')
    ax.set_ylabel(r'CV$^2$')
    ax.set_title(r'CV$^2$ vs Mean: HVG Selection')
    ax.legend(fontsize=8, markerscale=3)

    # Panel 2: Ratio distribution
    ax2 = axes[1]
    log_ratio = np.log2(np.clip(ratio, 1e-10, None))
    ax2.hist(log_ratio[~is_hvg], bins=100, color='grey', alpha=0.6,
             label='Other genes', density=True)
    ax2.hist(log_ratio[is_hvg], bins=50, color='#e41a1c', alpha=0.6,
             label=f'Top {n_actual} HVGs', density=True)
    if is_hvg.any():
        min_hvg_ratio = ratio[is_hvg].min()
        ax2.axvline(np.log2(min_hvg_ratio), color='dodgerblue', ls='--', lw=1.5,
                    label=f'Min HVG ratio = {min_hvg_ratio:.1f}')
    ax2.set_xlabel(r'$\log_2$(CV$^2$ ratio to trend)')
    ax2.set_ylabel('Density')
    ax2.set_title(r'Distribution of CV$^2$ Ratio')
    ax2.legend(fontsize=8)

    plt.suptitle(f'{len(mean_v):,} genes with non-zero mean | '
                 f'top {n_actual} HVGs highlighted (select_hvg)', fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    top_hvg_names = gene_names[hvg_idx_full]
    top_hvg_ratios = ratio[hvg_in_valid] if len(hvg_in_valid) > 0 else np.array([])
    console.print(f'  Top 10 HVGs (select_hvg): {", ".join(top_hvg_names[:10])}')
    console.print(f'[green]Saved[/green] {save_path}')

    return top_hvg_names, top_hvg_ratios
