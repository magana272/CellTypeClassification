from __future__ import annotations

from collections import Counter
import os
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix as sk_confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

import allen_brain.cell_data.cell_dataset as cell_dataset
import allen_brain.cell_data.cell_preprocess as cell_preprocess
from allen_brain.models.config import ModelPredictions, EvalMetrics

FIG_DIR = 'figures'
console = Console()


def _to_numpy(a: Any) -> np.ndarray:
    if hasattr(a, 'detach'):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def _class_names(ds: cell_dataset.GeneExpressionDataset) -> np.ndarray:
    return np.asarray(ds.labelencoder.classes_)


def _string_labels(ds: cell_dataset.GeneExpressionDataset) -> np.ndarray:
    return _class_names(ds)[_to_numpy(ds.y)]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _palette(n: int) -> list[tuple[float, ...]]:
    """Return a colour palette that scales beyond 20 classes."""
    return sns.color_palette('tab20', n) if n <= 20 else sns.color_palette('husl', n)



class DatasetVisualizer:
    """Visualization suite for GeneExpressionDataset."""

    def __init__(
        self,
        ds: cell_dataset.GeneExpressionDataset,
        fig_dir: str = 'figures',
        seed: int = 42,
    ) -> None:
        self.ds = ds
        self.fig_dir = fig_dir
        self.seed = seed

    def plot_class_distribution(self, save_path: str | None = None) -> None:
        if save_path is None:
            save_path = os.path.join(self.fig_dir, 'fig_class_distribution.png')
        _ensure_dir(save_path)
        label_counter = Counter(_string_labels(self.ds))
        label_counter = sorted(label_counter.items(), key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(range(len(label_counter)), [count for _, count in label_counter],
                    color=_palette(len(label_counter)))
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

    def plot_pca(
        self,
        n_components: int = 50,
        save_path: str | None = None,
        file_name: str = 'fig_pca.png',
    ) -> tuple[PCA, np.ndarray]:
        if save_path is None:
            save_path = self.fig_dir
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, file_name)

        X = _to_numpy(self.ds.X)
        y = _to_numpy(self.ds.y)
        class_names = _class_names(self.ds)

        console.print('Running PCA on training set...')
        pca = PCA(n_components=n_components, random_state=self.seed)
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

        palette = _palette(len(class_names))
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

    def plot_umap(
        self,
        X_pca: np.ndarray | None = None,
        max_cells: int = 1000,
        n_neighbors: int = 30,
        min_dist: float = 0.3,
        save_path: str | None = None,
    ) -> np.ndarray:
        if save_path is None:
            save_path = os.path.join(self.fig_dir, 'fig_umap.png')
        _ensure_dir(save_path)
        X_pca = _to_numpy(self.ds.X) if X_pca is None else _to_numpy(X_pca)
        labels = _string_labels(self.ds)
        class_names = _class_names(self.ds)

        rng = np.random.default_rng(self.seed)
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

        palette = _palette(len(class_names))
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

    def plot_heatmap(
        self,
        gene_names: np.ndarray,
        n_genes: int = 10,
        n_cells_per_type: int = 50,
        save_path: str | None = None,
    ) -> Any:
        if save_path is None:
            save_path = os.path.join(self.fig_dir, 'fig_heatmap.png')
        _ensure_dir(save_path)
        X = _to_numpy(self.ds.X)
        labels = _string_labels(self.ds)
        class_names = _class_names(self.ds)
        gene_names = np.asarray(gene_names)

        top_gene_names, top_indices = get_top_hvg_genes(X, gene_names, top_n=n_genes)

        rng = np.random.default_rng(self.seed)
        sampled_idx: list[int] = []
        sampled_types: list[str] = []
        for cls in class_names:
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) == 0:
                continue
            chosen = rng.choice(cls_idx, size=min(n_cells_per_type, len(cls_idx)), replace=False)
            sampled_idx.extend(chosen)
            sampled_types.extend([cls] * len(chosen))

        sampled_idx_arr = np.array(sampled_idx)
        sampled_types_arr = np.array(sampled_types)
        sort_order = np.argsort(sampled_types_arr)
        sampled_idx_arr = sampled_idx_arr[sort_order]
        sampled_types_arr = sampled_types_arr[sort_order]

        heat_data = X[sampled_idx_arr][:, top_indices]

        palette = _palette(len(class_names))
        color_map = {cls: palette[i] for i, cls in enumerate(class_names)}
        row_colors = pd.Series(sampled_types_arr, name='Cell Type').map(color_map)

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

    def plot_violin(
        self,
        gene_names: np.ndarray,
        top_n: int = 6,
        save_path: str | None = None,
    ) -> None:
        if save_path is None:
            save_path = os.path.join(self.fig_dir, 'fig_violin.png')
        _ensure_dir(save_path)
        console.print(f'Plotting violin plots for top {top_n} most variable genes...')
        X = _to_numpy(self.ds.X)
        labels = _string_labels(self.ds)
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

    def plot_cv2(
        self,
        gene_names: np.ndarray,
        n_top: int = 1000,
        save_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if save_path is None:
            save_path = os.path.join(self.fig_dir, 'fig_cv2.png')
        _ensure_dir(save_path)

        console.print(f'Computing CV^2 statistics for {self.ds.X.shape[1]:,} genes ...')
        X = _to_numpy(self.ds.X)
        gene_names = np.asarray(gene_names)

        mean_v, var_v, cv2_v, valid = _compute_cv2_stats(X)
        gene_names_v = gene_names[valid]

        log_mean = np.log10(mean_v)
        log_cv2  = np.log10(cv2_v)

        console.print('  Fitting LOWESS trend ...')
        trend_log_mean, trend_log_cv2, trend_fn = _fit_cv2_trend(log_mean, log_cv2)

        fitted_log_cv2 = trend_fn(log_mean)
        ratio = cv2_v / np.power(10.0, fitted_log_cv2)

        hvg_idx_full = cell_preprocess.select_hvg(X, n_top)
        full_to_valid = np.full(X.shape[1], -1, dtype=int)
        full_to_valid[valid] = np.arange(valid.sum())
        hvg_in_valid = full_to_valid[hvg_idx_full]
        hvg_in_valid = hvg_in_valid[hvg_in_valid >= 0]
        is_hvg = np.zeros(len(mean_v), dtype=bool)
        is_hvg[hvg_in_valid] = True
        n_actual = int(is_hvg.sum())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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



def get_top_hvg_genes(
    X: np.ndarray, gene_names: np.ndarray, top_n: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Top HVGs via variance of log-normalized expression."""
    X = _to_numpy(X)
    gene_names = np.asarray(gene_names)
    top_indices = cell_preprocess.select_hvg(X, top_n)
    return gene_names[top_indices], top_indices


def _compute_cv2_stats(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-gene mean, variance, and CV^2 on library-normalized data."""
    lib = np.maximum(X.sum(axis=1, keepdims=True, dtype=np.float64), 1.0)
    X_norm = (X / lib * 1e4).astype(np.float32)
    mean = X_norm.mean(axis=0).astype(np.float64)
    var  = X_norm.var(axis=0).astype(np.float64)
    valid = mean > 0
    mean_v = mean[valid]
    var_v  = var[valid]
    cv2_v  = var_v / (mean_v ** 2)
    return mean_v, var_v, cv2_v, valid


def _fit_cv2_trend(
    log_mean: np.ndarray, log_cv2: np.ndarray, frac: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Fit a LOWESS trend to log10(CV^2) vs log10(mean)."""
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



class ModelComparisonVisualizer:
    """Plots for comparing multiple trained models (ROC, confusion, F1, etc.)."""

    def __init__(
        self,
        results: dict[str, ModelPredictions],
        fig_dir: str = 'figures',
    ) -> None:
        self.results = results
        self.fig_dir = fig_dir
        os.makedirs(fig_dir, exist_ok=True)

    # -- ROC ----------------------------------------------------------------

    def plot_roc_per_model(self, save_path: str | None = None) -> None:
        """2x2 grid of per-class ROC curves, one subplot per model."""
        if not self.results:
            return
        if save_path is None:
            save_path = os.path.join(self.fig_dir, 'roc_curves_all_models.png')
        _ensure_dir(save_path)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes_flat = axes.flatten()
        for idx, (name, r) in enumerate(self.results.items()):
            if idx >= 4:
                break
            ax = axes_flat[idx]
            y_bin = label_binarize(r.y_true, classes=range(r.n_classes))
            for c in range(r.n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, c], r.y_probs[:, c])
                auc_val = roc_auc_score(y_bin[:, c], r.y_probs[:, c])
                ax.plot(fpr, tpr, alpha=0.5,
                        label=f'{r.class_names[c]} ({auc_val:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            macro_auc = roc_auc_score(y_bin, r.y_probs, average='macro',
                                      multi_class='ovr')
            ax.set_title(f'{name} (macro AUC={macro_auc:.4f})')
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.legend(fontsize=6, loc='lower right')
        for idx in range(len(self.results), 4):
            axes_flat[idx].axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        console.print(f'[green]Saved[/green] {save_path}')

    def plot_roc_comparison(self, save_path: str | None = None) -> None:
        """Single plot with macro-avg ROC per model overlaid."""
        if not self.results:
            return
        if save_path is None:
            save_path = os.path.join(self.fig_dir, 'roc_curves_comparison.png')
        _ensure_dir(save_path)
        fig, ax = plt.subplots(figsize=(8, 7))
        for name, r in self.results.items():
            y_bin = label_binarize(r.y_true, classes=range(r.n_classes))
            all_fpr = np.linspace(0, 1, 200)
            mean_tpr = np.zeros_like(all_fpr)
            for c in range(r.n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, c], r.y_probs[:, c])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= r.n_classes
            macro_auc = roc_auc_score(y_bin, r.y_probs, average='macro',
                                      multi_class='ovr')
            ax.plot(all_fpr, mean_tpr, linewidth=2,
                    label=f'{name} (AUC={macro_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Macro-Average ROC Comparison')
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        console.print(f'[green]Saved[/green] {save_path}')

    # -- Confusion matrices -------------------------------------------------

    def plot_confusion_matrices(self, save_path: str | None = None) -> None:
        """Grid of normalized confusion matrix heatmaps, one per model."""
        if not self.results:
            return
        if save_path is None:
            save_path = os.path.join(self.fig_dir,
                                     'confusion_matrices_comparison.png')
        _ensure_dir(save_path)
        n = len(self.results)
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes_flat = np.atleast_1d(axes).flat
        for idx, (name, r) in enumerate(self.results.items()):
            ax = axes_flat[idx]
            cm = sk_confusion_matrix(r.y_true, r.y_pred, normalize='true')
            sns.heatmap(cm, annot=False, cmap='Blues', vmin=0, vmax=1,
                        xticklabels=r.class_names,
                        yticklabels=r.class_names, ax=ax)
            ax.tick_params(axis='both', labelsize=5)
            ax.set_title(name)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        for idx in range(n, nrows * ncols):
            axes_flat[idx].axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        console.print(f'[green]Saved[/green] {save_path}')

    # -- Per-class F1 -------------------------------------------------------

    def plot_per_class_f1(self, save_path: str | None = None) -> None:
        """Grouped bar chart of per-class F1 across models."""
        if not self.results:
            return
        if save_path is None:
            save_path = os.path.join(self.fig_dir,
                                     'per_class_f1_comparison.png')
        _ensure_dir(save_path)
        first = next(iter(self.results.values()))
        class_names = first.class_names
        n_classes = len(class_names)
        model_names = list(self.results.keys())
        f1_data: dict[str, list[float]] = {}
        for name, r in self.results.items():
            report = classification_report(
                r.y_true, r.y_pred, target_names=class_names,
                output_dict=True, zero_division=0)
            f1_data[name] = [report[cn]['f1-score'] for cn in class_names]

        x = np.arange(n_classes)
        width = 0.8 / len(model_names)
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, name in enumerate(model_names):
            ax.bar(x + i * width, f1_data[name], width, label=name)
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score Comparison')
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        console.print(f'[green]Saved[/green] {save_path}')

    # -- Accuracy / metrics bar chart from CSV ------------------------------

    @staticmethod
    def plot_accuracy_comparison(
        save_dir: str,
        csv_path: str = 'results.csv',
    ) -> None:
        """Grouped bar chart of accuracy, F1-macro, F1-weighted from CSV."""
        if not os.path.exists(csv_path):
            console.print(f'[yellow]{csv_path} not found[/yellow], skipping')
            return
        df = pd.read_csv(csv_path)
        metrics = ['accuracy', 'f1_macro', 'f1_weighted']
        available = [m for m in metrics if m in df.columns]
        if not available:
            return
        x = np.arange(len(df))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, m in enumerate(available):
            ax.bar(x + i * width, df[m], width,
                   label=m.replace('_', ' ').title())
        ax.set_xticks(x + width)
        ax.set_xticklabels(df['model'])
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        path = os.path.join(save_dir, 'accuracy_comparison.png')
        plt.savefig(path, dpi=150)
        plt.close(fig)
        console.print(f'[green]Saved[/green] {path}')

    # -- Metrics table as figure --------------------------------------------

    @staticmethod
    def plot_metrics_table(
        save_dir: str,
        csv_path: str = 'results.csv',
    ) -> None:
        """Render a metrics table as a figure."""
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].round(4)
        fig, ax = plt.subplots(figsize=(12, 2 + 0.5 * len(df)))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('Model Evaluation Metrics', fontsize=14, pad=20)
        plt.tight_layout()
        path = os.path.join(save_dir, 'metrics_table.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        console.print(f'[green]Saved[/green] {path}')

    # -- Metric heatmap (datasets x models) ---------------------------------

    @staticmethod
    def plot_metric_heatmap(
        all_results: dict[str, dict[str, EvalMetrics | None]],
        metric: str,
        model_names: list[str],
        save_dir: str,
        title: str | None = None,
        cmap: str = 'YlGn',
        filename: str | None = None,
    ) -> None:
        """Generic datasets-x-models heatmap for any EvalMetrics field."""
        datasets = list(all_results.keys())
        matrix = np.full((len(datasets), len(model_names)), np.nan)
        for i, ds in enumerate(datasets):
            for j, m in enumerate(model_names):
                r = all_results[ds].get(m)
                if r is not None:
                    matrix[i, j] = getattr(r, metric, np.nan)
        fig, ax = plt.subplots(
            figsize=(max(8, len(model_names) * 2), max(4, len(datasets))))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap=cmap,
                    xticklabels=model_names, yticklabels=datasets,
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(title or f'{metric} Across Datasets')
        plt.tight_layout()
        fname = filename or f'{metric}_heatmap.png'
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        console.print(f'[green]Saved[/green] {path}')

    # -- Confusion matrices from EvalMetrics (multi-dataset) ----------------

    @staticmethod
    def plot_eval_confusion_matrices(
        model_results: dict[str, EvalMetrics | None],
        save_dir: str,
        suptitle: str | None = None,
    ) -> None:
        """Plot confusion matrix grid from EvalMetrics dicts."""
        items = [(k, v) for k, v in model_results.items()
                 if v is not None and v.confusion_matrix is not None]
        if not items:
            return
        n = len(items)
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 5 * nrows))
        axes_flat = np.atleast_1d(axes).flat
        for idx, (label, m) in enumerate(items):
            ax = axes_flat[idx]
            cm = m.confusion_matrix
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
            sns.heatmap(cm_norm, annot=False, cmap='Blues', ax=ax,
                        vmin=0, vmax=1)
            ax.set_title(label)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        for idx in range(n, nrows * ncols):
            axes_flat[idx].axis('off')
        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
        plt.tight_layout()
        path = os.path.join(save_dir,
                            f'{suptitle or "confusion"}_matrices.png')
        plt.savefig(path, dpi=150)
        plt.close()
        console.print(f'[green]Saved[/green] {path}')
