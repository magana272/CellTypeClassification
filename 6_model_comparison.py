"""Model comparison: AUC curves, accuracy bar charts, confusion matrices, per-class F1."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    classification_report, f1_score,
)
from sklearn.preprocessing import label_binarize

from allen_brain.models import train as T
from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.models.CellTypeGNN import build_graph_data

DATA_DIR = 'data/10x'
SAVE_DIR = 'figures'
BATCH_SIZE = 1024

MODELS = {
    'MLP': ('CellTypeMLP', True, False),
    'CNN': ('CellTypeCNN', False, False),
    'Transformer': ('CellTypeTOSICA', True, False),
    'GNN': ('CellTypeGNN', False, True),
}


def _load_and_predict(model_cls_name, squeeze_channel, is_graph):
    """Load best checkpoint and collect probabilities + predictions on test set."""
    ckpt = T.find_best_ckpt(model_cls_name)
    if ckpt is None:
        return None
    saved_kw = T._load_model_kwargs(ckpt, model_name=model_cls_name)

    if is_graph:
        kw_path = os.path.join(os.path.dirname(ckpt), 'model_kwargs.json')
        k = 15
        if os.path.exists(kw_path):
            import json
            with open(kw_path) as f:
                kw_data = json.load(f)
            k = kw_data.get('k_neighbors', 15)
        data = build_graph_data(DATA_DIR, k_neighbors=k).to(T.DEVICE)
        n_features = data.x.shape[1]
        n_classes = int(data.y.max().item()) + 1
        model = T.build_model(model_cls_name, n_features, n_classes, **saved_kw)
        model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
        y_probs, y_true = T._collect_graph_probabilities(model, data, data.test_mask)
        y_pred = y_probs.argmax(1)
        class_names = list(np.load(f'{DATA_DIR}/class_names.npy', allow_pickle=True))
    else:
        ds_test = make_dataset(DATA_DIR, split='test')
        hvg_path = os.path.join(os.path.dirname(ckpt), 'hvg_indices.npy')
        if os.path.exists(hvg_path):
            hvg_idx = np.load(hvg_path)
            ds_test.X = np.asarray(ds_test.X[:, hvg_idx])
            ds_test.gene_names = ds_test.gene_names[hvg_idx]
        n_features = len(ds_test.gene_names)
        model = T.build_model(model_cls_name, n_features, ds_test.n_classes, **saved_kw)
        model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
        pin = T.DEVICE.type == 'cuda'
        loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
        y_probs, y_true = T._collect_probabilities(model, loader, squeeze_channel)
        y_pred = y_probs.argmax(1)
        class_names = list(ds_test.class_names)

    return dict(y_probs=y_probs, y_pred=y_pred, y_true=y_true,
                class_names=class_names, n_classes=len(class_names))


def _collect_all():
    results = {}
    for name, (cls_name, squeeze, is_graph) in MODELS.items():
        print(f'Loading {name}...')
        r = _load_and_predict(cls_name, squeeze, is_graph)
        if r is not None:
            results[name] = r
            print(f'  {name}: {len(r["y_true"])} test samples')
        else:
            print(f'  {name}: no checkpoint found, skipping')
    return results


def plot_roc_per_model(results, save_dir):
    """2x2 grid of per-class ROC curves, one subplot per model."""
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, (name, r) in enumerate(results.items()):
        ax = axes[idx]
        n_classes = r['n_classes']
        y_bin = label_binarize(r['y_true'], classes=range(n_classes))
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], r['y_probs'][:, c])
            auc_val = roc_auc_score(y_bin[:, c], r['y_probs'][:, c])
            ax.plot(fpr, tpr, alpha=0.5, label=f'{r["class_names"][c]} ({auc_val:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        macro_auc = roc_auc_score(y_bin, r['y_probs'], average='macro', multi_class='ovr')
        ax.set_title(f'{name} (macro AUC={macro_auc:.4f})')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(fontsize=6, loc='lower right')
    for idx in range(len(results), 4):
        axes[idx].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves_all_models.png'), dpi=150)
    plt.close(fig)
    print(f'Saved roc_curves_all_models.png')


def plot_roc_comparison(results, save_dir):
    """Single plot with macro-avg ROC per model overlaid."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, r in results.items():
        n_classes = r['n_classes']
        y_bin = label_binarize(r['y_true'], classes=range(n_classes))
        # Compute macro-average ROC
        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], r['y_probs'][:, c])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_classes
        macro_auc = roc_auc_score(y_bin, r['y_probs'], average='macro', multi_class='ovr')
        ax.plot(all_fpr, mean_tpr, linewidth=2, label=f'{name} (AUC={macro_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Macro-Average ROC Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves_comparison.png'), dpi=150)
    plt.close(fig)
    print(f'Saved roc_curves_comparison.png')


def plot_accuracy_comparison(save_dir, csv_path='results.csv'):
    """Grouped bar chart of accuracy, F1-macro, F1-weighted from results.csv."""
    if not os.path.exists(csv_path):
        print(f'{csv_path} not found, skipping accuracy comparison plot')
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
        ax.bar(x + i * width, df[m], width, label=m.replace('_', ' ').title())
    ax.set_xticks(x + width)
    ax.set_xticklabels(df['model'])
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=150)
    plt.close(fig)
    print(f'Saved accuracy_comparison.png')


def plot_confusion_matrices(results, save_dir):
    """2x2 grid of confusion matrix heatmaps."""
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for idx, (name, r) in enumerate(results.items()):
        ax = axes[idx]
        cm = confusion_matrix(r['y_true'], r['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=r['class_names'], yticklabels=r['class_names'],
                    ax=ax)
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    for idx in range(len(results), 4):
        axes[idx].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices_comparison.png'), dpi=150)
    plt.close(fig)
    print(f'Saved confusion_matrices_comparison.png')


def plot_per_class_f1(results, save_dir):
    """Grouped bar chart of per-class F1 across models."""
    if not results:
        return
    first = next(iter(results.values()))
    class_names = first['class_names']
    n_classes = len(class_names)
    model_names = list(results.keys())
    f1_data = {}
    for name, r in results.items():
        report = classification_report(r['y_true'], r['y_pred'],
                                       target_names=class_names,
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
    plt.savefig(os.path.join(save_dir, 'per_class_f1_comparison.png'), dpi=150)
    plt.close(fig)
    print(f'Saved per_class_f1_comparison.png')


def plot_metrics_table(save_dir, csv_path='results.csv'):
    """Render a metrics table as a figure."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    # Round numeric columns
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
    plt.savefig(os.path.join(save_dir, 'metrics_table.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f'Saved metrics_table.png')


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print('=' * 60)
    print('MODEL COMPARISON')
    print('=' * 60)

    results = _collect_all()
    if not results:
        print('No trained models found. Run 4_*.py scripts first.')
        return

    plot_roc_per_model(results, SAVE_DIR)
    plot_roc_comparison(results, SAVE_DIR)
    plot_accuracy_comparison(SAVE_DIR)
    plot_confusion_matrices(results, SAVE_DIR)
    plot_per_class_f1(results, SAVE_DIR)
    plot_metrics_table(SAVE_DIR)

    print(f'\nAll figures saved to {SAVE_DIR}/')


if __name__ == '__main__':
    main()
