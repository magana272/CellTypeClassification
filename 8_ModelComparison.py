from __future__ import annotations

import math
import sys
from typing import Any

from pygments import console
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.fx import Transformer
from allen_brain.TOSICA.train import balance_populations, set_seed, read_gmt, create_pathway_mask, get_gmt, MyDataSet, create_model
from allen_brain.TOSICA import TOSICA_model

# My implemented models
from allen_brain.models.CellTypeAttention import TOSICA as my_implementation_TOSICA
from allen_brain.models.CellTypeCNN import CellTypeCNN
from allen_brain.models.CellTypeGNN import CellTypeGNN, GraphBuilder
from allen_brain.models.CellTypeMLP import MLP_Model
from allen_brain.TOSICA.train import todense
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
# Other imports
from tqdm import tqdm
import time
import os
import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData
import scanpy as sc


CONFIG: dict[str, int | float] = {
    'seed': 1,
    'n_genes': 2000,
    'n_pathways': 200,
    'n_classes': 75,
    'dropout': 0.5,
    'hvg': 10_000,
    'max_g': 300,
    'max_gs': 300,
    'mask_ratio': 0.015,
    'n_unannotated': 1,
    'batch_size': 128,
    'embed_dim': 48,
    'depth': 2,
    'num_heads': 4,
    'lr': 0.001,
    'epochs': 10,
    'lrf': 0.01,
    'n_step': 10000

}
#label_name='Celltype',max_g=300,max_gs=300, mask_ratio = 0.015,n_unannotated = 1,batch_size=8, embed_dim=48,depth=2,num_heads=4,lr=0.001, epochs= 10, lrf=0.01):
PROJECT: dict[str, str | list[str]] = {
    'name': 'TOSICA_comparison',
    'model_types': ['TOSICA', 'my_TOSICA', 'MLP', 'GNN', 'CNN'],
    "mask_path": 'TOSICA_comparison/mask.npy',
    "pathway_path": 'TOSICA_comparison/pathway.csv',
    "label_dictionary_path": 'TOSICA_comparison/label_dictionary.csv',
    "model_weight_path": 'TOSICA_comparison/{}-{}.pth',
}


PRE_CONFIG: dict[str, bool | str | int] = {
   'laten': False, 'save_att': 'X_att',
   'save_lantent': 'X_lat', 'n_step': 10000,
   'cutoff': 0.1, 'n_unannotated': 1,
   'batch_size': 4096, 'embed_dim': 48,
   'depth': 2, 'num_heads': 4
}
DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'




MODELS: dict[str, type[nn.Module]] = {
    'TOSICA': TOSICA_model.Transformer,
    'my_TOSICA': my_implementation_TOSICA,
    'MLP': MLP_Model,
    'GNN': CellTypeGNN,
    'CNN': CellTypeCNN
}

def split_dataset(adata: AnnData, label_name: str = 'subclass_label', train_ratio: float = 0.7, val_ratio: float = 0.15,
                  save_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split adata into train / val / test sets (default 70/15/15).

    If save_dir is provided, saves splits to .npy files incrementally
    to avoid holding all splits in memory at once.
    """
    label_encoder: LabelEncoder = LabelEncoder()
    genes: np.ndarray = np.array(adata.var_names)

    labels: np.ndarray = label_encoder.fit_transform(adata.obs[label_name].astype('str').values)
    inverse: np.ndarray = label_encoder.inverse_transform(range(labels.max() + 1))
    print(f"Encoded labels: {inverse}")
    X: np.ndarray = np.asarray(todense(adata), dtype=np.float32)
    print(f"Original data shape: {X.shape}, labels shape: {labels.shape}")

    # Balance populations: downsample large classes to the median class size
    ct_names: np.ndarray
    ct_counts: np.ndarray
    ct_names, ct_counts = np.unique(labels, return_counts=True)
    max_per_class: int = int(np.median(ct_counts))
    print(f"Class counts: min={ct_counts.min()}, median={max_per_class}, max={ct_counts.max()}")
    balanced_idx: list[np.ndarray] = []
    for ct in ct_names:
        ct_idx: np.ndarray = np.where(labels == ct)[0]
        balanced_idx.append(np.random.choice(ct_idx, min(len(ct_idx), max_per_class), replace=False))
    balanced_idx_arr: np.ndarray = np.concatenate(balanced_idx)
    X = X[balanced_idx_arr]
    labels = labels[balanced_idx_arr]
    print(f"Balanced: {len(labels)} samples, {len(ct_names)} classes, {max_per_class} per class")

    n: int = len(X)
    n_train: int = int(n * train_ratio)
    n_val: int = int(n * val_ratio)
    print(f"Total samples: {n}, genes: {X.shape[1]}, classes: {len(inverse)}")

    indices: np.ndarray = np.random.permutation(n)
    train_idx: np.ndarray = indices[:n_train]
    val_idx: np.ndarray = indices[n_train:n_train + n_val]
    test_idx: np.ndarray = indices[n_train + n_val:]
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")

    # Save each split to disk one at a time, then free it
    splits: list[tuple[str, np.ndarray]] = [('train', train_idx), ('val', val_idx), ('test', test_idx)]
    for name, idx in splits:
        np.save(os.path.join(save_dir, f'exp_{name}.npy'), X[idx])
        np.save(os.path.join(save_dir, f'label_{name}.npy'), labels[idx].astype(np.int64))
    del X, labels

    return inverse, genes


def _forward_model(model: nn.Module, exp: torch.Tensor, model_type: str, edge_index: torch.Tensor | None = None) -> torch.Tensor:
    """Call model.forward(), returning only the logits regardless of model type."""
    if model_type == 'TOSICA':
        _, pred, _ = model(exp)
        return pred
    elif model_type == 'CNN':
        return model(exp.unsqueeze(1))
    elif model_type == 'GNN':
        return model(exp, edge_index)
    else:
        # my_TOSICA, MLP — all return logits directly
        return model(exp)


def _train_epoch(model: nn.Module, optimizer: optim.Optimizer, data_loader: DataLoader, device: torch.device, epoch: int, model_type: str) -> tuple[float, float]:
    """Train one epoch, handling different model forward signatures."""
    model.train()
    loss_fn: nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    accu_loss: torch.Tensor = torch.zeros(1).to(device)
    accu_num: torch.Tensor = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num: int = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp: torch.Tensor
        label: torch.Tensor
        exp, label = data
        exp, label = exp.to(device), label.to(device)
        sample_num += exp.shape[0]
        pred: torch.Tensor = _forward_model(model, exp, model_type)
        pred_classes: torch.Tensor = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label).sum()
        loss: torch.Tensor = loss_fn(pred, label)
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def _eval_epoch(model: nn.Module, data_loader: DataLoader, device: torch.device, epoch: int, model_type: str) -> tuple[float, float]:
    """Evaluate one epoch, handling different model forward signatures."""
    model.eval()
    loss_fn: nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    accu_num: torch.Tensor = torch.zeros(1).to(device)
    accu_loss: torch.Tensor = torch.zeros(1).to(device)
    sample_num: int = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp: torch.Tensor
        labels: torch.Tensor
        exp, labels = data
        exp, labels = exp.to(device), labels.to(device)
        sample_num += exp.shape[0]
        pred: torch.Tensor = _forward_model(model, exp, model_type)
        pred_classes: torch.Tensor = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        loss: torch.Tensor = loss_fn(pred, labels)
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def _train_graph_epoch(model: nn.Module, graph_data: Data, optimizer: optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """Train one epoch for GNN using full-graph forward."""
    model.train()
    optimizer.zero_grad()
    logits: torch.Tensor = model(graph_data.x, graph_data.edge_index)
    loss_fn: nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    loss: torch.Tensor = loss_fn(logits[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    acc: float = (logits[graph_data.train_mask].argmax(1) == graph_data.y[graph_data.train_mask]).float().mean().item()
    return loss.item(), acc


@torch.no_grad()
def _eval_graph_epoch(model: nn.Module, graph_data: Data, device: torch.device) -> tuple[float, float]:
    """Evaluate one epoch for GNN using full-graph forward."""
    model.eval()
    logits: torch.Tensor = model(graph_data.x, graph_data.edge_index)
    loss_fn: nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    loss: torch.Tensor = loss_fn(logits[graph_data.val_mask], graph_data.y[graph_data.val_mask])
    acc: float = (logits[graph_data.val_mask].argmax(1) == graph_data.y[graph_data.val_mask]).float().mean().item()
    return loss.item(), acc


def fit_model(adata: AnnData, gmt_path: str | None, project: str | None = None, pre_weights: str = '', label_name: str = 'subclass_label',
              max_g: int = 300, max_gs: int = 300, mask_ratio: float = 0.015, n_unannotated: int = 1, batch_size: int = 8,
              embed_dim: int = 48, depth: int = 2, num_heads: int = 4, lr: float = 0.001, epochs: int = 10, lrf: float = 0.01,
              model_type: str = 'TOSICA') -> None:
    set_seed(CONFIG['seed'])
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    today: str = time.strftime('%Y%m%d', time.localtime(time.time()))
    project = project or gmt_path.replace('.gmt', '') + '_%s' % today
    project_path: str = os.getcwd() + '/%s' % project
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    tb_writer: SummaryWriter = SummaryWriter(log_dir=f'runs/model_comparison/{model_type}')
    splits_exist: bool = all(os.path.exists(project_path + f'/{f}.npy')
                       for f in ('exp_train', 'label_train', 'exp_val', 'label_val',
                                 'exp_test', 'label_test'))
    genes: np.ndarray = np.array(adata.var_names)
    if splits_exist:
        inverse: np.ndarray = pd.read_csv(project_path + '/label_dictionary.csv', index_col=0).values.flatten()
        print('Split data loaded!')
    else:
        print('Split data not found, creating new split...')
        inverse, _ = split_dataset(adata, label_name, save_dir=project_path)
        pd.DataFrame(inverse, columns=[label_name]).to_csv(project_path + '/label_dictionary.csv', quoting=None)
        print('Split data created and saved!')
    exp_train: np.ndarray = np.load(project_path + '/exp_train.npy')
    label_train: np.ndarray = np.load(project_path + '/label_train.npy')
    exp_val: np.ndarray = np.load(project_path + '/exp_val.npy')
    label_val: np.ndarray = np.load(project_path + '/label_val.npy')
    if gmt_path is None:
        mask: np.ndarray = np.random.binomial(1, mask_ratio, size=(len(genes), max_gs))
        pathway: list[str] = list()
        for i in range(max_gs):
            x: str = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)

    if not os.path.exists(project_path + '/mask.npy') or not os.path.exists(project_path + '/pathway.csv'):
        reactome_dict: dict[str, list[str]] = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask, pathway = create_pathway_mask(feature_list=genes,
                                            dict_pathway=reactome_dict,
                                            add_missing=n_unannotated,
                                            fully_connected=True)
        pathway = pathway[np.sum(mask, axis=0) > 4]
        mask = mask[:, np.sum(mask, axis=0) > 4]
        pathway = pathway[sorted(np.argsort(np.sum(mask, axis=0))[-min(max_gs, mask.shape[1]):])]
        mask = mask[:, sorted(np.argsort(np.sum(mask, axis=0))[-min(max_gs, mask.shape[1]):])]
        np.save(PROJECT.get('mask_path'), mask)
        pd.DataFrame(pathway).to_csv(PROJECT.get('pathway_path'))
        pd.DataFrame(inverse, columns=[label_name]).to_csv(PROJECT.get('label_dictionary_path'), quoting=None)
        print('Mask created and saved!')
    else:
        mask = np.load(PROJECT.get('mask_path'))
        pathway = pd.read_csv(PROJECT.get('pathway_path'), index_col=0)['0'].tolist()
        print('Mask loaded!')

    num_genes: int = len(exp_train[0])
    num_classes: np.int64 = np.int64(torch.max(torch.tensor(label_train)) + 1)

    #  Build model
    model: nn.Module
    if model_type == 'TOSICA':
        model = create_model(num_classes=num_classes, num_genes=num_genes,
                             mask=mask, embed_dim=embed_dim, depth=depth,
                             num_heads=num_heads, has_logits=False).to(device)
    elif model_type == 'my_TOSICA':
        mask_tensor: torch.Tensor = torch.from_numpy(mask).float() if isinstance(mask, np.ndarray) else mask.float()
        model = my_implementation_TOSICA(
            n_genes=num_genes, n_pathways=mask.shape[1], n_classes=num_classes,
            mask=mask_tensor, embed_dim=embed_dim, n_heads=num_heads,
            n_layers=depth).to(device)
    elif model_type == 'MLP':
        model = MLP_Model(input_dim=num_genes, n_classes=num_classes).to(device)
    elif model_type == 'CNN':
        model = CellTypeCNN(seq_len=num_genes, n_classes=num_classes).to(device)
    elif model_type == 'GNN':
        model = CellTypeGNN(in_dim=num_genes, hidden_dim=embed_dim,
                            n_classes=num_classes).to(device)
    else:
        raise ValueError('Invalid model type: {}'.format(model_type))

    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict: dict[str, Any] = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))
    print('Model built!')

    pg: list[torch.nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
    optimizer: optim.SGD = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    scheduler: lr_scheduler.LambdaLR = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if model_type == 'GNN':
        # GNN uses full-graph training, not mini-batch DataLoaders
        exp_train_np: np.ndarray = exp_train.numpy() if isinstance(exp_train, torch.Tensor) else exp_train
        exp_val_np: np.ndarray = exp_val.numpy() if isinstance(exp_val, torch.Tensor) else exp_val
        label_train_t: torch.Tensor = torch.tensor(label_train).long() if not isinstance(label_train, torch.Tensor) else label_train.long()
        label_val_t: torch.Tensor = torch.tensor(label_val).long() if not isinstance(label_val, torch.Tensor) else label_val.long()
        X_all: np.ndarray = np.vstack([exp_train_np, exp_val_np]).astype(np.float32)
        y_all: torch.Tensor = torch.cat([label_train_t, label_val_t])
        n_train: int = len(exp_train_np)
        n_total: int = len(X_all)
        train_mask: torch.Tensor = torch.zeros(n_total, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask: torch.Tensor = ~train_mask
        edge_index: torch.Tensor = GraphBuilder.build_knn_edges(X_all, k=15)
        graph_data: Data = Data(
            x=torch.from_numpy(X_all),
            edge_index=edge_index,
            y=y_all,
            train_mask=train_mask,
            val_mask=val_mask
        ).to(device)
        for epoch in range(epochs):
            train_loss: float
            train_acc: float
            train_loss, train_acc = _train_graph_epoch(model, graph_data, optimizer, device)
            scheduler.step()
            val_loss: float
            val_acc: float
            val_loss, val_acc = _eval_graph_epoch(model, graph_data, device)
            print("[epoch {}] train_loss: {:.3f}, train_acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}".format(
                epoch, train_loss, train_acc, val_loss, val_acc))
            tags: list[str] = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            torch.save(model.state_dict(), project_path + "/{}-{}.pth".format(model_type, epoch))
    else:
        exp_train_t: torch.Tensor = torch.as_tensor(exp_train, dtype=torch.float32)
        label_train_t: torch.Tensor = torch.as_tensor(label_train, dtype=torch.long)
        exp_val_t: torch.Tensor = torch.as_tensor(exp_val, dtype=torch.float32)
        label_val_t: torch.Tensor = torch.as_tensor(label_val, dtype=torch.long)
        train_dataset: MyDataSet = MyDataSet(exp_train_t, label_train_t)
        val_dataset: MyDataSet = MyDataSet(exp_val_t, label_val_t)
        train_loader: DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                                                   shuffle=True, drop_last=True)
        valid_loader: DataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4,
                                                   shuffle=False, drop_last=True)
        for epoch in range(epochs):
            train_loss: float
            train_acc: float
            train_loss, train_acc = _train_epoch(model=model, optimizer=optimizer,
                                                 data_loader=train_loader, device=device,
                                                 epoch=epoch, model_type=model_type)
            scheduler.step()
            val_loss: float
            val_acc: float
            val_loss, val_acc = _eval_epoch(model=model, data_loader=valid_loader,
                                            device=device, epoch=epoch, model_type=model_type)
            tags: list[str] = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            torch.save(model.state_dict(), project_path + "/{}-{}.pth".format(model_type, epoch))

def _instantiate_model(model_type: str, num_genes: int, num_classes: int, mask: np.ndarray, embed_dim: int, depth: int, num_heads: int) -> nn.Module:
    """Create a model instance with the correct constructor args for each type."""
    if model_type == 'TOSICA':
        return create_model(num_classes=num_classes, num_genes=num_genes,
                            mask=mask, embed_dim=embed_dim, depth=depth,
                            num_heads=num_heads, has_logits=False)
    elif model_type == 'my_TOSICA':
        mask_tensor: torch.Tensor = torch.from_numpy(mask).float() if isinstance(mask, np.ndarray) else mask.float()
        return my_implementation_TOSICA(
            n_genes=num_genes, n_pathways=mask.shape[1], n_classes=num_classes,
            mask=mask_tensor, embed_dim=embed_dim, n_heads=num_heads, n_layers=depth)
    elif model_type == 'MLP':
        return MLP_Model(input_dim=num_genes, n_classes=num_classes)
    elif model_type == 'CNN':
        return CellTypeCNN(seq_len=num_genes, n_classes=num_classes)
    elif model_type == 'GNN':
        return CellTypeGNN(in_dim=num_genes, hidden_dim=embed_dim, n_classes=num_classes)
    else:
        raise ValueError('Invalid model type: {}'.format(model_type))



def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Parameters
    ----------
    probs : (n_samples, n_classes) softmax probabilities
    labels : (n_samples,) true class indices
    """
    max_probs: np.ndarray = probs.max(axis=1)
    preds: np.ndarray = probs.argmax(axis=1)
    correct: np.ndarray = (preds == labels).astype(float)

    bin_boundaries: np.ndarray = np.linspace(0, 1, n_bins + 1)
    ece: float = 0.0
    for i in range(n_bins):
        in_bin: np.ndarray = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc: float = correct[in_bin].mean()
        bin_conf: float = max_probs[in_bin].mean()
        ece += in_bin.sum() / len(labels) * abs(bin_acc - bin_conf)
    return ece


def plot_reliability_diagrams(model_results: dict[str, dict[str, Any]], save_dir: str = 'figures') -> None:
    """Reliability diagram (predicted confidence vs actual accuracy) per model."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    n_models: int = len(model_results)
    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    axes = axes[0]

    for ax, (name, r) in zip(axes, model_results.items()):
        max_probs: np.ndarray = r['probs'].max(axis=1)
        correct: np.ndarray = (r['probs'].argmax(axis=1) == r['labels']).astype(int)

        frac_pos: np.ndarray
        mean_pred: np.ndarray
        frac_pos, mean_pred = calibration_curve(
            correct, max_probs, n_bins=15, strategy='uniform')

        ax.plot(mean_pred, frac_pos, 's-', label=name)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect')
        ax.set_xlabel('Mean Predicted Confidence')
        ax.set_ylabel('Fraction Correct')
        ece: float = compute_ece(r['probs'], r['labels'])
        ax.set_title(f'{name} (ECE={ece:.4f})')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reliability_diagrams.png'), dpi=150)
    plt.close()
    print(f'Saved {save_dir}/reliability_diagrams.png')


def plot_confidence_distributions(model_results: dict[str, dict[str, Any]], save_dir: str = 'figures') -> None:
    """Per-class confidence distribution (violin of max softmax per true class)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    for name, r in model_results.items():
        max_probs: np.ndarray = r['probs'].max(axis=1)
        labels: np.ndarray = r['labels']
        class_names: list[str] = r['class_names']

        df: pd.DataFrame = pd.DataFrame({
            'confidence': max_probs,
            'true_class': [class_names[i] if i < len(class_names) else 'Unknown'
                           for i in labels],
        })

        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=(max(14, len(class_names) * 0.7), 6))
        sns.violinplot(data=df, x='true_class', y='confidence', ax=ax,
                       inner='quartile', cut=0)
        ax.set_xlabel('True Cell Type')
        ax.set_ylabel('Max Softmax Confidence')
        ax.set_title(f'{name}: Confidence Distribution per Class')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'confidence_dist_{name}.png'), dpi=150)
        plt.close()
        print(f'Saved {save_dir}/confidence_dist_{name}.png')


def main() -> dict[str, AnnData]:

    if os.path.exists('data/10x/data.h5ad'):
        adata: AnnData = sc.read_h5ad('data/10x/data.h5ad')
    else:
        X_mat: np.ndarray = np.load('data/10x/matrix.npy', mmap_mode='r')
        cells: np.ndarray = np.load('data/10x/matrix_cells.npy', allow_pickle=True)
        genes: np.ndarray = np.load('data/10x/matrix_genes.npy', allow_pickle=True)
        meta: pd.DataFrame = pd.read_csv('data/10x/metadata.csv', index_col=0)
        obs: pd.DataFrame = meta.reindex(cells).reset_index()
        obs.index = obs.index.astype(str)
        var: pd.DataFrame = pd.DataFrame(index=genes)
        adata = AnnData(np.asarray(X_mat), obs=obs, var=var)
        adata.write_h5ad('data/10x/data.h5ad')
    models: list[str] = ['TOSICA', 'my_TOSICA', 'MLP', 'GNN', 'CNN']

    #  Train all models
    for model_name in models:
        print(f'\n=== Training {model_name} ===')
        fit_model(adata, gmt_path='CellTypeClassification/allen_brain/TOSICA/resources/reactome.gmt',
                  project='TOSICA_comparison', model_type=model_name,
                  batch_size=CONFIG['batch_size'])

    #  Evaluate all models
    device: torch.device = torch.device(DEVICE)
    mask: np.ndarray = np.load(PROJECT.get('mask_path'))
    pathway: pd.DataFrame = pd.read_csv(PROJECT.get('pathway_path'), index_col=0)
    dictionary: pd.DataFrame = pd.read_table(PROJECT.get('label_dictionary_path'), sep=',', header=0, index_col=0)
    n_c: int = len(dictionary)
    label_name: str = dictionary.columns[0]
    dictionary.loc[dictionary.shape[0]] = 'Unknown'
    dic: dict[int, str] = {i: dictionary[label_name][i] for i in range(len(dictionary))}
    num_genes: int = adata.shape[1]

    results: dict[str, AnnData] = {}
    calibration_data: dict[str, dict[str, Any]] = {}
    for model_type in models:
        print(f'\n=== Evaluating {model_type} ===')
        model_path: str = PROJECT.get('model_weight_path').format(model_type, CONFIG['epochs'] - 1)
        model: nn.Module = _instantiate_model(model_type, num_genes, n_c,
                                   mask, CONFIG['embed_dim'],
                                   CONFIG['depth'], CONFIG['num_heads'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        #  Gene2token analysis (only for original TOSICA)
        if model_type == 'TOSICA':
            parm: dict[str, np.ndarray] = {}
            for name, parameters in model.named_parameters():
                parm[name] = parameters.detach().cpu().numpy()
            gene2token: np.ndarray = parm['feature_embed.fe.weight']
            gene2token = gene2token.reshape(
                (int(gene2token.shape[0] / CONFIG['embed_dim']),
                 CONFIG['embed_dim'], adata.shape[1]))
            gene2token = abs(gene2token)
            gene2token = np.max(gene2token, axis=1)
            gene2token_df: pd.DataFrame = pd.DataFrame(gene2token)
            gene2token_df.columns = adata.var_names
            gene2token_df.index = pathway['0']
            gene2token_df.to_csv('TOSICA_comparison/gene2token_weights.csv')

        #  Load test set
        project_path: str = os.getcwd() + '/TOSICA_comparison'
        exp_test_np: np.ndarray = np.load(project_path + '/exp_test.npy').astype(np.float32)
        label_test: np.ndarray = np.load(project_path + '/label_test.npy').astype(np.int64)
        exp_test: torch.Tensor = torch.from_numpy(exp_test_np)

        #  Build GNN graph if needed
        edge_index: torch.Tensor | None = None
        if model_type == 'GNN':
            edge_index = GraphBuilder.build_knn_edges(exp_test_np, k=15).to(device)

        #  Run inference on test set
        all_line: int = len(exp_test)
        n_line: int = 0
        adata_list: list[AnnData] = []
        all_preds: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []  # full softmax probability matrices for calibration
        while n_line < all_line:
            chunk_size: int = min(CONFIG['n_step'], all_line - n_line)
            if chunk_size % CONFIG['batch_size'] == 1 and chunk_size > 2:
                chunk_size -= 2
            chunk_exp: torch.Tensor = exp_test[n_line:n_line + chunk_size]
            print(f'{model_type}: processing cells {n_line} to {n_line + chunk_size}')

            data_loader: DataLoader = torch.utils.data.DataLoader(
                chunk_exp, batch_size=CONFIG['batch_size'],
                shuffle=False, pin_memory=True)

            with torch.no_grad():
                for step, data in enumerate(data_loader):
                    exp: torch.Tensor = data.to(device)

                    pre_logits: torch.Tensor
                    weights: torch.Tensor | None
                    if model_type == 'TOSICA':
                        _, pre_logits, weights = model(exp)
                    elif model_type == 'my_TOSICA':
                        pre_logits, weights = model(exp, return_attention=True)
                    elif model_type == 'GNN':
                        full_logits: torch.Tensor = model(exp_test.to(device), edge_index)
                        batch_start: int = n_line + step * CONFIG['batch_size']
                        batch_end: int = min(batch_start + CONFIG['batch_size'], all_line)
                        pre_logits = full_logits[batch_start:batch_end]
                        weights = None
                    elif model_type == 'CNN':
                        pre_logits = model(exp.unsqueeze(1))
                        weights = None
                    else:  # MLP
                        pre_logits = model(exp)
                        weights = None

                    pre: torch.Tensor = torch.squeeze(pre_logits).cpu()
                    pre = F.softmax(pre, dim=1) if pre.dim() > 1 else F.softmax(pre.unsqueeze(0), dim=1)
                    predict_class: np.ndarray = np.empty(shape=0)
                    pre_class: np.ndarray = np.empty(shape=0)
                    for i in range(len(pre)):
                        if torch.max(pre, dim=1)[0][i] >= .1:
                            predict_class = np.r_[predict_class, torch.max(pre, dim=1)[1][i].numpy()]
                        else:
                            predict_class = np.r_[predict_class, n_c]
                        pre_class = np.r_[pre_class, torch.max(pre, dim=1)[0][i]]
                    meta: pd.DataFrame = pd.DataFrame(
                        np.c_[predict_class, pre_class],
                        columns=['Prediction', 'Probability'])
                    meta.index = meta.index.astype('str')

                    all_preds.append(predict_class)
                    all_probs.append(pre.numpy().astype(np.float32))

                    if model_type in ('TOSICA', 'my_TOSICA') and weights is not None:
                        n_pw: int = len(pathway) if isinstance(pathway, list) else len(pathway)
                        att: np.ndarray = torch.squeeze(weights).cpu().numpy()
                        att = att[:, 0:(n_pw - PRE_CONFIG['n_unannotated'])]
                        att = att.astype('float32')
                        pw_vals: pd.Series | list[str] = pathway['0'] if isinstance(pathway, pd.DataFrame) else pathway
                        pw_trimmed: np.ndarray | list[str] = pw_vals[:n_pw - PRE_CONFIG['n_unannotated']]
                        if isinstance(pw_trimmed, pd.Series):
                            pw_trimmed = pw_trimmed.values
                        varinfo: pd.DataFrame = pd.DataFrame(
                            pw_trimmed, index=pw_trimmed,
                            columns=['pathway_index'])
                        new: AnnData = sc.AnnData(att, obs=meta, var=varinfo)
                    else:
                        new: AnnData = sc.AnnData(pre.numpy().astype('float32'), obs=meta)
                    adata_list.append(new)

            n_line += chunk_size

        print(f'{model_type}: {all_line} test cells processed')

        #  Compute metrics
        y_pred: np.ndarray = np.concatenate(all_preds).astype(np.int64)
        y_true: np.ndarray = label_test[:len(y_pred)]
        known_mask: np.ndarray = y_pred < n_c
        acc: float = accuracy_score(y_true, np.where(known_mask, y_pred, -1))
        f1_macro: float = f1_score(y_true, np.where(known_mask, y_pred, -1), average='macro', zero_division=0)
        f1_weighted: float = f1_score(y_true, np.where(known_mask, y_pred, -1), average='weighted', zero_division=0)
        precision: float = precision_score(y_true, np.where(known_mask, y_pred, -1), average='weighted', zero_division=0)
        recall: float = recall_score(y_true, np.where(known_mask, y_pred, -1), average='weighted', zero_division=0)
        # Confidence calibration
        probs_matrix: np.ndarray = np.concatenate(all_probs, axis=0)
        ece: float = compute_ece(probs_matrix, y_true)

        print(f'\n {model_type} Test Metrics ')
        print(f'  Accuracy:           {acc:.4f}')
        print(f'  F1 (macro):         {f1_macro:.4f}')
        print(f'  F1 (weighted):      {f1_weighted:.4f}')
        print(f'  Precision (weighted):{precision:.4f}')
        print(f'  Recall (weighted):  {recall:.4f}')
        print(f'  Unknown rate:       {1 - known_mask.mean():.4f}')
        print(f'  ECE:                {ece:.4f}')

        # Store calibration data for post-loop plotting
        calibration_data[model_type] = {
            'probs': probs_matrix,
            'labels': y_true,
            'class_names': [dic[i] for i in range(n_c)],
        }

        result: AnnData = ad.concat(adata_list)
        result.obs['Prediction'] = result.obs['Prediction'].map(dic)
        results[model_type] = result

    # Generate calibration plots for all models
    os.makedirs('figures', exist_ok=True)
    plot_reliability_diagrams(calibration_data, save_dir='figures')
    plot_confidence_distributions(calibration_data, save_dir='figures')

    return results


if __name__ == '__main__':
    results: dict[str, AnnData] = main()
    results['TOSICA'].write_h5ad('TOSICA_comparison/TOSICA_results.h5ad')
    results['my_TOSICA'].write_h5ad('TOSICA_comparison/my_TOSICA_results.h5ad')
    results['MLP'].write_h5ad('TOSICA_comparison/MLP_results.h5ad')
    results['GNN'].write_h5ad('TOSICA_comparison/GNN_results.h5ad')
    results['CNN'].write_h5ad('TOSICA_comparison/CNN_results.h5ad')



