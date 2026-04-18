"""Shared training utilities: dataloaders, class weights, epoch loop, checkpointing, evaluation."""

from __future__ import annotations

import gc
import glob
import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import optuna
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from allen_brain.cell_data.cell_dataset import GeneExpressionDataset, make_dataset
from allen_brain.cell_data.cell_preprocess import select_hvg
from allen_brain.models import get_model
from allen_brain.models.losses import build_criterion
from allen_brain.models.config import ExperimentConfig, EvalMetrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
console = Console()
_OPTIMIZERS: dict[str, type[optim.Optimizer]] = {
    'adamw': optim.AdamW, 'adam': optim.Adam, 'sgd': optim.SGD,
}



def _resolve_optimizer(name_or_cls: str | type[optim.Optimizer]) -> type[optim.Optimizer]:
    """Resolve optimizer string or class to a class."""
    if isinstance(name_or_cls, str):
        return _OPTIMIZERS[name_or_cls]
    return name_or_cls


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_vram(model: nn.Module, batch_size: int, n_features: int) -> int:
    """Rough lower-bound VRAM estimate (bytes) for training a model."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    activation_bytes = batch_size * n_features * 4
    return param_bytes + activation_bytes


def build_model(model_name: str, n_features: int, n_classes: int,
                device: torch.device = DEVICE, **kwargs: Any) -> nn.Module:
    model = get_model(model_name, n_features, n_classes, **kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f'Model parameters: {n_params:,}')
    return model


def make_writer_and_ckpt(cfg: ExperimentConfig | dict[str, Any],
                         n_features: int) -> tuple[SummaryWriter, str]:
    run_name = make_run_name(cfg['model'], n_features, cfg['batch_size'],
                             cfg['epochs'], cfg['lr'], cfg['weight_decay'])
    log_dir = f'runs/{cfg["model"]}/{run_name}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, os.path.join(log_dir, 'best_model.pt')


def _log_normalize(X: np.ndarray) -> np.ndarray:
    """Library-size normalize + log1p: log1p(X / lib_size * 1e4)."""
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True)
    lib = np.maximum(lib, 1.0)
    return np.log1p(X / lib * 1e4)


def _apply_normalization(X_train: np.ndarray, X_val: np.ndarray,
                         normalize: str | None) -> tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    """Apply normalization to train/val arrays.

    normalize: None, 'log', 'standard', or 'log+standard'.
    Returns (X_train, X_val, scaler_or_None).
    """
    if not normalize or normalize == 'none':
        return X_train, X_val, None

    scaler: StandardScaler | None = None
    if normalize in ('log', 'log+standard'):
        console.print('Applying log normalization (log1p of library-size-normalized counts)...')
        X_train = _log_normalize(X_train)
        X_val = _log_normalize(X_val)

    if normalize in ('standard', 'log+standard'):
        console.print('Applying StandardScaler (fit on train)...')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)

    return X_train, X_val, scaler


def _apply_normalization_test(X_test: np.ndarray, normalize: str | None,
                              scaler: StandardScaler | None) -> np.ndarray:
    """Apply the same normalization to a test array."""
    if not normalize or normalize == 'none':
        return X_test
    if normalize in ('log', 'log+standard'):
        X_test = _log_normalize(X_test)
    if normalize in ('standard', 'log+standard') and scaler is not None:
        X_test = scaler.transform(X_test).astype(np.float32)
    return X_test


def class_weights(ds: GeneExpressionDataset, device: torch.device = DEVICE) -> torch.Tensor:
    counts = np.bincount(ds.y, minlength=ds.n_classes).astype(np.float32)
    w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    return w / w.sum() * ds.n_classes


def build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                    epochs: int,
                    opt_cls: str | type[optim.Optimizer] = optim.AdamW,
                    ) -> tuple[optim.Optimizer, optim.lr_scheduler.CosineAnnealingLR]:
    opt_cls = _resolve_optimizer(opt_cls)
    kwargs: dict[str, Any] = dict(lr=lr, weight_decay=weight_decay)
    if opt_cls is optim.SGD:
        kwargs.update(momentum=0.9, nesterov=True)
    optimizer = opt_cls(model.parameters(), **kwargs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler


def prep_batch(xb: torch.Tensor, yb: torch.Tensor,
               squeeze_channel: bool = False,
               device: torch.device = DEVICE) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == 'cuda'
    xb = xb.to(device, non_blocking=non_blocking)
    yb = yb.to(device, non_blocking=non_blocking)
    if squeeze_channel and xb.dim() == 3:
        xb = xb.squeeze(1)
    return xb, yb


def _autocast(xb: torch.Tensor) -> torch.autocast:
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=xb.is_cuda)


def train_batch(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor,
                criterion: nn.Module,
                optimizer: optim.Optimizer) -> tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    with _autocast(xb):
        logits = model(xb)
        loss = criterion(logits, yb)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss, logits


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              optimizer: optim.Optimizer, train: bool = True,
              squeeze_channel: bool = False,
              accumulation_steps: int = 1) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        if train and accumulation_steps > 1:
            optimizer.zero_grad(set_to_none=True)
        for i, (xb, yb) in enumerate(loader):
            xb, yb = prep_batch(xb, yb, squeeze_channel)
            if train:
                if accumulation_steps > 1:
                    with _autocast(xb):
                        logits = model(xb)
                        loss = criterion(logits, yb) / accumulation_steps
                    loss.backward()
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    # Record unscaled loss for metrics
                    total_loss += loss.item() * accumulation_steps * len(yb)
                else:
                    loss, logits = train_batch(model, xb, yb, criterion, optimizer)
                    total_loss += loss.item() * len(yb)
            else:
                with _autocast(xb):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total   += len(yb)
    return total_loss / total, correct / total


def log_epoch(writer: SummaryWriter, epoch: int,
              tr_loss: float, tr_acc: float,
              vl_loss: float, vl_acc: float) -> None:
    writer.add_scalar('Loss/train', tr_loss, epoch)
    writer.add_scalar('Loss/val', vl_loss, epoch)
    writer.add_scalar('Accuracy/train', tr_acc, epoch)
    writer.add_scalar('Accuracy/val', vl_acc, epoch)


def print_header() -> None:
    console.print(f'{"Epoch":>6} | {"Train Loss":>10} | {"Train Acc":>9} | {"Val Loss":>9} | {"Val Acc":>8} | LR')
    console.print('-' * 72)


def print_row(epoch: int, tr_loss: float, tr_acc: float,
              vl_loss: float, vl_acc: float, lr: float, flag: str) -> None:
    console.print(f'{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>8.4f} | {vl_loss:>9.4f} | {vl_acc:>8.4f} | {lr:.2e}{flag}')


def make_run_name(model_name: str, n_hvg: int, batch_size: int,
                  epochs: int, lr: float, wd: float) -> str:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f'{model_name}_hvg{n_hvg}_bs{batch_size}_epochs{epochs}_lr{lr}_wd{wd}_{ts}'


def _step_epoch(model: nn.Module,
                loaders: tuple[DataLoader, DataLoader],
                criterion: nn.Module, optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler,
                squeeze_channel: bool,
                accumulation_steps: int = 1,
                ) -> tuple[float, float, float, float]:
    train_loader, val_loader = loaders
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                train=True, squeeze_channel=squeeze_channel,
                                accumulation_steps=accumulation_steps)
    vl_loss, vl_acc = run_epoch(model, val_loader, criterion, optimizer,
                                train=False, squeeze_channel=squeeze_channel)
    scheduler.step()
    return tr_loss, tr_acc, vl_loss, vl_acc


def suggest_hparams(trial: optuna.trial.Trial, model_name: str) -> BaseHParams:
    """Suggest hyperparameters via per-model TrainConfig registry.

    Each model defines its own ranges in its TrainConfig subclass.
    Falls back to a wide generic search for unknown models.
    """
    from allen_brain.models import get_train_config
    from allen_brain.models.config import BaseHParams

    tc = get_train_config(model_name)
    if tc is not None:
        return tc.suggest_hparams(trial)

    # Fallback: wide search for unknown models
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.05, 0.5)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
    optimizer = trial.suggest_categorical('optimizer', ['adamw', 'adam', 'sgd'])
    loss = trial.suggest_categorical('loss', ['cross_entropy', 'focal'])
    focal_gamma = (trial.suggest_float('focal_gamma', 0.5, 5.0)
                   if loss == 'focal' else 2.0)
    normalize = trial.suggest_categorical(
        'normalize', ['none', 'log', 'standard', 'log+standard'])
    return BaseHParams(
        lr=lr, weight_decay=wd, dropout=dropout,
        label_smoothing=label_smoothing, optimizer=optimizer,
        loss=loss, focal_gamma=focal_gamma, normalize=normalize,
    )


def _model_kwargs_from_params(params: BaseHParams, model_name: str) -> dict[str, Any]:
    """Extract model constructor kwargs via per-model TrainConfig registry."""
    from allen_brain.models import get_train_config

    tc = get_train_config(model_name)
    if tc is not None:
        return tc.model_kwargs_from_params(params).to_dict()
    return dict(dropout=params.dropout)


def _tune_writer_ckpt(cfg: ExperimentConfig | dict[str, Any],
                      trial_number: int) -> tuple[SummaryWriter, str]:
    log_dir = f'runs/{cfg["model"]}/tune/trial_{trial_number}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, os.path.join(log_dir, 'best_model.pt')


def run_optuna_study(cfg: ExperimentConfig | dict[str, Any],
                     objective: Any, n_trials: int,
                     tune_epochs: int) -> dict[str, Any] | None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=cfg.get('seed', 0)),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=tune_epochs, reduction_factor=3))
    console.print(f'Hparam search: {n_trials} trials x up to {tune_epochs} epochs (Hyperband)')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, gc_after_trial=True)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        console.print('[bold yellow]WARNING[/bold yellow]: no trials completed (all pruned/failed) — '
              'falling back to default lr/weight_decay. '
              'Lower tune_batch_size or set PYTORCH_ALLOC_CONF=expandable_segments:True.')
        return None
    console.print(f'Best trial: val_acc=[bold green]{study.best_value:.4f}[/bold green]  params={study.best_params}')
    return study.best_params


def _cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(model: nn.Module,
          loaders: tuple[DataLoader, DataLoader],
          criterion: nn.Module, optimizer: optim.Optimizer,
          scheduler: optim.lr_scheduler._LRScheduler,
          epochs: int, writer: SummaryWriter, ckpt: str,
          device: torch.device = DEVICE, squeeze_channel: bool = False,
          patience: int = 5, compile_model: bool = False,
          trial: optuna.trial.Trial | None = None,
          accumulation_steps: int = 1) -> float:
    if compile_model and device.type == 'cuda':
        model = torch.compile(model)
    best_loss, best_acc, no_improve = float('inf'), 0.0, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, vl_loss, vl_acc = _step_epoch(
            model, loaders, criterion, optimizer, scheduler,
            squeeze_channel, accumulation_steps=accumulation_steps)
        if trial is not None:
            trial.report(vl_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        improved = vl_loss < best_loss - 1e-4
        if improved:
            best_loss, best_acc, no_improve = vl_loss, vl_acc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
        flag = ' *' if improved else ''
        lr = scheduler.get_last_lr()[0]
        print_row(epoch, tr_loss, tr_acc, vl_loss, vl_acc, lr, flag)
        log_epoch(writer, epoch, tr_loss, tr_acc, vl_loss, vl_acc)
        if no_improve >= patience:
            console.print(f'Early stopping at epoch {epoch}')
            break
    return best_acc



def _save_model_kwargs(ckpt_dir: str, model_kw: dict[str, Any]) -> None:
    """Save model constructor kwargs as JSON next to the checkpoint."""
    # Filter out non-serialisable values (e.g. torch.Tensor masks)
    serialisable = {k: v for k, v in model_kw.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
    with open(os.path.join(ckpt_dir, 'model_kwargs.json'), 'w') as f:
        json.dump(serialisable, f, indent=2)


def _infer_model_kwargs(model_name: str, ckpt_path: str) -> dict[str, Any]:
    """Infer architectural kwargs via per-model TrainConfig registry."""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    from allen_brain.models import get_train_config
    tc = get_train_config(model_name)
    if tc is not None:
        return tc.infer_model_kwargs(sd)
    return {}


def _load_model_kwargs(ckpt_path: str, model_name: str | None = None) -> dict[str, Any]:
    """Load saved model kwargs, falling back to inference from state_dict."""
    p = os.path.join(os.path.dirname(ckpt_path), 'model_kwargs.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    if model_name is not None:
        console.print('No model_kwargs.json found, inferring architecture from checkpoint...')
        kw = _infer_model_kwargs(model_name, ckpt_path)
        if kw:
            console.print(f'  Inferred: {kw}')
        return kw
    return {}



def find_best_ckpt(model_name: str, data_tag: str | None = None) -> str | None:
    """Find the most recent best_model.pt for a given model name.

    Parameters
    ----------
    data_tag : optional str
        If provided, only return checkpoints whose path contains this tag
        (e.g. 'pbmc', '10x').
    """
    pattern = f'runs/{model_name}/*/best_model.pt'
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    matches = [m for m in matches if '/tune/' not in m]
    if data_tag:
        matches = [m for m in matches if data_tag in m]
    return matches[-1] if matches else None


def save_hyperparameters(model_name: str, best_params: dict[str, Any],
                         cfg: ExperimentConfig | dict[str, Any],
                         save_dir: str = 'finalhyperparameter') -> None:
    """Write best hyperparameters to a human-readable text file."""
    os.makedirs(save_dir, exist_ok=True)
    merged = cfg.to_dict() if isinstance(cfg, ExperimentConfig) else dict(cfg)
    merged.pop('device', None)
    if best_params:
        merged.update(best_params)
    path = os.path.join(save_dir, f'{model_name}_hyperparameters.txt')
    with open(path, 'w') as f:
        for k in sorted(merged):
            f.write(f'{k} = {merged[k]}\n')
    console.print(f'[green]Saved[/green] hyperparameters to {path}')


def load_hyperparameters(model_name: str,
                         hp_dir: str = 'finalhyperparameter') -> dict[str, Any]:
    """Load best hyperparameters from a saved text file.

    Returns a dict (empty if file not found).
    """
    path = os.path.join(hp_dir, f'{model_name}_hyperparameters.txt')
    if not os.path.exists(path):
        console.print(f'[yellow]No saved hyperparameters at {path}[/yellow]')
        return {}
    params: dict[str, Any] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip()
            try:
                if '.' in val or 'e' in val.lower():
                    params[key] = float(val)
                else:
                    params[key] = int(val)
            except ValueError:
                if val.lower() in ('true', 'false'):
                    params[key] = val.lower() == 'true'
                elif val.lower() == 'none':
                    params[key] = None
                else:
                    params[key] = val
    console.print(f'Loaded hyperparameters from {path}')
    return params


def append_results_csv(model_name: str, metrics: EvalMetrics | dict[str, Any],
                       csv_path: str = 'results.csv') -> None:
    """Append one row of evaluation metrics to results CSV."""
    row: dict[str, Any] = {'model': model_name}
    m = metrics.to_dict() if isinstance(metrics, EvalMetrics) else metrics
    for k, v in m.items():
        if k != 'confusion_matrix':
            row[k] = v
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        # Replace existing row for same model, or append
        df_old = df_old[df_old['model'] != model_name]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
    console.print(f'Results for {model_name} written to {csv_path}')



def _save_confusion_matrix(cm: np.ndarray, class_names: list[str],
                           save_path: str) -> None:
    """Save a confusion matrix heatmap as a PNG."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'[green]Saved[/green] confusion matrix to {save_path}')


def _collect_predictions(model: nn.Module, loader: DataLoader,
                         squeeze_channel: bool = False,
                         device: torch.device = DEVICE,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and collect all predictions and labels."""
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = prep_batch(xb, yb, squeeze_channel, device)
            with _autocast(xb):
                logits = model(xb)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(yb.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def _collect_probabilities(model: nn.Module, loader: DataLoader,
                           squeeze_channel: bool = False,
                           device: torch.device = DEVICE,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and collect softmax probabilities and labels."""
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = prep_batch(xb, yb, squeeze_channel, device)
            with _autocast(xb):
                logits = model(xb)
            all_probs.append(F.softmax(logits, dim=1).float().cpu())
            all_labels.append(yb.cpu())
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     class_names: list[str],
                     save_dir: str | None = None) -> EvalMetrics:
    """Compute full classification metrics and optionally save confusion matrix."""
    acc = float((y_pred == y_true).mean())
    f1_mac = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    prec_mac = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    prec_w = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    rec_mac = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    rec_w = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    console.print(Panel('[bold]EVALUATION RESULTS[/bold]',
                        border_style='cyan', expand=False))
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0)
    console.print(report)
    console.print(f'Accuracy: [bold]{acc:.4f}[/bold]')
    console.print(f'F1 (macro): [bold]{f1_mac:.4f}[/bold]  F1 (weighted): [bold]{f1_w:.4f}[/bold]')
    console.print(f'Precision (macro): [bold]{prec_mac:.4f}[/bold]  '
          f'Precision (weighted): [bold]{prec_w:.4f}[/bold]')
    console.print(f'Recall (macro): [bold]{rec_mac:.4f}[/bold]  '
          f'Recall (weighted): [bold]{rec_w:.4f}[/bold]')

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _save_confusion_matrix(cm, class_names,
                               os.path.join(save_dir, 'confusion_matrix.png'))

    return EvalMetrics(
        accuracy=acc,
        f1_macro=f1_mac,
        f1_weighted=f1_w,
        precision_macro=prec_mac,
        precision_weighted=prec_w,
        recall_macro=rec_mac,
        recall_weighted=rec_w,
        confusion_matrix=cm,
    )



class Trainer:
    """High-level training orchestrator backed by an :class:`ExperimentConfig`."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    # -- dataloaders --

    def make_dataloaders(
        self,
        data_dir: str,
        drop_last_train: bool = True,
        device: torch.device = DEVICE,
        n_hvg: int | None = None,
        normalize: str | None = None,
    ) -> tuple[DataLoader, DataLoader, np.ndarray | None, StandardScaler | None]:
        ds = make_dataset(data_dir, split='train')
        ds_val = make_dataset(data_dir, split='val')
        hvg_idx: np.ndarray | None = None
        if n_hvg is not None and 0 < n_hvg < len(ds.gene_names):
            console.print(f'Selecting top {n_hvg} HVGs by variance on train split...')
            hvg_idx = np.sort(select_hvg(np.asarray(ds.X), n_hvg))
            ds.X = np.asarray(ds.X[:, hvg_idx])
            ds_val.X = np.asarray(ds_val.X[:, hvg_idx])
            ds.gene_names = ds.gene_names[hvg_idx]
            ds_val.gene_names = ds_val.gene_names[hvg_idx]

        # Apply normalization
        X_train, X_val, scaler = _apply_normalization(
            np.asarray(ds.X), np.asarray(ds_val.X), normalize)
        ds.X = X_train
        ds_val.X = X_val

        pin = device.type == 'cuda'
        train_loader = DataLoader(ds, batch_size=self.cfg['batch_size'], shuffle=True,
                                  drop_last=drop_last_train, pin_memory=pin)
        val_loader = DataLoader(ds_val, batch_size=self.cfg['batch_size'], shuffle=False,
                                drop_last=False, pin_memory=pin)
        console.print(f'train: {len(ds)} cells, {ds.n_classes} classes, {len(ds.gene_names)} genes')
        console.print(f'val:   {len(ds_val)} cells')
        return train_loader, val_loader, hvg_idx, scaler

    # -- hparam search --

    def run_hparam_search(
        self,
        ds: GeneExpressionDataset,
        loaders: tuple[DataLoader, DataLoader],
        squeeze_channel: bool,
        n_trials: int = 15,
        tune_epochs: int = 5,
        data_dir: str | None = None,
        n_hvg_range: tuple[int, int, int] | None = None,
        extra_model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Optuna hparam search over all hyperparameters.

        Parameters
        ----------
        n_hvg_range : tuple (min, max, step), optional
            When provided, n_hvg is tuned alongside other params.
        extra_model_kwargs : dict, optional
            Extra kwargs forwarded to build_model (e.g. mask, n_pathways for TOSICA).
        """
        cfg = self.cfg
        base_normalize = cfg.get('normalize') or 'none'
        default_weights = class_weights(ds)
        extra_kw = extra_model_kwargs or {}

        def objective(trial: optuna.trial.Trial) -> float:
            model: nn.Module | None = None
            optimizer: optim.Optimizer | None = None
            scheduler: optim.lr_scheduler._LRScheduler | None = None
            writer: SummaryWriter | None = None
            trial_loaders = loaders
            try:
                params = suggest_hparams(trial, cfg['model'])
                model_kw = _model_kwargs_from_params(params, cfg['model'])
                model_kw.update(extra_kw)

                trial_normalize = params.normalize
                n_hvg = cfg.get('n_hvg')
                if n_hvg_range is not None:
                    n_hvg = trial.suggest_int(
                        'n_hvg', n_hvg_range[0], n_hvg_range[1],
                        step=n_hvg_range[2])

                needs_rebuild = (n_hvg_range is not None
                                 or trial_normalize != base_normalize)
                if needs_rebuild and data_dir is not None:
                    norm = trial_normalize if trial_normalize != 'none' else None
                    tl, vl, _, _ = self.make_dataloaders(
                        data_dir, n_hvg=n_hvg, normalize=norm)
                    trial_loaders = (tl, vl)
                    trial_ds = tl.dataset
                    n_feat = len(trial_ds.gene_names)
                    model = build_model(cfg['model'], n_feat, trial_ds.n_classes,
                                        **model_kw)
                    w = class_weights(trial_ds)
                else:
                    n_feat = len(ds.gene_names)
                    model = build_model(cfg['model'], n_feat, ds.n_classes,
                                        **model_kw)
                    w = default_weights

                criterion = build_criterion(params.loss, weight=w,
                                            label_smoothing=params.label_smoothing,
                                            gamma=params.focal_gamma)
                optimizer, scheduler = build_optimizer(
                    model, params.lr, params.weight_decay, tune_epochs,
                    opt_cls=params.optimizer)
                writer, ckpt = _tune_writer_ckpt(cfg, trial.number)
                accum = cfg.get('accumulation_steps', 1)
                return train(model, trial_loaders, criterion, optimizer, scheduler,
                             tune_epochs, writer, ckpt, squeeze_channel=squeeze_channel,
                             compile_model=False, trial=trial,
                             accumulation_steps=accum)
            except torch.cuda.OutOfMemoryError:
                console.print(f'Trial {trial.number}: [yellow]CUDA OOM[/yellow] — pruning')
                raise optuna.TrialPruned()
            finally:
                if writer is not None:
                    writer.close()
                del model, optimizer, scheduler, writer
                _cuda_cleanup()

        return run_optuna_study(cfg, objective, n_trials, tune_epochs)

    # -- train with tuning --

    def train_with_tuning(
        self,
        data_dir: str,
        squeeze_channel: bool,
        n_trials: int = 15,
        tune_epochs: int = 5,
        n_hvg_range: tuple[int, int, int] | None = None,
        extra_model_kwargs: dict[str, Any] | None = None,
    ) -> tuple[float, str, dict[str, Any]]:
        cfg = self.cfg
        normalize = cfg.get('normalize')
        train_loader, val_loader, hvg_idx, scaler = self.make_dataloaders(
            data_dir, n_hvg=cfg.get('n_hvg'), normalize=normalize)
        loaders = (train_loader, val_loader)
        ds = train_loader.dataset

        best_params = self.run_hparam_search(
            ds, loaders, squeeze_channel,
            n_trials=n_trials, tune_epochs=tune_epochs,
            data_dir=data_dir, n_hvg_range=n_hvg_range,
            extra_model_kwargs=extra_model_kwargs)

        # Apply best params (fall back to cfg defaults)
        bp: dict[str, Any] = best_params or {}
        lr = bp.get('lr', cfg['lr'])
        wd = bp.get('weight_decay', cfg['weight_decay'])
        opt_name = bp.get('optimizer', cfg.get('optimizer', 'adamw'))
        loss_name = bp.get('loss', cfg.get('loss', 'cross_entropy'))
        label_smoothing = bp.get('label_smoothing', cfg.get('label_smoothing', 0.1))
        focal_gamma = bp.get('focal_gamma', cfg.get('focal_gamma', 2.0))
        dropout = bp.get('dropout', cfg.get('dropout', 0.1))
        normalize = bp.get('normalize', cfg.get('normalize'))
        if normalize == 'none':
            normalize = None

        # Rebuild dataloaders with best n_hvg + normalize
        best_n_hvg = bp.get('n_hvg', cfg.get('n_hvg'))
        train_loader, val_loader, hvg_idx, scaler = self.make_dataloaders(
            data_dir, n_hvg=best_n_hvg, normalize=normalize)
        loaders = (train_loader, val_loader)
        ds = train_loader.dataset

        # Build model with best architectural params
        model_kw: dict[str, Any] = dict(dropout=dropout)
        for k in ('n_layers', 'hidden_dim', 'n_stages', 'n_heads', 'embed_dim'):
            if k in bp:
                model_kw[k] = bp[k]
        if extra_model_kwargs:
            model_kw.update(extra_model_kwargs)
        model = build_model(cfg['model'], len(ds.gene_names), ds.n_classes, **model_kw)

        criterion = build_criterion(loss_name, weight=class_weights(ds),
                                    label_smoothing=label_smoothing, gamma=focal_gamma)
        optimizer, scheduler = build_optimizer(
            model, lr, wd, cfg['epochs'], opt_cls=opt_name)
        writer, ckpt = make_writer_and_ckpt(cfg, len(ds.gene_names))

        ckpt_dir = os.path.dirname(ckpt)
        if hvg_idx is not None:
            np.save(os.path.join(ckpt_dir, 'hvg_indices.npy'), hvg_idx)
        if scaler is not None:
            with open(os.path.join(ckpt_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
        if normalize:
            with open(os.path.join(ckpt_dir, 'normalize.txt'), 'w') as f:
                f.write(normalize)
        _save_model_kwargs(ckpt_dir, model_kw)

        console.print(f'Training {cfg["epochs"]} epochs with best params on {DEVICE}...')
        print_header()
        accum = cfg.get('accumulation_steps', 1)
        best = train(model, loaders, criterion, optimizer, scheduler,
                     cfg['epochs'], writer, ckpt, squeeze_channel=squeeze_channel,
                     accumulation_steps=accum)
        console.print(f'\nBest validation accuracy: [bold green]{best:.4f}[/bold green]')
        return best, ckpt, bp

    # -- train with grid --

    def train_with_grid(
        self,
        data_dir: str,
        squeeze_channel: bool,
        grid: list[dict[str, Any]],
        tune_epochs: int,
        extra_model_kwargs: dict[str, Any] | None = None,
    ) -> tuple[float, str, dict[str, Any]]:
        """Deterministic grid search: train each config for tune_epochs, then
        do full training with the winner.

        Parameters
        ----------
        grid : list[dict]
            Each dict has keys: lr, weight_decay, dropout, label_smoothing,
            optimizer, loss, focal_gamma, normalize, and model-specific
            architectural keys (n_layers, hidden_dim, etc.).
        """
        cfg = self.cfg
        extra_kw = extra_model_kwargs or {}
        best_idx, best_val_acc = 0, -1.0
        cached_normalize: object = object()  # sentinel
        loaders: tuple[DataLoader, DataLoader] | None = None
        ds: GeneExpressionDataset | None = None

        console.print(f'Grid search: {len(grid)} configs x {tune_epochs} epochs')

        for i, params in enumerate(grid):
            console.print(Panel(f'[bold]Grid config {i+1}/{len(grid)}[/bold]: {params}',
                                border_style='cyan'))
            model: nn.Module | None = None
            optimizer: optim.Optimizer | None = None
            scheduler: optim.lr_scheduler._LRScheduler | None = None
            writer: SummaryWriter | None = None
            try:
                norm = params.get('normalize', cfg.get('normalize'))
                if norm == 'none':
                    norm = None
                n_hvg = params.get('n_hvg', cfg.get('n_hvg'))

                if norm != cached_normalize or loaders is None:
                    train_loader, val_loader, _, _ = self.make_dataloaders(
                        data_dir, n_hvg=n_hvg, normalize=norm)
                    loaders = (train_loader, val_loader)
                    ds = train_loader.dataset
                    cached_normalize = norm

                model_kw = _model_kwargs_from_params(params, cfg['model'])
                model_kw.update(extra_kw)

                model = build_model(cfg['model'], len(ds.gene_names), ds.n_classes,
                                    **model_kw)
                w = class_weights(ds)
                criterion = build_criterion(
                    params.get('loss', cfg.get('loss', 'cross_entropy')),
                    weight=w,
                    label_smoothing=params.get('label_smoothing', 0.1),
                    gamma=params.get('focal_gamma', 2.0))
                optimizer, scheduler = build_optimizer(
                    model, params['lr'], params['weight_decay'],
                    tune_epochs, opt_cls=params.get('optimizer', 'adamw'))
                writer, ckpt = _tune_writer_ckpt(cfg, i)

                accum = cfg.get('accumulation_steps', 1)
                print_header()
                val_acc = train(model, loaders, criterion, optimizer, scheduler,
                                tune_epochs, writer, ckpt,
                                squeeze_channel=squeeze_channel,
                                accumulation_steps=accum)

                console.print(f'Config {i+1} val_acc: [bold]{val_acc:.4f}[/bold]')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_idx = i
            finally:
                if writer is not None:
                    writer.close()
                del model, optimizer, scheduler, writer
                _cuda_cleanup()

        # --- Phase 2: full training with winner ---
        bp = grid[best_idx]
        console.print(Panel(f'[bold green]Winner: config {best_idx+1}[/bold green] '
                            f'(val_acc={best_val_acc:.4f})\n{bp}',
                            border_style='green'))

        lr = bp.get('lr', cfg['lr'])
        wd = bp.get('weight_decay', cfg['weight_decay'])
        opt_name = bp.get('optimizer', cfg.get('optimizer', 'adamw'))
        loss_name = bp.get('loss', cfg.get('loss', 'cross_entropy'))
        label_smoothing = bp.get('label_smoothing', cfg.get('label_smoothing', 0.1))
        focal_gamma = bp.get('focal_gamma', cfg.get('focal_gamma', 2.0))
        dropout = bp.get('dropout', cfg.get('dropout', 0.1))
        normalize = bp.get('normalize', cfg.get('normalize'))
        if normalize == 'none':
            normalize = None

        best_n_hvg = bp.get('n_hvg', cfg.get('n_hvg'))
        train_loader, val_loader, hvg_idx, scaler = self.make_dataloaders(
            data_dir, n_hvg=best_n_hvg, normalize=normalize)
        loaders = (train_loader, val_loader)
        ds = train_loader.dataset

        model_kw_final: dict[str, Any] = dict(dropout=dropout)
        for k in ('n_layers', 'hidden_dim', 'n_stages', 'n_heads', 'embed_dim'):
            if k in bp:
                model_kw_final[k] = bp[k]
        if extra_model_kwargs:
            model_kw_final.update(extra_model_kwargs)
        model = build_model(cfg['model'], len(ds.gene_names), ds.n_classes, **model_kw_final)

        criterion = build_criterion(loss_name, weight=class_weights(ds),
                                    label_smoothing=label_smoothing, gamma=focal_gamma)
        optimizer_final, scheduler_final = build_optimizer(
            model, lr, wd, cfg['epochs'], opt_cls=opt_name)
        writer_final, ckpt = make_writer_and_ckpt(cfg, len(ds.gene_names))

        ckpt_dir = os.path.dirname(ckpt)
        if hvg_idx is not None:
            np.save(os.path.join(ckpt_dir, 'hvg_indices.npy'), hvg_idx)
        if scaler is not None:
            with open(os.path.join(ckpt_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
        if normalize:
            with open(os.path.join(ckpt_dir, 'normalize.txt'), 'w') as f:
                f.write(normalize)
        _save_model_kwargs(ckpt_dir, model_kw_final)

        console.print(f'Training {cfg["epochs"]} epochs with best params on {DEVICE}...')
        print_header()
        accum = cfg.get('accumulation_steps', 1)
        best = train(model, loaders, criterion, optimizer_final, scheduler_final,
                     cfg['epochs'], writer_final, ckpt, squeeze_channel=squeeze_channel,
                     accumulation_steps=accum)
        console.print(f'\nBest validation accuracy: [bold green]{best:.4f}[/bold green]')
        return best, ckpt, bp

    # -- train single --

    def train_single(
        self,
        data_dir: str,
        squeeze_channel: bool,
        extra_model_kwargs: dict[str, Any] | None = None,
        hp_dir: str = 'finalhyperparameter',
    ) -> tuple[float, str, dict[str, Any]]:
        """Single training run using saved best hyperparameters (no search).

        Returns (best_val_acc, ckpt_path, merged_params).
        """
        cfg = self.cfg
        saved = load_hyperparameters(cfg['model'], hp_dir)
        bp: dict[str, Any] = {**saved}
        # cfg overrides for session-specific settings
        for k in ('epochs', 'batch_size', 'accumulation_steps', 'seed'):
            if k in cfg:
                bp[k] = cfg[k]

        lr = bp.get('lr', cfg.get('lr', 3e-4))
        wd = bp.get('weight_decay', cfg.get('weight_decay', 1e-6))
        opt_name = bp.get('optimizer', cfg.get('optimizer', 'adamw'))
        loss_name = bp.get('loss', cfg.get('loss', 'cross_entropy'))
        label_smoothing = bp.get('label_smoothing', cfg.get('label_smoothing', 0.1))
        focal_gamma = bp.get('focal_gamma', cfg.get('focal_gamma', 2.0))
        dropout = bp.get('dropout', cfg.get('dropout', 0.1))
        normalize = bp.get('normalize', cfg.get('normalize'))
        if normalize == 'none':
            normalize = None
        n_hvg = bp.get('n_hvg', cfg.get('n_hvg', 0))
        n_hvg = int(n_hvg) if n_hvg else None

        train_loader, val_loader, hvg_idx, scaler = self.make_dataloaders(
            data_dir, n_hvg=n_hvg, normalize=normalize)
        loaders = (train_loader, val_loader)
        ds = train_loader.dataset

        model_kw: dict[str, Any] = dict(dropout=dropout)
        for k in ('n_layers', 'hidden_dim', 'n_stages', 'n_heads', 'embed_dim'):
            if k in bp:
                model_kw[k] = int(bp[k]) if isinstance(bp[k], float) and bp[k] == int(bp[k]) else bp[k]
        if extra_model_kwargs:
            model_kw.update(extra_model_kwargs)
        model = build_model(cfg['model'], len(ds.gene_names), ds.n_classes, **model_kw)

        criterion = build_criterion(loss_name, weight=class_weights(ds),
                                    label_smoothing=label_smoothing, gamma=focal_gamma)
        optimizer, scheduler = build_optimizer(
            model, lr, wd, cfg['epochs'], opt_cls=opt_name)
        writer, ckpt = make_writer_and_ckpt(cfg, len(ds.gene_names))

        ckpt_dir = os.path.dirname(ckpt)
        if hvg_idx is not None:
            np.save(os.path.join(ckpt_dir, 'hvg_indices.npy'), hvg_idx)
        if scaler is not None:
            with open(os.path.join(ckpt_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
        if normalize:
            with open(os.path.join(ckpt_dir, 'normalize.txt'), 'w') as f:
                f.write(normalize)
        _save_model_kwargs(ckpt_dir, model_kw)

        console.print(f'Training {cfg["epochs"]} epochs with fixed best params on {DEVICE}...')
        print_header()
        accum = cfg.get('accumulation_steps', 1)
        best = train(model, loaders, criterion, optimizer, scheduler,
                     cfg['epochs'], writer, ckpt, squeeze_channel=squeeze_channel,
                     accumulation_steps=accum)
        console.print(f'\nBest validation accuracy: [bold green]{best:.4f}[/bold green]')
        return best, ckpt, bp

    # -- evaluate --

    def evaluate(
        self,
        data_dir: str,
        ckpt_path: str,
        squeeze_channel: bool = False,
        extra_model_kwargs: dict[str, Any] | None = None,
    ) -> EvalMetrics:
        """Load best checkpoint and evaluate on test set with full metrics.

        Returns EvalMetrics with accuracy, f1, precision, recall, confusion_matrix.
        """
        cfg = self.cfg
        ds_test = make_dataset(data_dir, split='test')
        ckpt_dir = os.path.dirname(ckpt_path)

        # Apply HVG indices if saved during training
        hvg_path = os.path.join(ckpt_dir, 'hvg_indices.npy')
        if os.path.exists(hvg_path):
            hvg_idx = np.load(hvg_path)
            ds_test.X = np.asarray(ds_test.X[:, hvg_idx])
            ds_test.gene_names = ds_test.gene_names[hvg_idx]
            console.print(f'Applied HVG selection: {len(hvg_idx)} genes')

        # Apply normalization if saved during training
        normalize: str | None = None
        norm_path = os.path.join(ckpt_dir, 'normalize.txt')
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                normalize = f.read().strip()
        scaler: StandardScaler | None = None
        scaler_path = os.path.join(ckpt_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        if normalize:
            console.print(f'Applying saved normalization: {normalize}')
            ds_test.X = _apply_normalization_test(np.asarray(ds_test.X), normalize, scaler)

        n_features = len(ds_test.gene_names)
        n_classes = ds_test.n_classes
        class_names = list(ds_test.class_names)

        # Merge saved architectural kwargs with any extra kwargs (e.g. mask)
        saved_kw = _load_model_kwargs(ckpt_path, model_name=cfg['model'])
        if extra_model_kwargs:
            saved_kw.update(extra_model_kwargs)
        model = build_model(cfg['model'], n_features, n_classes, **saved_kw)

        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        console.print(f'Loaded checkpoint: {ckpt_path}')

        pin = DEVICE.type == 'cuda'
        test_loader = DataLoader(ds_test, batch_size=cfg['batch_size'],
                                 shuffle=False, pin_memory=pin)

        y_pred, y_true = _collect_predictions(model, test_loader, squeeze_channel)
        save_dir = os.path.dirname(ckpt_path)
        return _compute_metrics(y_true, y_pred, class_names, save_dir)

