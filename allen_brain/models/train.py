"""Shared training utilities: dataloaders, class weights, epoch loop, checkpointing, evaluation."""

import gc
import glob
import json
import os
import pickle

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
console = Console()
_OPTIMIZERS = {'adamw': optim.AdamW, 'adam': optim.Adam, 'sgd': optim.SGD}


def _resolve_optimizer(name_or_cls):
    """Resolve optimizer string or class to a class."""
    if isinstance(name_or_cls, str):
        return _OPTIMIZERS[name_or_cls]
    return name_or_cls


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_vram(model, batch_size, n_features):
    """Rough lower-bound VRAM estimate (bytes) for training a model."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    activation_bytes = batch_size * n_features * 4
    return param_bytes + activation_bytes


def build_model(model_name: str, n_features: int, n_classes: int, device=DEVICE, **kwargs):
    model = get_model(model_name, n_features, n_classes, **kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f'Model parameters: {n_params:,}')
    return model


def make_writer_and_ckpt(cfg, n_features):
    run_name = make_run_name(cfg['model'], n_features, cfg['batch_size'],
                             cfg['epochs'], cfg['lr'], cfg['weight_decay'])
    log_dir = f'runs/{cfg["model"]}/{run_name}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, os.path.join(log_dir, 'best_model.pt')




def train_with_tuning(cfg, data_dir, squeeze_channel,
                      n_trials=15, tune_epochs=5,
                      n_hvg_range=None, extra_model_kwargs=None):
    normalize = cfg.get('normalize')
    train_loader, val_loader, hvg_idx, scaler = make_dataloaders(
        data_dir, cfg['batch_size'], n_hvg=cfg.get('n_hvg'),
        normalize=normalize)
    loaders = (train_loader, val_loader)
    ds = train_loader.dataset

    best_params = run_hparam_search(
        cfg, ds, loaders, squeeze_channel,
        n_trials=n_trials, tune_epochs=tune_epochs,
        data_dir=data_dir, n_hvg_range=n_hvg_range,
        extra_model_kwargs=extra_model_kwargs)

    # Apply best params (fall back to cfg defaults)
    bp = best_params or {}
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
    train_loader, val_loader, hvg_idx, scaler = make_dataloaders(
        data_dir, cfg['batch_size'], n_hvg=best_n_hvg,
        normalize=normalize)
    loaders = (train_loader, val_loader)
    ds = train_loader.dataset

    # Build model with best architectural params
    model_kw = dict(dropout=dropout)
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

    scaler = None
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


def make_dataloaders(data_dir, batch_size, drop_last_train=True, device=DEVICE,
                     n_hvg=None, normalize=None):
    ds = make_dataset(data_dir, split='train')
    ds_val = make_dataset(data_dir, split='val')
    hvg_idx = None
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
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                              drop_last=drop_last_train, pin_memory=pin)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            drop_last=False, pin_memory=pin)
    console.print(f'train: {len(ds)} cells, {ds.n_classes} classes, {len(ds.gene_names)} genes')
    console.print(f'val:   {len(ds_val)} cells')
    return train_loader, val_loader, hvg_idx, scaler


def class_weights(ds: GeneExpressionDataset, device=DEVICE):
    counts = np.bincount(ds.y, minlength=ds.n_classes).astype(np.float32)
    w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    return w / w.sum() * ds.n_classes


def build_optimizer(model, lr, weight_decay, epochs, opt_cls=optim.AdamW):
    opt_cls = _resolve_optimizer(opt_cls)
    kwargs = dict(lr=lr, weight_decay=weight_decay)
    if opt_cls is optim.SGD:
        kwargs.update(momentum=0.9, nesterov=True)
    optimizer = opt_cls(model.parameters(), **kwargs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler


def prep_batch(xb, yb, squeeze_channel=False, device=DEVICE):
    non_blocking = device.type == 'cuda'
    xb = xb.to(device, non_blocking=non_blocking)
    yb = yb.to(device, non_blocking=non_blocking)
    if squeeze_channel and xb.dim() == 3:
        xb = xb.squeeze(1)
    return xb, yb


def _autocast(xb):
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=xb.is_cuda)


def train_batch(model, xb, yb, criterion, optimizer):
    optimizer.zero_grad(set_to_none=True)
    with _autocast(xb):
        logits = model(xb)
        loss = criterion(logits, yb)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss, logits


def run_epoch(model, loader, criterion, optimizer,
              train=True, squeeze_channel=False, accumulation_steps=1):
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


def log_epoch(writer, epoch, tr_loss, tr_acc, vl_loss, vl_acc):
    writer.add_scalar('Loss/train', tr_loss, epoch)
    writer.add_scalar('Loss/val', vl_loss, epoch)
    writer.add_scalar('Accuracy/train', tr_acc, epoch)
    writer.add_scalar('Accuracy/val', vl_acc, epoch)


def print_header():
    console.print(f'{"Epoch":>6} | {"Train Loss":>10} | {"Train Acc":>9} | {"Val Loss":>9} | {"Val Acc":>8} | LR')
    console.print('-' * 72)


def print_row(epoch, tr_loss, tr_acc, vl_loss, vl_acc, lr, flag):
    console.print(f'{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>8.4f} | {vl_loss:>9.4f} | {vl_acc:>8.4f} | {lr:.2e}{flag}')


def make_run_name(model_name, n_hvg, batch_size, epochs, lr, wd):
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f'{model_name}_hvg{n_hvg}_bs{batch_size}_epochs{epochs}_lr{lr}_wd{wd}_{ts}'


def _step_epoch(model, loaders, criterion, optimizer, scheduler,
                squeeze_channel, accumulation_steps=1):
    train_loader, val_loader = loaders
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                train=True, squeeze_channel=squeeze_channel,
                                accumulation_steps=accumulation_steps)
    vl_loss, vl_acc = run_epoch(model, val_loader, criterion, optimizer,
                                train=False, squeeze_channel=squeeze_channel)
    scheduler.step()
    return tr_loss, tr_acc, vl_loss, vl_acc


def suggest_hparams(trial, model_name):
    """Suggest hyperparameters narrowed around previous best results.

    Ranges are tightened per-model based on finalhyparameter/*.txt so Optuna
    converges faster with fewer trials.
    """
    # --- Model-specific priors (from previous best) ---
    if model_name == 'CellTypeMLP':
        # best: lr=4.3e-5, wd=5.3e-7, dropout=0.13, ls=0.06, adamw, CE
        lr = trial.suggest_float('lr', 2e-5, 2e-4, log=True)
        wd = trial.suggest_float('weight_decay', 1e-7, 5e-6, log=True)
        params = dict(lr=lr, weight_decay=wd)
        params['dropout'] = trial.suggest_float('dropout', 0.05, 0.25)
        params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.12)
        params['optimizer'] = trial.suggest_categorical('optimizer', ['adamw', 'adam'])
        params['loss'] = trial.suggest_categorical('loss', ['cross_entropy', 'focal'])
        params['n_layers'] = trial.suggest_int('n_layers', 1, 3)
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [256, 512])

    elif model_name == 'CellTypeCNN':
        # best: lr=7.9e-3, wd=1.7e-4, dropout=0.19, ls=0.02, adamw, CE, stages=5
        lr = trial.suggest_float('lr', 2e-3, 2e-2, log=True)
        wd = trial.suggest_float('weight_decay', 5e-5, 5e-4, log=True)
        params = dict(lr=lr, weight_decay=wd)
        params['dropout'] = trial.suggest_float('dropout', 0.1, 0.3)
        params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.06)
        params['optimizer'] = trial.suggest_categorical('optimizer', ['adamw', 'adam'])
        params['loss'] = trial.suggest_categorical('loss', ['cross_entropy', 'focal'])
        params['n_stages'] = trial.suggest_int('n_stages', 4, 5)

    elif model_name == 'CellTypeTOSICA':
        # best: lr=7.0e-3, wd=7.3e-4, dropout=0.41, ls=0.06, adam, focal(γ=0.65)
        lr = trial.suggest_float('lr', 2e-3, 2e-2, log=True)
        wd = trial.suggest_float('weight_decay', 2e-4, 2e-3, log=True)
        params = dict(lr=lr, weight_decay=wd)
        params['dropout'] = trial.suggest_categorical('dropout', [0.4137788066524075])
        params['label_smoothing'] = trial.suggest_categorical('label_smoothing', [0.06092275383467414])
        params['optimizer'] = trial.suggest_categorical('optimizer', ['adam'])
        params['loss'] = trial.suggest_categorical('loss', ['focal'])
        params['n_layers'] = trial.suggest_int('n_layers', 1,5, step=1)
        params['n_heads'] = trial.suggest_categorical('n_heads', [2, 4, 8])
        params['embed_dim'] = trial.suggest_categorical('embed_dim', [48, 64])

    elif model_name == 'CellTypeGNN':
        # best: lr=1.9e-3, wd=8.9e-7, dropout=0.38, ls=0.07, adam, focal(γ=1.94)
        lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
        wd = trial.suggest_float('weight_decay', 1e-7, 5e-6, log=True)
        params = dict(lr=lr, weight_decay=wd)
        params['dropout'] = trial.suggest_float('dropout', 0.25, 0.5)
        params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.15)
        params['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
        params['loss'] = trial.suggest_categorical('loss', ['focal', 'cross_entropy'])
        params['n_layers'] = trial.suggest_int('n_layers', 1, 3)
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128, 256])
        params['k_neighbors'] = trial.suggest_int('k_neighbors', 7, 11)

    else:
        # Fallback: wide search for unknown models
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        wd = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
        params = dict(lr=lr, weight_decay=wd)
        params['dropout'] = trial.suggest_float('dropout', 0.05, 0.5)
        params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.2)
        params['optimizer'] = trial.suggest_categorical('optimizer', ['adamw', 'adam', 'sgd'])
        params['loss'] = trial.suggest_categorical('loss', ['cross_entropy', 'focal'])

    # Shared: focal gamma + normalize
    if params['loss'] == 'focal':
        if model_name == 'CellTypeTOSICA':
            params['focal_gamma'] = trial.suggest_float('focal_gamma', 0.3, 1.5)
        elif model_name == 'CellTypeGNN':
            params['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 3.0)
        else:
            params['focal_gamma'] = trial.suggest_float('focal_gamma', 0.5, 5.0)
    else:
        params['focal_gamma'] = 2.0
    params['normalize'] = trial.suggest_categorical(
        'normalize', ['none', 'log', 'standard', 'log+standard'])

    return params


def _model_kwargs_from_params(params, model_name):
    """Extract model constructor kwargs from suggested params dict."""
    kw = dict(dropout=params['dropout'])
    if model_name == 'CellTypeMLP':
        kw.update(n_layers=params['n_layers'], hidden_dim=params['hidden_dim'])
    elif model_name == 'CellTypeCNN':
        kw['n_stages'] = params['n_stages']
        kw['use_checkpointing'] = True
    elif model_name == 'CellTypeTOSICA':
        kw.update(n_layers=params['n_layers'], n_heads=params['n_heads'],
                  embed_dim=params['embed_dim'])
    elif model_name == 'CellTypeGNN':
        kw.update(n_layers=params['n_layers'], hidden_dim=params['hidden_dim'])
    return kw


def _tune_writer_ckpt(cfg, trial_number):
    log_dir = f'runs/{cfg["model"]}/tune/trial_{trial_number}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, os.path.join(log_dir, 'best_model.pt')


def run_optuna_study(cfg, objective, n_trials, tune_epochs):
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


def _cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_hparam_search(cfg, ds, loaders, squeeze_channel,
                      n_trials=15, tune_epochs=5,
                      data_dir=None, n_hvg_range=None,
                      extra_model_kwargs=None):
    """Optuna hparam search over all hyperparameters.

    Parameters
    ----------
    n_hvg_range : tuple (min, max, step), optional
        When provided, n_hvg is tuned alongside other params.
    extra_model_kwargs : dict, optional
        Extra kwargs forwarded to build_model (e.g. mask, n_pathways for TOSICA).
    """
    base_normalize = cfg.get('normalize') or 'none'
    default_weights = class_weights(ds)
    extra_kw = extra_model_kwargs or {}

    def objective(trial):
        model = optimizer = scheduler = writer = None
        trial_loaders = loaders
        try:
            params = suggest_hparams(trial, cfg['model'])
            model_kw = _model_kwargs_from_params(params, cfg['model'])
            model_kw.update(extra_kw)

            trial_normalize = params['normalize']
            n_hvg = cfg.get('n_hvg')
            if n_hvg_range is not None:
                n_hvg = trial.suggest_int(
                    'n_hvg', n_hvg_range[0], n_hvg_range[1],
                    step=n_hvg_range[2])

            needs_rebuild = (n_hvg_range is not None
                             or trial_normalize != base_normalize)
            if needs_rebuild and data_dir is not None:
                norm = trial_normalize if trial_normalize != 'none' else None
                tl, vl, _, _ = make_dataloaders(
                    data_dir, cfg['batch_size'], n_hvg=n_hvg,
                    normalize=norm)
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

            criterion = build_criterion(params['loss'], weight=w,
                                        label_smoothing=params['label_smoothing'],
                                        gamma=params['focal_gamma'])
            optimizer, scheduler = build_optimizer(
                model, params['lr'], params['weight_decay'], tune_epochs,
                opt_cls=params['optimizer'])
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


def _graph_step(model, data, criterion, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = criterion(logits[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    acc = (logits[train_mask].argmax(1) == data.y[train_mask]).float().mean().item()
    return loss.item(), acc


def _graph_eval(model, data, criterion, val_mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[val_mask], data.y[val_mask]).item()
        acc = (logits[val_mask].argmax(1) == data.y[val_mask]).float().mean().item()
    return loss, acc


def train_graph(model, data, criterion, optimizer, scheduler, epochs, writer, ckpt,
                patience=5, trial=None):
    best_loss, best_acc, no_improve = float('inf'), 0.0, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _graph_step(model, data, criterion, optimizer, data.train_mask)
        vl_loss, vl_acc = _graph_eval(model, data, criterion, data.val_mask)
        scheduler.step()
        improved = vl_loss < best_loss - 1e-4
        if improved:
            best_loss, best_acc, no_improve = vl_loss, vl_acc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
        lr = scheduler.get_last_lr()[0]
        print_row(epoch, tr_loss, tr_acc, vl_loss, vl_acc, lr, ' *' if improved else '')
        log_epoch(writer, epoch, tr_loss, tr_acc, vl_loss, vl_acc)
        if trial is not None:
            trial.report(vl_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if no_improve >= patience:
            console.print(f'Early stopping at epoch {epoch}')
            break
    return best_acc


def run_graph_hparam_search(cfg, data_dir, n_features, n_classes, weights,
                            n_trials=15, tune_epochs=30):
    from allen_brain.models.CellTypeGNN import build_graph_data
    graph_cache = {}

    def _get_graph(k, norm):
        key = (k, norm)
        if key not in graph_cache:
            graph_cache[key] = build_graph_data(
                data_dir, k_neighbors=k,
                normalize=norm if norm != 'none' else None).to(DEVICE)
        return graph_cache[key]

    def objective(trial):
        model = optimizer = scheduler = writer = None
        try:
            params = suggest_hparams(trial, 'CellTypeGNN')
            k = params['k_neighbors']
            data = _get_graph(k, params['normalize'])

            model_kw = _model_kwargs_from_params(params, 'CellTypeGNN')
            model = build_model('CellTypeGNN', n_features, n_classes, **model_kw)

            criterion = build_criterion(params['loss'], weight=weights,
                                        label_smoothing=params['label_smoothing'],
                                        gamma=params['focal_gamma'])
            optimizer, scheduler = build_optimizer(
                model, params['lr'], params['weight_decay'], tune_epochs,
                opt_cls=params['optimizer'])
            writer, ckpt = _tune_writer_ckpt(cfg, trial.number)
            return train_graph(model, data, criterion, optimizer, scheduler,
                               tune_epochs, writer, ckpt, patience=5,
                               trial=trial)
        except torch.cuda.OutOfMemoryError:
            console.print(f'Trial {trial.number}: [yellow]CUDA OOM[/yellow] — pruning')
            raise optuna.TrialPruned()
        finally:
            if writer is not None:
                writer.close()
            del model, optimizer, scheduler, writer
            _cuda_cleanup()

    result = run_optuna_study(cfg, objective, n_trials, tune_epochs)
    graph_cache.clear()
    return result


def train_graph_with_tuning(cfg, data_dir, n_features, n_classes, weights,
                            n_trials=15, tune_epochs=30):
    from allen_brain.models.CellTypeGNN import build_graph_data

    best_params = run_graph_hparam_search(
        cfg, data_dir, n_features, n_classes, weights,
        n_trials=n_trials, tune_epochs=tune_epochs)

    # Apply best params (fall back to cfg defaults)
    bp = best_params or {}
    lr = bp.get('lr', cfg['lr'])
    wd = bp.get('weight_decay', cfg['weight_decay'])
    opt_name = bp.get('optimizer', cfg.get('optimizer', 'adamw'))
    loss_name = bp.get('loss', cfg.get('loss', 'cross_entropy'))
    label_smoothing = bp.get('label_smoothing', cfg.get('label_smoothing', 0.1))
    focal_gamma = bp.get('focal_gamma', cfg.get('focal_gamma', 2.0))
    dropout = bp.get('dropout', cfg.get('dropout', 0.3))
    k_neighbors = bp.get('k_neighbors', cfg.get('k_neighbors', 15))
    normalize = bp.get('normalize', cfg.get('normalize'))
    if normalize == 'none':
        normalize = None

    # Build graph with best k + normalize
    data = build_graph_data(data_dir, k_neighbors=k_neighbors,
                            normalize=normalize).to(DEVICE)

    # Build model with best architectural params
    model_kw = dict(dropout=dropout)
    for k in ('n_layers', 'hidden_dim'):
        if k in bp:
            model_kw[k] = bp[k]
    model = build_model('CellTypeGNN', n_features, n_classes, **model_kw)

    criterion = build_criterion(loss_name, weight=weights,
                                label_smoothing=label_smoothing, gamma=focal_gamma)
    optimizer, scheduler = build_optimizer(
        model, lr, wd, cfg['epochs'], opt_cls=opt_name)
    writer, ckpt = make_writer_and_ckpt(cfg, n_features)
    _save_model_kwargs(os.path.dirname(ckpt), model_kw)
    console.print(f'\nData: {data}')
    console.print(f'Training {cfg["epochs"]} epochs with best params on {DEVICE}...')
    print_header()
    best = train_graph(model, data, criterion, optimizer, scheduler,
                       cfg['epochs'], writer, ckpt)
    console.print(f'\nBest validation accuracy: [bold green]{best:.4f}[/bold green]')
    return best, ckpt, bp


def train(model, loaders, criterion, optimizer, scheduler, epochs, writer, ckpt,
          device=DEVICE, squeeze_channel=False, patience=5, compile_model=False,
          trial=None, accumulation_steps=1):
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


# ---------------------------------------------------------------------------
# Model kwargs persistence (so evaluate can reconstruct the exact architecture)
# ---------------------------------------------------------------------------

def _save_model_kwargs(ckpt_dir, model_kw):
    """Save model constructor kwargs as JSON next to the checkpoint."""
    # Filter out non-serialisable values (e.g. torch.Tensor masks)
    serialisable = {k: v for k, v in model_kw.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
    with open(os.path.join(ckpt_dir, 'model_kwargs.json'), 'w') as f:
        json.dump(serialisable, f, indent=2)


def _infer_model_kwargs(model_name, ckpt_path):
    """Infer architectural kwargs by peeking at state_dict weight shapes."""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    kw = {}
    if model_name == 'CellTypeMLP':
        if 'first.fc.0.weight' in sd:
            kw['hidden_dim'] = sd['first.fc.0.weight'].shape[0]
        n_hidden = sum(1 for k in sd if k.startswith('hidden.') and k.endswith('.fc.0.weight'))
        kw['n_layers'] = 1 + n_hidden
    elif model_name == 'CellTypeCNN':
        n_stages = sum(1 for k in sd if k.startswith('stages.') and k.endswith('.0.conv1.weight'))
        if n_stages > 0:
            kw['n_stages'] = n_stages
    elif model_name == 'CellTypeGNN':
        n_layers = sum(1 for k in sd if k.startswith('blocks.') and 'conv.lin_l.weight' in k)
        if n_layers > 0:
            kw['n_layers'] = n_layers
        if 'blocks.0.conv.lin_l.weight' in sd:
            kw['hidden_dim'] = sd['blocks.0.conv.lin_l.weight'].shape[0]
        elif 'encoder.0.weight' in sd:
            kw['hidden_dim'] = sd['encoder.0.weight'].shape[0]
    elif model_name == 'CellTypeTOSICA':
        n_layers = sum(1 for k in sd if k.startswith('transformer.') and 'self_attn.in_proj_weight' in k)
        if n_layers > 0:
            kw['n_layers'] = n_layers
    return kw


def _load_model_kwargs(ckpt_path, model_name=None):
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


# ---------------------------------------------------------------------------
# Hyperparameter & results persistence
# ---------------------------------------------------------------------------

def find_best_ckpt(model_name):
    """Find the most recent best_model.pt for a given model name."""
    pattern = f'runs/{model_name}/*/best_model.pt'
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    matches = [m for m in matches if '/tune/' not in m]
    return matches[-1] if matches else None


def save_hyperparameters(model_name, best_params, cfg, save_dir='finalhyperparameter'):
    """Write best hyperparameters to a human-readable text file."""
    os.makedirs(save_dir, exist_ok=True)
    merged = dict(cfg)
    merged.pop('device', None)
    if best_params:
        merged.update(best_params)
    path = os.path.join(save_dir, f'{model_name}_hyperparameters.txt')
    with open(path, 'w') as f:
        for k in sorted(merged):
            f.write(f'{k} = {merged[k]}\n')
    console.print(f'[green]Saved[/green] hyperparameters to {path}')


def append_results_csv(model_name, metrics, csv_path='results.csv'):
    """Append one row of evaluation metrics to results CSV."""
    row = {'model': model_name}
    for k, v in metrics.items():
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _save_confusion_matrix(cm, class_names, save_path):
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


def _collect_predictions(model, loader, squeeze_channel=False, device=DEVICE):
    """Run inference and collect all predictions and labels."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = prep_batch(xb, yb, squeeze_channel, device)
            with _autocast(xb):
                logits = model(xb)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(yb.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def _collect_probabilities(model, loader, squeeze_channel=False, device=DEVICE):
    """Run inference and collect softmax probabilities and labels."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = prep_batch(xb, yb, squeeze_channel, device)
            with _autocast(xb):
                logits = model(xb)
            all_probs.append(F.softmax(logits, dim=1).float().cpu())
            all_labels.append(yb.cpu())
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def _collect_graph_probabilities(model, data, mask):
    """Run graph inference and collect softmax probabilities for masked nodes."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    probs = F.softmax(logits[mask], dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    return probs, labels


def _compute_metrics(y_true, y_pred, class_names, save_dir=None):
    """Compute full classification metrics and optionally save confusion matrix."""
    acc = (y_pred == y_true).mean()
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm

    console.print(Panel('[bold]EVALUATION RESULTS[/bold]',
                        border_style='cyan', expand=False))
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0)
    console.print(report)
    console.print(f'Accuracy: [bold]{acc:.4f}[/bold]')
    console.print(f'F1 (macro): [bold]{metrics["f1_macro"]:.4f}[/bold]  F1 (weighted): [bold]{metrics["f1_weighted"]:.4f}[/bold]')
    console.print(f'Precision (macro): [bold]{metrics["precision_macro"]:.4f}[/bold]  '
          f'Precision (weighted): [bold]{metrics["precision_weighted"]:.4f}[/bold]')
    console.print(f'Recall (macro): [bold]{metrics["recall_macro"]:.4f}[/bold]  '
          f'Recall (weighted): [bold]{metrics["recall_weighted"]:.4f}[/bold]')

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _save_confusion_matrix(cm, class_names,
                               os.path.join(save_dir, 'confusion_matrix.png'))
    return metrics


def evaluate(cfg, data_dir, ckpt_path, squeeze_channel=False,
             extra_model_kwargs=None):
    """Load best checkpoint and evaluate on test set with full metrics.

    Returns dict with accuracy, f1, precision, recall, confusion_matrix.
    """
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
    normalize = None
    norm_path = os.path.join(ckpt_dir, 'normalize.txt')
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            normalize = f.read().strip()
    scaler = None
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


def evaluate_graph(cfg, data, ckpt_path, n_features, n_classes,
                   class_names=None):
    """Load best checkpoint and evaluate graph model on test mask with full metrics.

    Returns dict with accuracy, f1, precision, recall, confusion_matrix.
    """
    saved_kw = _load_model_kwargs(ckpt_path, model_name=cfg['model'])
    model = build_model(cfg['model'], n_features, n_classes, **saved_kw)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    console.print(f'Loaded checkpoint: {ckpt_path}')

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    mask = data.test_mask
    y_pred = logits[mask].argmax(1).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    save_dir = os.path.dirname(ckpt_path)
    return _compute_metrics(y_true, y_pred, class_names, save_dir)
