"""GNN-specific training, tuning, and evaluation.

Extracted from train.py — all graph-based training logic lives here.
Shared utilities (build_model, build_optimizer, etc.) are imported from train.py.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.data import Data
from torch.utils.tensorboard.writer import SummaryWriter
import optuna

from allen_brain.models.config import ExperimentConfig, EvalMetrics
from allen_brain.models.train import (
    DEVICE, console,
    build_model, build_optimizer, build_criterion,
    make_writer_and_ckpt, _save_model_kwargs, _load_model_kwargs,
    _tune_writer_ckpt, _cuda_cleanup, run_optuna_study,
    suggest_hparams, _model_kwargs_from_params,
    print_header, print_row, log_epoch,
    load_hyperparameters, _compute_metrics,
)
from allen_brain.models.CellTypeGNN import build_graph_data


# ---------------------------------------------------------------------------
# Core graph training loop (module-level, used by both class and wrappers)
# ---------------------------------------------------------------------------

def _graph_step(
    model: nn.Module,
    data: Data,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_mask: torch.Tensor,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = criterion(logits[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    acc = (logits[train_mask].argmax(1) == data.y[train_mask]).float().mean().item()
    return loss.item(), acc


def _graph_eval(
    model: nn.Module,
    data: Data,
    criterion: nn.Module,
    val_mask: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[val_mask], data.y[val_mask]).item()
        acc = (logits[val_mask].argmax(1) == data.y[val_mask]).float().mean().item()
    return loss, acc


def train_graph(
    model: nn.Module,
    data: Data,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epochs: int,
    writer: SummaryWriter,
    ckpt: str,
    patience: int = 5,
    trial: optuna.trial.Trial | None = None,
) -> float:
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


# ---------------------------------------------------------------------------
# Probability collection helper (module-level)
# ---------------------------------------------------------------------------

def _collect_graph_probabilities(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run graph inference and collect softmax probabilities for masked nodes."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    probs = F.softmax(logits[mask], dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    return probs, labels


# ---------------------------------------------------------------------------
# GraphTrainer class
# ---------------------------------------------------------------------------

class GraphTrainer:
    """GNN-specific training orchestration backed by an ``ExperimentConfig``."""

    def __init__(self, cfg: ExperimentConfig, device: torch.device = DEVICE) -> None:
        self.cfg = cfg
        self.device = device

    # -- hyperparameter search ------------------------------------------------

    def run_hparam_search(
        self,
        data_dir: str,
        n_features: int,
        n_classes: int,
        weights: torch.Tensor,
        n_trials: int = 15,
        tune_epochs: int = 30,
    ) -> dict[str, Any] | None:
        graph_cache: dict[tuple[int, str], Data] = {}

        def _get_graph(k: int, norm: str | None) -> Data:
            key = (k, norm)
            if key not in graph_cache:
                graph_cache[key] = build_graph_data(
                    data_dir, k_neighbors=k,
                    normalize=norm if norm != 'none' else None).to(self.device)
            return graph_cache[key]

        cfg = self.cfg

        def objective(trial: optuna.trial.Trial) -> float:
            model: nn.Module | None = None
            optimizer: optim.Optimizer | None = None
            scheduler: optim.lr_scheduler._LRScheduler | None = None
            writer: SummaryWriter | None = None
            try:
                params = suggest_hparams(trial, 'CellTypeGNN')
                k = params.k_neighbors
                data = _get_graph(k, params.normalize)

                model_kw = _model_kwargs_from_params(params, 'CellTypeGNN')
                model = build_model('CellTypeGNN', n_features, n_classes, **model_kw)

                criterion = build_criterion(params.loss, weight=weights,
                                            label_smoothing=params.label_smoothing,
                                            gamma=params.focal_gamma)
                optimizer, scheduler = build_optimizer(
                    model, params.lr, params.weight_decay, tune_epochs,
                    opt_cls=params.optimizer)
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

    # -- full tuning + final training ----------------------------------------

    def train_with_tuning(
        self,
        data_dir: str,
        n_features: int,
        n_classes: int,
        weights: torch.Tensor,
        n_trials: int = 15,
        tune_epochs: int = 30,
    ) -> tuple[float, str, dict[str, Any]]:
        best_params = self.run_hparam_search(
            data_dir, n_features, n_classes, weights,
            n_trials=n_trials, tune_epochs=tune_epochs)

        cfg = self.cfg
        bp: dict[str, Any] = best_params or {}
        lr: float = bp.get('lr', cfg['lr'])
        wd: float = bp.get('weight_decay', cfg['weight_decay'])
        opt_name: str = bp.get('optimizer', cfg.get('optimizer', 'adamw'))
        loss_name: str = bp.get('loss', cfg.get('loss', 'cross_entropy'))
        label_smoothing: float = bp.get('label_smoothing', cfg.get('label_smoothing', 0.1))
        focal_gamma: float = bp.get('focal_gamma', cfg.get('focal_gamma', 2.0))
        dropout: float = bp.get('dropout', cfg.get('dropout', 0.3))
        k_neighbors: int = bp.get('k_neighbors', cfg.get('k_neighbors', 15))
        normalize: str | None = bp.get('normalize', cfg.get('normalize'))
        if normalize == 'none':
            normalize = None

        data: Data = build_graph_data(data_dir, k_neighbors=k_neighbors,
                                      normalize=normalize).to(self.device)

        model_kw: dict[str, Any] = dict(dropout=dropout)
        for k in ('n_layers', 'hidden_dim'):
            if k in bp:
                model_kw[k] = bp[k]
        model: nn.Module = build_model('CellTypeGNN', n_features, n_classes, **model_kw)

        criterion: nn.Module = build_criterion(loss_name, weight=weights,
                                               label_smoothing=label_smoothing, gamma=focal_gamma)
        optimizer: optim.Optimizer
        scheduler: optim.lr_scheduler._LRScheduler
        optimizer, scheduler = build_optimizer(
            model, lr, wd, cfg['epochs'], opt_cls=opt_name)
        writer: SummaryWriter
        ckpt: str
        writer, ckpt = make_writer_and_ckpt(cfg, n_features)
        _save_model_kwargs(os.path.dirname(ckpt), model_kw)
        console.print(f'\nData: {data}')
        console.print(f'Training {cfg["epochs"]} epochs with best params on {self.device}...')
        print_header()
        best: float = train_graph(model, data, criterion, optimizer, scheduler,
                                  cfg['epochs'], writer, ckpt)
        console.print(f'\nBest validation accuracy: [bold green]{best:.4f}[/bold green]')
        return best, ckpt, bp

    # -- single run with saved hparams ---------------------------------------

    def train_single(
        self,
        data_dir: str,
        n_features: int,
        n_classes: int,
        weights: torch.Tensor,
        hp_dir: str = 'finalhyperparameter',
    ) -> tuple[float, str, dict[str, Any]]:
        """Single graph training run using saved best hyperparameters (no search).

        Returns (best_val_acc, ckpt_path, merged_params).
        """
        cfg = self.cfg
        saved: dict[str, Any] = load_hyperparameters(cfg['model'], hp_dir)
        bp: dict[str, Any] = {**saved}
        for k in ('epochs', 'batch_size', 'seed'):
            if k in cfg:
                bp[k] = cfg[k]

        lr: float = bp.get('lr', cfg.get('lr', 3e-4))
        wd: float = bp.get('weight_decay', cfg.get('weight_decay', 1e-5))
        opt_name: str = bp.get('optimizer', cfg.get('optimizer', 'adamw'))
        loss_name: str = bp.get('loss', cfg.get('loss', 'cross_entropy'))
        label_smoothing: float = bp.get('label_smoothing', cfg.get('label_smoothing', 0.1))
        focal_gamma: float = bp.get('focal_gamma', cfg.get('focal_gamma', 2.0))
        dropout: float = bp.get('dropout', cfg.get('dropout', 0.3))
        k_neighbors: int = int(bp.get('k_neighbors', cfg.get('k_neighbors', 15)))
        normalize: str | None = bp.get('normalize', cfg.get('normalize'))
        if normalize == 'none':
            normalize = None

        data: Data = build_graph_data(data_dir, k_neighbors=k_neighbors,
                                      normalize=normalize).to(self.device)

        model_kw: dict[str, Any] = dict(dropout=dropout)
        for k in ('n_layers', 'hidden_dim'):
            if k in bp:
                model_kw[k] = int(bp[k]) if isinstance(bp[k], float) and bp[k] == int(bp[k]) else bp[k]
        model: nn.Module = build_model('CellTypeGNN', n_features, n_classes, **model_kw)

        criterion: nn.Module = build_criterion(loss_name, weight=weights,
                                               label_smoothing=label_smoothing, gamma=focal_gamma)
        optimizer: optim.Optimizer
        scheduler: optim.lr_scheduler._LRScheduler
        optimizer, scheduler = build_optimizer(
            model, lr, wd, cfg['epochs'], opt_cls=opt_name)
        writer: SummaryWriter
        ckpt: str
        writer, ckpt = make_writer_and_ckpt(cfg, n_features)
        _save_model_kwargs(os.path.dirname(ckpt), model_kw)

        console.print(f'\nData: {data}')
        console.print(f'Training {cfg["epochs"]} epochs with fixed best params on {self.device}...')
        print_header()
        best: float = train_graph(model, data, criterion, optimizer, scheduler,
                                  cfg['epochs'], writer, ckpt)
        console.print(f'\nBest validation accuracy: [bold green]{best:.4f}[/bold green]')
        return best, ckpt, bp

    # -- evaluation -----------------------------------------------------------

    def evaluate(
        self,
        data: Data,
        ckpt_path: str,
        n_features: int,
        n_classes: int,
        class_names: list[str] | None = None,
    ) -> EvalMetrics:
        """Load best checkpoint and evaluate graph model on test mask with full metrics."""
        cfg = self.cfg
        saved_kw: dict[str, Any] = _load_model_kwargs(ckpt_path, model_name=cfg['model'])
        model: nn.Module = build_model(cfg['model'], n_features, n_classes, **saved_kw)

        model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        console.print(f'Loaded checkpoint: {ckpt_path}')

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)

        mask: torch.Tensor = data.test_mask
        y_pred: np.ndarray = logits[mask].argmax(1).cpu().numpy()
        y_true: np.ndarray = data.y[mask].cpu().numpy()

        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]

        save_dir: str = os.path.dirname(ckpt_path)
        raw_metrics: dict[str, Any] = _compute_metrics(y_true, y_pred, class_names, save_dir)
        return EvalMetrics(
            accuracy=raw_metrics['accuracy'],
            f1_macro=raw_metrics['f1_macro'],
            f1_weighted=raw_metrics['f1_weighted'],
            precision_macro=raw_metrics['precision_macro'],
            precision_weighted=raw_metrics['precision_weighted'],
            recall_macro=raw_metrics['recall_macro'],
            recall_weighted=raw_metrics['recall_weighted'],
            confusion_matrix=raw_metrics.get('confusion_matrix'),
        )


# ---------------------------------------------------------------------------
# Backward-compat module-level wrappers
# ---------------------------------------------------------------------------

def run_graph_hparam_search(
    cfg: ExperimentConfig | dict[str, Any],
    data_dir: str,
    n_features: int,
    n_classes: int,
    weights: torch.Tensor,
    n_trials: int = 15,
    tune_epochs: int = 30,
) -> dict[str, Any] | None:
    ecfg = ExperimentConfig(**cfg) if isinstance(cfg, dict) else cfg
    return GraphTrainer(ecfg).run_hparam_search(
        data_dir, n_features, n_classes, weights,
        n_trials=n_trials, tune_epochs=tune_epochs)


def train_graph_with_tuning(
    cfg: ExperimentConfig | dict[str, Any],
    data_dir: str,
    n_features: int,
    n_classes: int,
    weights: torch.Tensor,
    n_trials: int = 15,
    tune_epochs: int = 30,
) -> tuple[float, str, dict[str, Any]]:
    ecfg = ExperimentConfig(**cfg) if isinstance(cfg, dict) else cfg
    return GraphTrainer(ecfg).train_with_tuning(
        data_dir, n_features, n_classes, weights,
        n_trials=n_trials, tune_epochs=tune_epochs)


def train_graph_single(
    cfg: ExperimentConfig | dict[str, Any],
    data_dir: str,
    n_features: int,
    n_classes: int,
    weights: torch.Tensor,
    hp_dir: str = 'finalhyperparameter',
) -> tuple[float, str, dict[str, Any]]:
    """Single graph training run using saved best hyperparameters (no search).

    Returns (best_val_acc, ckpt_path, merged_params).
    """
    ecfg = ExperimentConfig(**cfg) if isinstance(cfg, dict) else cfg
    return GraphTrainer(ecfg).train_single(
        data_dir, n_features, n_classes, weights, hp_dir=hp_dir)


def evaluate_graph(
    cfg: ExperimentConfig | dict[str, Any],
    data: Data,
    ckpt_path: str,
    n_features: int,
    n_classes: int,
    class_names: list[str] | None = None,
) -> EvalMetrics:
    """Load best checkpoint and evaluate graph model on test mask with full metrics."""
    ecfg = ExperimentConfig(**cfg) if isinstance(cfg, dict) else cfg
    return GraphTrainer(ecfg).evaluate(
        data, ckpt_path, n_features, n_classes, class_names=class_names)
