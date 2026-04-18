import numpy as np

from allen_brain.models import train as T
from allen_brain.models.config import ExperimentConfig
from allen_brain.models.gnn_train import GraphTrainer
from allen_brain.models.CellTypeGNN import GraphBuilder

DATA_DIR = 'data/10x'
K_NEIGHBORS = 10

cfg = ExperimentConfig(
    model='CellTypeGNN',
    seed=42,
    batch_size=256,
    n_hvg=0,
    optimizer='adamw',
    lr=3e-4,
    weight_decay=1e-5,
    epochs=200,
    loss='cross_entropy',
    label_smoothing=0.1,
    k_neighbors=K_NEIGHBORS,
    normalize='log+standard',
)


def main() -> None:
    gb = GraphBuilder(k_neighbors=K_NEIGHBORS, normalize='log+standard')
    data = gb.build_graph_data(DATA_DIR).to(T.DEVICE)
    n_classes: int = int(data.y.max().item()) + 1
    class_names: list[str] = list(np.load(f'{DATA_DIR}/class_names.npy', allow_pickle=True))
    weights = GraphBuilder.masked_class_weights(data.y, data.train_mask, n_classes)
    n_features: int = data.x.shape[1]
    del data

    trainer = GraphTrainer(cfg)
    best_acc, ckpt, best_params = trainer.train_single(
        DATA_DIR, n_features, n_classes, weights)
    T.save_hyperparameters('CellTypeGNN', best_params, cfg)

    best_k: int = int(best_params.get('k_neighbors', K_NEIGHBORS))
    best_normalize: str | None = best_params.get('normalize', 'log+standard')
    if best_normalize == 'none':
        best_normalize = None
    eval_data = GraphBuilder(k_neighbors=best_k,
                             normalize=best_normalize).build_graph_data(DATA_DIR).to(T.DEVICE)
    metrics = trainer.evaluate(eval_data, ckpt, n_features, n_classes,
                               class_names=class_names)
    T.append_results_csv('GNN', metrics)


if __name__ == '__main__':
    main()
