import numpy as np

from allen_brain.models import train as T
from allen_brain.models.CellTypeGNN import build_graph_data, masked_class_weights

SEED = 42
BATCH_SIZE = 256
N_HVG = 0
DATA_DIR = 'data/10x'
K_NEIGHBORS = 10
N_TRIALS = 8
TUNE_EPOCHS = 20

NORMALIZE = 'log+standard'  # None, 'log', 'standard', or 'log+standard'

COFIG = {
    'model': 'CellTypeGNN',
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'n_hvg': N_HVG,
    'device': str(T.DEVICE),
    'optimizer': 'adamw',
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'epochs': 200,
    'loss': 'cross_entropy',
    'label_smoothing': 0.1,
    'k_neighbors': K_NEIGHBORS,
    'normalize': NORMALIZE,
}


def main():
    # Load data once to get n_features, n_classes, weights for the search
    data = build_graph_data(DATA_DIR, k_neighbors=K_NEIGHBORS,
                            normalize=NORMALIZE).to(T.DEVICE)
    n_classes = int(data.y.max().item()) + 1
    class_names = list(np.load(f'{DATA_DIR}/class_names.npy', allow_pickle=True))
    weights = masked_class_weights(data.y, data.train_mask, n_classes)
    n_features = data.x.shape[1]
    del data  # search will rebuild graphs lazily per k_neighbors

    best_acc, ckpt, best_params = T.train_graph_with_tuning(
        COFIG, DATA_DIR, n_features, n_classes, weights,
        n_trials=N_TRIALS, tune_epochs=TUNE_EPOCHS)
    T.save_hyperparameters('CellTypeGNN', best_params, COFIG)

    # Rebuild graph with best k + normalize for evaluation
    best_k = best_params.get('k_neighbors', K_NEIGHBORS)
    best_normalize = best_params.get('normalize', NORMALIZE)
    if best_normalize == 'none':
        best_normalize = None
    eval_data = build_graph_data(DATA_DIR, k_neighbors=best_k,
                                 normalize=best_normalize).to(T.DEVICE)
    metrics = T.evaluate_graph(COFIG, eval_data, ckpt, n_features, n_classes,
                               class_names=class_names)
    T.append_results_csv('GNN', metrics)


if __name__ == '__main__':
    main()
