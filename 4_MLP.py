from allen_brain.models import train as T

SEED = 42
BATCH_SIZE = 8192
N_HVG = 0
DATA_DIR = 'data/10x'
TUNE_EPOCHS = 15

NORMALIZE = 'log+standard'  # None, 'log', 'standard', or 'log+standard'

COFIG = {
    'model': 'CellTypeMLP',
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'n_hvg': N_HVG,
    'device': str(T.DEVICE),
    'optimizer': 'adamw',
    'lr': 3e-4,
    'weight_decay': 1e-6,
    'epochs': 20,
    'loss': 'cross_entropy',
    'label_smoothing': 0.1,
    'normalize': NORMALIZE,
}

# Three hardcoded configs based on known-good hyperparameters (95.4% accuracy).
GRID = [
    {   # Config 1: exact known-good params
        'lr': 4.335e-5, 'weight_decay': 5.337e-7,
        'dropout': 0.133, 'label_smoothing': 0.061,
        'optimizer': 'adamw', 'loss': 'cross_entropy', 'focal_gamma': 2.0,
        'normalize': 'log+standard',
        'n_layers': 2, 'hidden_dim': 512,
    },
    {   # Config 2: higher lr + more regularization
        'lr': 8e-5, 'weight_decay': 1e-6,
        'dropout': 0.18, 'label_smoothing': 0.08,
        'optimizer': 'adamw', 'loss': 'cross_entropy', 'focal_gamma': 2.0,
        'normalize': 'log+standard',
        'n_layers': 2, 'hidden_dim': 512,
    },
    {   # Config 3: deeper network + conservative lr
        'lr': 3e-5, 'weight_decay': 1e-6,
        'dropout': 0.15, 'label_smoothing': 0.05,
        'optimizer': 'adamw', 'loss': 'cross_entropy', 'focal_gamma': 2.0,
        'normalize': 'log+standard',
        'n_layers': 3, 'hidden_dim': 512,
    },
]


def main():
    best_acc, ckpt, best_params = T.train_with_grid(
        COFIG, DATA_DIR, squeeze_channel=True,
        grid=GRID, tune_epochs=TUNE_EPOCHS)
    T.save_hyperparameters('CellTypeMLP', best_params, COFIG)
    metrics = T.evaluate(COFIG, DATA_DIR, ckpt, squeeze_channel=True)
    T.append_results_csv('MLP', metrics)


if __name__ == '__main__':
    main()
