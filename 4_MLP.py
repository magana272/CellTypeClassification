from allen_brain.models import train as T

SEED = 42
BATCH_SIZE = 24576 * 2
N_HVG = 0
DATA_DIR = 'data/10x'
N_TRIALS = 10
TUNE_EPOCHS = 50

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
}


def main():
    best_acc, ckpt, best_params = T.train_with_tuning(
        COFIG, DATA_DIR, squeeze_channel=True,
        n_trials=N_TRIALS, tune_epochs=TUNE_EPOCHS)
    T.save_hyperparameters('CellTypeMLP', best_params, COFIG)
    metrics = T.evaluate(COFIG, DATA_DIR, ckpt, squeeze_channel=True)
    T.append_results_csv('MLP', metrics)


if __name__ == '__main__':
    main()
