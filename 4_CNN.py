from allen_brain.models import train as T

SEED = 42
BATCH_SIZE = 512 * 2
N_HVG = 2000
N_HVG_RANGE = (500, 5000, 500)
DATA_DIR = 'data/10x'
N_TRIALS = 5
TUNE_EPOCHS = 30

COFIG = {
    'model': 'CellTypeCNN',
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'n_hvg': N_HVG,
    'device': str(T.DEVICE),
    'optimizer': 'adamw',
    'lr': 3e-4,
    'weight_decay': 1e-6,
    'epochs': TUNE_EPOCHS,
    'loss': 'cross_entropy',
    'label_smoothing': 0.1,
}


def main():
    best_acc, ckpt = T.train_with_tuning(
        COFIG, DATA_DIR, squeeze_channel=False,
        n_trials=N_TRIALS, tune_epochs=TUNE_EPOCHS,
        n_hvg_range=N_HVG_RANGE)
    T.evaluate(COFIG, DATA_DIR, ckpt, squeeze_channel=False)


if __name__ == '__main__':
    main()
