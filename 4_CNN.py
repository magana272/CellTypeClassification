from allen_brain.models import train as T

SEED = 42
BATCH_SIZE = 16384
ACCUMULATION_STEPS = 1  # A100 fits 16384 directly; no accumulation overhead
N_HVG = 2000
N_HVG_RANGE = (1000, 3000, 500)
DATA_DIR = 'data/10x'
N_TRIALS = 8
TUNE_EPOCHS = 10
FINAL_EPOCHS = 30


NORMALIZE = 'log+standard'  # None, 'log', 'standard', or 'log+standard'

COFIG = {
    'model': 'CellTypeCNN',
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'accumulation_steps': ACCUMULATION_STEPS,
    'n_hvg': N_HVG,
    'device': str(T.DEVICE),
    'optimizer': 'adamw',
    'lr': 3e-4,
    'weight_decay': 1e-6,
    'epochs': FINAL_EPOCHS,
    'loss': 'cross_entropy',
    'label_smoothing': 0.1,
    'normalize': NORMALIZE,
}


def main():
    best_acc, ckpt, best_params = T.train_with_tuning(
        COFIG, DATA_DIR, squeeze_channel=False,
        n_trials=N_TRIALS, tune_epochs=TUNE_EPOCHS,
        n_hvg_range=N_HVG_RANGE)
    T.save_hyperparameters('CellTypeCNN', best_params, COFIG)
    metrics = T.evaluate(COFIG, DATA_DIR, ckpt, squeeze_channel=False)
    T.append_results_csv('CNN', metrics)


if __name__ == '__main__':
    main()
