from torch import nn, optim

from allen_brain.models import train as T


SEED = 42
BATCH_SIZE = 4096
N_HVG = 2000
DATA_DIR = 'data/smartseq'
N_TRIALS = 15
TUNE_EPOCHS = 5

COFIG = {
    'model': 'CellTypeMLP',
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'n_hvg': N_HVG,
    'device': str(T.DEVICE),
    'optimizer': optim.AdamW,
    'lr': 3e-4,
    'weight_decay': 1e-6,
    'epochs': 20,
    'loss': nn.CrossEntropyLoss,
}


def main():
    T.train_with_tuning(COFIG, DATA_DIR, squeeze_channel=True,
                        n_trials=N_TRIALS, tune_epochs=TUNE_EPOCHS)


if __name__ == '__main__':
    main()
