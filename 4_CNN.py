import torch
from torch import nn, optim

from allen_brain.models import train as T

torch.set_float32_matmul_precision('high')
SEED = 42
BATCH_SIZE = 4096//4
N_HVG = 2000
DATA_DIR = 'data/10x'
N_TRIALS = 2
TUNE_EPOCHS = 5
TUNE_BATCH_SIZE = 64

COFIG = {
    'model': 'CellTypeCNN',
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
    T.train_with_tuning(COFIG, DATA_DIR, squeeze_channel=False,
                        n_trials=N_TRIALS, tune_epochs=TUNE_EPOCHS,
                        tune_batch_size=TUNE_BATCH_SIZE)


if __name__ == '__main__':
    main()
