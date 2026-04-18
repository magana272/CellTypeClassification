from allen_brain.models import train as T
from allen_brain.models.config import ExperimentConfig

DATA_DIR = 'data/10x'

cfg = ExperimentConfig(
    model='CellTypeCNN',
    seed=42,
    batch_size=16384,
    accumulation_steps=1,
    n_hvg=2000,
    optimizer='adamw',
    lr=3e-4,
    weight_decay=1e-6,
    epochs=30,
    loss='cross_entropy',
    label_smoothing=0.1,
    normalize='log+standard',
)


def main() -> None:
    trainer = T.Trainer(cfg)
    best_acc, ckpt, best_params = trainer.train_single(
        DATA_DIR, squeeze_channel=False)
    T.save_hyperparameters('CellTypeCNN', best_params, cfg)
    metrics = trainer.evaluate(DATA_DIR, ckpt, squeeze_channel=False)
    T.append_results_csv('CNN', metrics)


if __name__ == '__main__':
    main()
