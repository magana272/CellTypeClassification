from allen_brain.models import train as T
from allen_brain.models.CellTypeAttention import build_pathway_mask
from allen_brain.cell_data.cell_dataset import make_dataset

SEED = 42
BATCH_SIZE = 24576
N_HVG = 0
DATA_DIR = 'data/10x'
GMT_PATH = 'data/reactome.gmt'
MAX_PATHWAYS = 500
MIN_PATHWAY_OVERLAP = 5
MAX_GENE_SET_SIZE = 500
N_TRIALS = 30
TUNE_EPOCHS = 50

NORMALIZE = 'log+standard'  # None, 'log', 'standard', or 'log+standard'

COFIG = {
    'model': 'CellTypeTOSICA',
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


def _build_pathway_kwargs():
    """Build extra_model_kwargs for TOSICA from training gene names."""
    ds = make_dataset(DATA_DIR, split='train')
    mask, n_pathways = build_pathway_mask(
        [str(g) for g in ds.gene_names],
        gmt_path=GMT_PATH, min_overlap=MIN_PATHWAY_OVERLAP,
        max_pathways=MAX_PATHWAYS, max_gene_set_size=MAX_GENE_SET_SIZE)
    return dict(mask=mask, n_pathways=n_pathways)


def main():
    extra_kw = _build_pathway_kwargs()
    best_acc, ckpt, best_params = T.train_with_tuning(
        COFIG, DATA_DIR, squeeze_channel=True,
        n_trials=N_TRIALS, tune_epochs=TUNE_EPOCHS,
        extra_model_kwargs=extra_kw)
    T.save_hyperparameters('CellTypeTOSICA', best_params, COFIG)
    metrics = T.evaluate(COFIG, DATA_DIR, ckpt, squeeze_channel=True)
    T.append_results_csv('Transformer', metrics)


if __name__ == '__main__':
    main()
