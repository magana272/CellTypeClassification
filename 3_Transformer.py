from allen_brain.models import train as T
from allen_brain.models.config import ExperimentConfig
from allen_brain.models.CellTypeAttention import build_pathway_mask
from allen_brain.cell_data.cell_dataset import make_dataset

DATA_DIR = 'data/10x'
GMT_PATH = 'data/reactome.gmt'
MAX_PATHWAYS = 300
MIN_PATHWAY_OVERLAP = 5
MAX_GENE_SET_SIZE = 300

cfg = ExperimentConfig(
    model='CellTypeTOSICA',
    seed=42,
    batch_size=4096,
    accumulation_steps=1,
    n_hvg=0,
    optimizer='adamw',
    lr=3e-3,
    weight_decay=1e-6,
    epochs=20,
    loss='cross_entropy',
    label_smoothing=0.1,
    normalize=None,
)


def _build_pathway_kwargs() -> dict:
    """Build extra_model_kwargs for TOSICA from training gene names."""
    ds = make_dataset(DATA_DIR, split='train')
    mask, n_pathways = build_pathway_mask(
        [str(g) for g in ds.gene_names],
        gmt_path=GMT_PATH, min_overlap=MIN_PATHWAY_OVERLAP,
        max_pathways=MAX_PATHWAYS, max_gene_set_size=MAX_GENE_SET_SIZE)
    return dict(mask=mask, n_pathways=n_pathways)


def main() -> None:
    extra_kw = _build_pathway_kwargs()
    trainer = T.Trainer(cfg)
    best_acc, ckpt, best_params = trainer.train_single(
        DATA_DIR, squeeze_channel=True,
        extra_model_kwargs=extra_kw)
    T.save_hyperparameters('CellTypeTOSICA', best_params, cfg)
    metrics = trainer.evaluate(DATA_DIR, ckpt, squeeze_channel=True)
    T.append_results_csv('Transformer', metrics)


if __name__ == '__main__':
    main()
