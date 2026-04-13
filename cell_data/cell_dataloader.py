import torch
from torch.utils.data import DataLoader

from .cell_dataset import make_datasets


def get_dataloaders(data_dir: str, batch_size: int = 256, num_workers: int = 0,
                    **preprocess_kwargs) -> dict:
    """Load raw data, preprocess, and create DataLoaders.

    Returns dict with 'train'/'val'/'test' DataLoaders plus metadata keys
    ('class_names', 'gene_names', 'scaler', 'n_classes').
    """
    result = make_datasets(data_dir, **preprocess_kwargs)

    pin = torch.cuda.is_available()
    for name in ('train', 'val', 'test'):
        result[name] = DataLoader(
            result[name],
            batch_size=batch_size,
            shuffle=(name == 'train'),
            num_workers=num_workers,
            pin_memory=pin,
            labelencoder = result[name].labelencoder,  # type: ignore
        )

    return result
