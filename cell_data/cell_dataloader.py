import torch
from torch.utils.data import DataLoader

from cell_data.cell_dataset import GeneExpressionDataset
from cell_data.load import CellSplit


def get_dataloaders(
    split: CellSplit,
    batch_size: int = 256,
    num_workers: int = 0,
    transform=None,
    target_transform=None,
) -> dict[str, DataLoader]:
    """Build {train, val, test} DataLoaders from a CellSplit.

    Expression rows are taken from ``split.X`` using the split's ``idx_*``
    arrays, so no external matrix needs to be threaded through.
    """
    specs = (
        ('train', split.idx_train, split.y_train, True),
        ('val',   split.idx_val,   split.y_val,   False),
        ('test',  split.idx_test,  split.y_test,  False),
    )
    pin = torch.cuda.is_available()
    loaders: dict[str, DataLoader] = {}
    for name, idx, y, shuffle in specs:
        ds = GeneExpressionDataset(
            split.X[idx], y,
            transform=transform,
            target_transform=target_transform,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin,
        )
    return loaders

