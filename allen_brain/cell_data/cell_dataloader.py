import torch
from torch.utils.data import DataLoader
from allen_brain.cell_data.cell_dataset import make_dataset


def get_data_loader(data_dir: str, split='train', batch_size: int = 256, num_workers: int = 0,
                    **preprocess_kwargs) -> DataLoader:
    """Load raw data, preprocess, and create DataLoader."""
    dataset = make_dataset(data_dir, split=split, **preprocess_kwargs)
    # pin = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=False,
    )

