import torch
from torch.utils.data import DataLoader
from allen_brain.cell_data.cell_dataset import make_dataset


def get_data_loader(data_dir: str, split='train', batch_size: int = 256, num_workers: int = 0,
                    **preprocess_kwargs) -> DataLoader:
    """Load raw data, preprocess, and create DataLoader."""
    dataset = make_dataset(data_dir, split=split, **preprocess_kwargs)
    pin = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=pin,
    )


# def get_dataloaders(data_dir: str, batch_size: int = 256, num_workers: int = 0,
#                     **preprocess_kwargs) -> dict:
#     """Load raw data, preprocess, and create DataLoaders.

#     Returns dict with 'train'/'val'/'test' DataLoaders plus metadata keys
#     ('class_names', 'gene_names', 'scaler', 'n_classes').
#     """
#     result = make_dataset(data_dir, **preprocess_kwargs)

#     pin = torch.cuda.is_available()
#     for name in ('train', 'val', 'test'):
#         result[name] = DataLoader(
#             result[name],
#             batch_size=batch_size,
#             shuffle=(name == 'train'),
#             num_workers=num_workers,
#             pin_memory=pin,
#         )

#     return result