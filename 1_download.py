"""Download all datasets and process into train/val/test .npy splits."""
from allen_brain.cell_data.cell_download import download_data
from allen_brain.cell_data.cell_load import (
    load_10x, load_smartseq, load_pbmc, load_pancreas,
    load_tabula_muris, load_lung,
)


def main():
    # Download raw files (CSVs + h5ad)
    download_data()

    load_10x()
    load_smartseq()
    load_pbmc()        
    load_pancreas()
    #load_tabula_muris()
    load_lung()


if __name__ == '__main__':
    main()
