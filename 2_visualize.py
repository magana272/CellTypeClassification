"""
3_visualize.py
Visualize both 10x and SmartSeq datasets.
Generates: class distribution, PCA, UMAP, heatmap, violin, and CV^2 plots.
All figures are saved under figures/*.png.
"""
import os

from rich.console import Console
from rich.panel import Panel

from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.cell_data.cell_load import ALL_DATASETS
from allen_brain.cell_data.cell_vis import DatasetVisualizer

SEED = 42
console = Console()

DATASETS = [(info['dir'], name) for name, info in ALL_DATASETS.items()]


def run_visualizations(data_dir: str, tag: str):
    """Run the full visualization suite for one dataset."""
    console.print(Panel(
        f"[bold]{tag.upper()}[/bold]  ·  {data_dir}",
        title="Dataset", border_style="cyan", expand=False,
    ))

    train_ds = make_dataset(data_dir, split="train")
    if train_ds is None:
        console.print(f"  [yellow]No train split found for {tag}[/yellow], loading full matrix ...")
        train_ds = make_dataset(data_dir)
    if train_ds is None:
        console.print(f"  [bold red]\\[SKIP][/bold red] No data found in {data_dir}")
        return

    gene_names = train_ds.gene_names
    fig_dir = "figures"
    vis = DatasetVisualizer(train_ds, fig_dir=fig_dir, seed=SEED)

    vis.plot_class_distribution(
        save_path=os.path.join(fig_dir, f"{tag}_class_distribution.png"),
    )

    pca, X_pca = vis.plot_pca(
        n_components=20,
        save_path=fig_dir,
        file_name=f"{tag}_pca.png",
    )

    vis.plot_umap(
        X_pca,
        max_cells=6000,
        save_path=os.path.join(fig_dir, f"{tag}_umap.png"),
    )

    vis.plot_heatmap(
        gene_names=gene_names,
        n_genes=20,
        save_path=os.path.join(fig_dir, f"{tag}_heatmap.png"),
    )

    vis.plot_violin(
        gene_names=gene_names,
        top_n=6,
        save_path=os.path.join(fig_dir, f"{tag}_violin.png"),
    )

    vis.plot_cv2(
        gene_names=gene_names,
        n_top=1000,
        save_path=os.path.join(fig_dir, f"{tag}_cv2.png"),
    )


def main():
    for data_dir, tag in DATASETS:
        run_visualizations(data_dir, tag)
    console.print("\nAll visualizations complete.")


if __name__ == "__main__":
    main()
