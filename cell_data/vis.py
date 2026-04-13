import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cell_data.cell_dataset as cell_dataset
from collections import Counter
def plot_class_distribution(ds: cell_dataset.GeneExpressionDataset):

    train_labels = ds.labelencoder.inverse_transform(ds.y)
    label_counts = Counter(train_labels)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(label_counts)), label_counts.values,
                color=sns.color_palette('tab20', len(label_counts)))
    ax.set_xticks(range(len(label_counts)))
    ax.set_xticklabels(label_counts.index, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Training Set — Cell Type Distribution')
    for bar, count in zip(bars, label_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig('fig_class_distribution.png', dpi=150)
    plt.show()
    print('Saved fig_class_distribution.png')
    
