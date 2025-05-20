import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap

def parse_args():
    parser = argparse.ArgumentParser(description="Detailed Visualization of Embeddings")
    parser.add_argument("--original_features", required=True, help="Path to original node features (numpy format)")
    parser.add_argument("--graphsage_embeddings", required=True, help="Path to GraphSAGE embeddings file (numpy format)")
    parser.add_argument("--method", choices=['tsne', 'umap'], default="umap", help="Dimensionality reduction method")
    parser.add_argument("--output_dir", default=".", help="Output directory for plots")
    return parser.parse_args()


def visualize_embeddings(original_features, embeddings, method='umap', output_dir='.'): 
    if method == 'tsne':
        reducer = TSNE(random_state=42)
    else:
        reducer = umap.UMAP(random_state=42)

    original_reduced = reducer.fit_transform(original_features)
    embeddings_reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(original_features[:, 0], original_features[:, 1], s=5, cmap='Spectral')
    plt.title("Original Node Features")
    plt.xlabel('Dimension 1')
    plt.ylabel("Dimension 2")

    plt.subplot(1, 2, 2)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=5, cmap='Spectral')
    plt.title("GraphSAGE Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/embeddings_comparison_{method}.png")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    import pandas as pd  # <-- explicitly import pandas
    original_features = pd.read_csv(args.original_features, index_col=0).values  # <-- explicitly load CSV
    
    embeddings = np.load(args.graphsage_embeddings)

    visualize_embeddings(
        original_features, 
        embeddings, 
        method=args.method, 
        output_dir=args.output_dir
    )

