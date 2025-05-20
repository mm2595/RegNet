import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd
import os
from regnet.models.regnet import RegNet

try:
    import umap
except ImportError:
    raise ImportError("UMAP is not installed. Install it using `pip install umap-learn`.")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VAE latent embeddings and reconstructions")
    parser.add_argument("--vae_embeddings", required=True, help="Path to saved VAE embeddings (numpy array or CSV)")
    parser.add_argument("--model_path", help="Path to saved model for reconstruction visualization")
    parser.add_argument("--expression_data", help="Path to original expression data for reconstruction comparison")
    parser.add_argument("--method", default='umap', choices=['tsne', 'umap'], help="Dimensionality reduction method")
    parser.add_argument("--output_dir", default=".", help="Output directory for embedding visualization")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run model on")
    parser.add_argument("--num_genes", type=int, default=10, help="Number of genes to visualize for reconstruction")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for loading model")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension for loading model")
    return parser.parse_args()

def visualize_vae(embeddings, method='umap', output_file='vae_embeddings.png'):
    """Visualize the VAE latent space in 2D using UMAP or t-SNE"""
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(random_state=42)

    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=5, c='blue')  
    plt.title(f"VAE Embeddings Visualized by {method.upper()}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def visualize_reconstructions(model, expr_data, output_file, num_genes=10, device="cpu"):
    """Visualize original vs. reconstructed gene expression profiles"""
    model.eval()
    
    # Create a dummy edge_index since we just need to pass it to the model
    # In a real application, you'd use the actual graph structure
    edge_index = torch.zeros((2, 1), dtype=torch.long, device=device)
    
    # Process a subset of genes for visualization
    gene_names = expr_data.index[:num_genes].tolist()
    x = torch.tensor(expr_data.values[:num_genes], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Get reconstructions
        x_recon, _, _, _, _, _ = model.reconstruct(x, edge_index)
        
    # Convert to numpy for plotting
    x_np = x.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()
    
    # Plot original vs reconstructed expressions
    fig, axes = plt.subplots(num_genes, 1, figsize=(12, 2*num_genes))
    
    for i, gene in enumerate(gene_names):
        ax = axes[i] if num_genes > 1 else axes
        
        # Plot original
        ax.plot(x_np[i], 'b-', label='Original', linewidth=2)
        
        # Plot reconstruction
        ax.plot(x_recon_np[i], 'r--', label='Reconstructed', linewidth=2)
        
        ax.set_title(f"Gene: {gene}")
        ax.set_ylabel("Expression")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    args = parse_args()
    
    # Load embeddings
    if args.vae_embeddings.endswith('.csv'):
        emb_df = pd.read_csv(args.vae_embeddings, index_col=0)
        embeddings = emb_df.values
    else:
        embeddings = np.load(args.vae_embeddings)

    # Visualize latent space
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/vae_embeddings_{args.method}.png"
    visualize_vae(embeddings, method=args.method, output_file=output_file)
    
    # Visualize reconstructions if model and expression data are provided
    if args.model_path and args.expression_data:
        # Load expression data
        expr_df = pd.read_csv(args.expression_data, index_col=0)
        
        # Load model
        device = torch.device(args.device)
        model = RegNet(expr_df.shape[1], args.hidden_dim, args.latent_dim).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Visualize reconstructions
        recon_output_file = f"{args.output_dir}/vae_reconstructions.png"
        visualize_reconstructions(model, expr_df, recon_output_file, 
                                 num_genes=args.num_genes, device=device)
        
        print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
