import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


def parse_args():
    p = argparse.ArgumentParser("RegNet visualisations")
    p.add_argument("--pretrain_dir", required=True,
                   help="Path to the *pretrain* output directory (contains embeddings CSVs)")
    p.add_argument("--expression_csv", required=False,
                   help="Expression matrix (for raw-feature comparison)")
    p.add_argument("--tf_file", required=False,
                   help="Optional TF list to colour points in embedding plot")
    p.add_argument("--method", choices=["umap", "tsne"], default="umap")
    p.add_argument("--n_top_edges", type=int, default=500,
                   help="Nr of top-confidence edges for prob-variance scatter")
    p.add_argument("--plots", nargs="*", default=["basic", "shrinkage"],
                   help="Which plot groups to generate: basic, shrinkage")
    p.add_argument("--debug", action="store_true", help="Enable debug mode")
    return p.parse_args()


def load_embeddings(path_csv):
    df = pd.read_csv(path_csv, index_col=0)
    return df.index.tolist(), df.values


def reduce_dim(mat, method="umap"):
    if method == "umap" and _HAS_UMAP:
        reducer = umap.UMAP(random_state=42)
    elif method == "tsne":
        reducer = TSNE(random_state=42)
    else:
        reducer = TSNE(random_state=42)
    return reducer.fit_transform(mat)


def plot_vae_embedding(genes, vae_emb, tf_genes, out_png, method="umap"):
    coords = reduce_dim(vae_emb, method)
    plt.figure(figsize=(8, 6))
    if tf_genes:
        is_tf = [g in tf_genes for g in genes]
        palette = {True: "tab:red", False: "tab:blue"}
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=is_tf, s=10,
                        palette=palette, legend=False, alpha=0.7)
    else:
        plt.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.7)
    plt.title(f"{method.upper()} of VAE latent means")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.tight_layout(); plt.savefig(out_png, dpi=300)
    plt.close()


def plot_gate_distribution(gate_csv, out_png):
    df = pd.read_csv(gate_csv, index_col=0)
    gate_mean = df.values.mean(1)
    plt.figure(figsize=(6, 4))
    sns.histplot(gate_mean, bins=50, kde=True)
    plt.xlabel("Mean gate value")
    plt.ylabel("# genes")
    plt.title("Distribution of gate coefficients")
    plt.tight_layout(); plt.savefig(out_png, dpi=300)
    plt.close()


def scatter_prob_variance(pred_csv, out_png, n_top=500):
    df = pd.read_csv(pred_csv)
    df_sort = df.sort_values("Prediction", ascending=False).head(n_top)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="Prediction", y="LogitVar", data=df_sort, alpha=0.6)
    plt.xlabel("Predicted probability")
    plt.ylabel("Logit variance")
    plt.title(f"Top {n_top} edge predictions vs uncertainty")
    plt.tight_layout(); plt.savefig(out_png, dpi=300)
    plt.close()


def safe_detach(tensor):
    """Safely detach a tensor (even if it doesn't require grad)"""
    if tensor.requires_grad:
        return tensor.detach()
    return tensor


def main():
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logging.info(f"Starting visualization with plots: {args.plots}")
    out_dir = args.pretrain_dir  # save figs alongside other outputs

    # --- embedding plot -------------------------------------------------- #
    try:
        gene_names, vae_emb = load_embeddings(os.path.join(args.pretrain_dir, "vae_embeddings.csv"))
        tf_set = set()
        if args.tf_file and os.path.isfile(args.tf_file):
            logging.info(f"Loading TF file: {args.tf_file}")
            tf_df = pd.read_csv(args.tf_file)
            if "TF" in tf_df.columns:
                tf_set = set(tf_df["TF"].astype(str))
                logging.info(f"Found {len(tf_set)} TFs")
            else:
                logging.warning(f"TF file does not contain 'TF' column. Columns: {tf_df.columns}")
        plot_vae_embedding(gene_names, vae_emb, tf_set,
                           os.path.join(out_dir, f"vae_{args.method}.png"), method=args.method)
        logging.info(f"VAE embedding plot created: vae_{args.method}.png")
    except Exception as e:
        logging.error(f"Error in VAE embedding plot: {e}")
        logging.debug(traceback.format_exc())

    # --- gate distribution ---------------------------------------------- #
    gate_csv = os.path.join(args.pretrain_dir, "gate_values.csv")
    if os.path.isfile(gate_csv):
        try:
            plot_gate_distribution(gate_csv, os.path.join(out_dir, "gate_distribution.png"))
            logging.info("Gate distribution plot created")
        except Exception as e:
            logging.error(f"Error in gate distribution plot: {e}")
            logging.debug(traceback.format_exc())
    else:
        logging.warning(f"Gate values CSV not found: {gate_csv}")

    # --- prob vs variance ------------------------------------------------ #
    pred_csv = os.path.join(args.pretrain_dir, "predictions_with_gene_names.csv")
    if os.path.isfile(pred_csv):
        try:
            scatter_prob_variance(pred_csv, os.path.join(out_dir, "prob_vs_variance.png"), args.n_top_edges)
            logging.info("Probability vs variance plot created")
        except Exception as e:
            logging.error(f"Error in prob vs variance plot: {e}")
            logging.debug(traceback.format_exc())
    else:
        logging.warning(f"Predictions CSV not found: {pred_csv}")

    # --- shrinkage plot -------------------------------------------------- #
    if "shrinkage" in args.plots:
        try:
            logging.info("Starting shrinkage plot")
            try:
                import torch
                logging.info("PyTorch imported successfully")
            except ImportError:
                logging.error("PyTorch not available - shrinkage plot requires PyTorch")
                raise
            
            try:
                from torch_scatter import scatter_mean
                logging.info("torch_scatter imported successfully")
            except ImportError:
                logging.error("torch_scatter not available - required for shrinkage plot")
                raise
            
            expr_df = None
            if args.expression_csv and os.path.isfile(args.expression_csv):
                logging.info(f"Loading expression data from {args.expression_csv}")
                expr_df = pd.read_csv(args.expression_csv, index_col=0)
            else:
                # fall back to expression matrix located two dirs up
                cand = os.path.join(out_dir, "..", "..", "BL--ExpressionData.csv")
                if os.path.isfile(cand):
                    logging.info(f"Using fallback expression data from {cand}")
                    expr_df = pd.read_csv(cand, index_col=0)
                else:
                    logging.warning(f"Expression data not found at {cand}")
            
            if expr_df is not None:
                logging.info(f"Expression data loaded: {expr_df.shape}")
                # Convert expression data to tensor
                x = torch.tensor(expr_df.values, dtype=torch.float32)
                logging.info(f"Expression tensor created: {x.shape}")
                
                # Load prediction data
                logging.info(f"Loading predictions from {pred_csv}")
                df_pred = pd.read_csv(pred_csv)
                logging.info(f"Prediction dataframe: {df_pred.shape}, columns: {df_pred.columns}")
                
                # Create gene-to-index mapping
                gene_to_idx = {g: i for i, g in enumerate(expr_df.index)}
                logging.info(f"Created gene to index mapping for {len(gene_to_idx)} genes")
                
                # Helper to map identifier to expression index
                def id_to_expr_idx(id_val):
                    """Map a TF/Target identifier to expression-matrix index.
                    The prediction CSV stores gene symbols, which match the
                    expression DataFrame index.  Any identifier that is not
                    found is ignored (function returns None)."""
                    return gene_to_idx.get(str(id_val), None)
                
                # Map prediction gene identifiers to expression indices (no fall-back to random)
                rows, cols = [], []
                for tf_id, tgt_id in zip(df_pred["TF_gene"], df_pred["Target_gene"]):
                    r_idx = id_to_expr_idx(tf_id)
                    c_idx = id_to_expr_idx(tgt_id)
                    if r_idx is not None and c_idx is not None:
                        rows.append(r_idx)
                        cols.append(c_idx)
                logging.info(f"Valid edges after mapping: {len(rows)} / {len(df_pred)}")
                if len(rows) == 0:
                    raise RuntimeError("Could not map any TF–Target pairs to expression indices; check mapping tables and identifier formats.")
                
                # Convert to tensor and create edge_index
                row_idx = torch.tensor(rows, dtype=torch.long)
                col_idx = torch.tensor(cols, dtype=torch.long)
                edge_index = torch.stack([row_idx, col_idx], dim=0)
                logging.info(f"Edge index tensor: {edge_index.shape}")
                
                # Compute neighbour mean   (incoming edges)
                logging.info("Computing neighbour mean (incoming edges)")
                neighbor_mean = scatter_mean(x[row_idx], col_idx, dim=0, dim_size=x.size(0))
                logging.info(f"Neighbor mean computed: {neighbor_mean.shape}")

                # Load model and fetch first GraphSAGE layer output
                try:
                    from regnet.models.regnet import RegNet
                    model_path = os.path.join(out_dir, "regnet_pretrained.pth")
                    state = torch.load(model_path, map_location="cpu")

                    # Infer hidden dimension from first GraphSAGE weight
                    w_key = next(k for k in state.keys() if 'graphsage.layers.0.agg.mlp.0.weight' in k)
                    hidden_dim_infer = state[w_key].shape[0]
                    input_dim = x.shape[1]
                    latent_dim_guess = 64
                    model = RegNet(input_dim, hidden_dim_infer, latent_dim_guess)
                    model.load_state_dict(state, strict=False)
                    model.eval()

                    with torch.no_grad():
                        first_layer = model.graphsage.layers[0]
                        h_first = first_layer(x, edge_index)    # (N, hidden)

                    # --- Effective shrinkage weight via local linearisation --- #
                    def compute_effective_w(x_feat, m_feat, agg_module, eps=1e-2, batch=2048):
                        """Finite-difference approximation of w_i^eff.

                        Parameters
                        ----------
                        x_feat   : (N,F) tensor of self features
                        m_feat   : (N,F) tensor of neighbour means (treated as prior)
                        agg_module : MLPAggregator (expects .forward(x,m))
                        eps      : small scalar step along delta direction
                        batch    : batch size for vectorised computation
                        Returns
                        -------
                        w_eff : (N,) tensor in [0,1]
                        """
                        N, F = x_feat.shape
                        delta = x_feat - m_feat
                        delta_norm = delta.norm(dim=1, keepdim=True)
                        # avoid zero - norm rows
                        delta_unit = torch.where(delta_norm > 0,
                                                  delta / delta_norm,
                                                  torch.zeros_like(delta))
                        w_list = []
                        for start in range(0, N, batch):
                            end = min(start + batch, N)
                            xi = x_feat[start:end]
                            mi = m_feat[start:end]
                            d  = delta_unit[start:end]

                            out0 = agg_module(xi, mi)
                            out_x = agg_module(xi + eps * d, mi)
                            out_m = agg_module(xi,         mi + eps * d)

                            Sx = (out_x - out0).norm(dim=1) / eps
                            Sm = (out_m - out0).norm(dim=1) / eps
                            w_eff = Sx / (Sx + Sm + 1e-9)
                            w_list.append(w_eff)
                        return torch.cat(w_list, dim=0)

                    alpha = compute_effective_w(x, neighbor_mean, first_layer.agg)
                    logging.info("Shrinkage weights (effective) computed via finite differences")
                except Exception as e:
                    logging.error(f"Error computing shrinkage: {e}")
                    logging.error(traceback.format_exc())
                    alpha = torch.zeros(x.size(0))

                # Compute in-degree (incoming edges)
                with torch.no_grad():
                    deg = torch.bincount(col_idx, minlength=x.size(0)).float()
                logging.info(f"In-degree vector computed: {deg.shape}")
                
                # Create dataframe for plotting - convert tensors to numpy safely
                alpha_np = safe_detach(alpha).cpu().numpy()
                deg_np = safe_detach(deg).cpu().numpy()
                
                # Create is_tf flags
                is_tf = [g in tf_set for g in expr_df.index]
                
                df_s = pd.DataFrame({
                    "deg": deg_np + 1e-3,  # Add small constant to avoid log(0)
                    "alpha": alpha_np,
                    "is_tf": is_tf
                })
                logging.info(f"DataFrame created: {df_s.shape}")
                
                # ---------- plots ---------------------------------------------------- #
                sns.set_style("whitegrid")

                # (1) alpha vs log-degree
                plt.figure(figsize=(6,4))
                sns.scatterplot(x="deg", y="alpha", hue="is_tf", data=df_s, alpha=0.6, s=10)
                plt.xscale("log");
                plt.xlabel("In-degree (log10)"); plt.ylabel("Shrinkage weight w_i");
                plt.title("Bayesian shrinkage vs in-degree (layer-1)");
                path1 = os.path.join(out_dir, "shrinkage_vs_degree.png")
                plt.tight_layout(); plt.savefig(path1, dpi=300); plt.close()

                # (2) histogram of w_i
                plt.figure(figsize=(5,4))
                sns.histplot(df_s["alpha"], bins=50, color="steelblue");
                plt.xlabel("Shrinkage weight w_i"); plt.ylabel("Count");
                plt.title("Distribution of shrinkage weights");
                path2 = os.path.join(out_dir, "shrinkage_hist.png")
                plt.tight_layout(); plt.savefig(path2, dpi=300); plt.close()

                # (3) w_i vs mean expression
                mean_expr = x.mean(dim=1).cpu().numpy();
                df_s["expr"] = mean_expr + 1e-6
                plt.figure(figsize=(6,4))
                sns.scatterplot(x="expr", y="alpha", hue="is_tf", data=df_s, s=10, alpha=0.6)
                plt.xscale("log");
                plt.xlabel("Mean expression (log10)"); plt.ylabel("Shrinkage weight w_i");
                plt.title("Shrinkage vs expression level");
                path3 = os.path.join(out_dir, "shrinkage_vs_expr.png")
                plt.tight_layout(); plt.savefig(path3, dpi=300); plt.close()

                logging.info("Shrinkage plots saved: %s, %s, %s", path1, path2, path3)
            else:
                logging.error("No expression data available for shrinkage plot")
        except Exception as e:
            logging.error(f"Shrinkage plot failed: {e}")
            logging.error(traceback.format_exc())

    # Build index-to-gene mapping (optional) -----------------------------------
    idx_to_gene = {}
    try:
        # We assume TF file path is provided and contains an 'index' column
        if args.tf_file and os.path.isfile(args.tf_file):
            tf_map_df = pd.read_csv(args.tf_file)
            if 'index' in tf_map_df.columns:
                idx_to_gene.update({int(idx): str(name) for idx, name in zip(tf_map_df['index'], tf_map_df.iloc[:,1])})
        # Try to guess a Target.csv file in the same directory
        if args.tf_file:
            cand_target = os.path.join(os.path.dirname(args.tf_file), 'Target.csv')
            if os.path.isfile(cand_target):
                tgt_map_df = pd.read_csv(cand_target)
                if 'index' in tgt_map_df.columns:
                    idx_to_gene.update({int(idx): str(name) for idx, name in zip(tgt_map_df['index'], tgt_map_df.iloc[:,1])})
        logging.info(f"Loaded index->gene mapping with {len(idx_to_gene)} entries")
    except Exception as e:
        logging.warning(f"Failed to load index mapping: {e}")
    
    # ---------------------------------------------------------------------------

    print("Visualisations saved to", out_dir)


if __name__ == "__main__":
    main() 