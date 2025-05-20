#!/usr/bin/env python
"""
Empirical-Bayes shrinkage diagnostics for the FIRST GraphSAGE layer.

The script assumes:
  • nodes  = genes
  • features = normalised-log expression of those genes across samples
    (matrix shape  n_genes × n_samples  after CSV load)
  • edge_index lists directed gene-gene edges (shape 2 × E)
  • the pretrained checkpoint contains parameters prefixed with 'graphsage.'

It creates four PNG panels:
  A  shrinkage arrows
  B  histogram of weights
  C  weight vs gene mean
  D  variance before vs after GraphSAGE

Author : 2025-05-07
"""

import argparse, os, numpy as np, pandas as pd, torch
from torch_scatter import scatter_mean
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# --------------------------------------------------------------------------- #
#                                  Helpers                                    #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GraphSAGE shrinkage diagnostics")
    p.add_argument("--expression", required=True,
                   help="CSV  (rows = genes, cols = samples)  log-norm counts")
    p.add_argument("--adjacency",  required=True,
                   help="CSV  two columns [source_gene,target_gene]  OR indices")
    p.add_argument("--model",      required=True,
                   help="Checkpoint (.pth/.pt) containing GraphSAGE weights")
    p.add_argument("--outdir",     required=True,
                   help="Directory to store PNGs")
    p.add_argument("--n_pairs", type=int, default=200,
                   help="# gene–sample pairs for arrow plot")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


def require(path: str, desc: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} not found at: {path}")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#                                  Loading                                    #
# --------------------------------------------------------------------------- #
def load_expression(path: str, dev: torch.device):
    df = pd.read_csv(path, index_col=0)
    return torch.tensor(df.values, dtype=torch.float32, device=dev), df.index.tolist()


def load_edges(path: str, gene_to_idx: dict, dev: torch.device):
    df = pd.read_csv(path)
    if {"TF_gene", "Target_gene"}.issubset(df.columns):
        src = [gene_to_idx[g] for g in df["TF_gene"]]
        tgt = [gene_to_idx[g] for g in df["Target_gene"]]
    else:  # assume numeric indices
        src, tgt = df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()
    return torch.tensor([src, tgt], dtype=torch.long, device=dev)


# --------------------------------------------------------------------------- #
#                                   Plots                                     #
# --------------------------------------------------------------------------- #
def panel_arrows(x, m, h1, n_pairs, out_png):
    n_genes, n_samples = x.shape
    rng = np.random.default_rng()
    g_idx = rng.choice(n_genes, n_pairs, replace=True)
    s_idx = rng.choice(n_samples, n_pairs, replace=True)

    plt.figure(figsize=(6, 6))
    for g, s in zip(g_idx, s_idx):
        # add tiny jitter to neighbourhood mean to avoid overplot at zero
        prior = m[g, s].item() + rng.normal(loc=0, scale=1e-6)
        raw   = x[g, s].item()
        post  = h1[g, s].item()
        # grey dot at (prior, raw)
        plt.plot(prior, raw, 'o', markersize=3, color="grey", alpha=0.4)
        # blue arrow to (prior, post)
        plt.plot([prior, prior], [raw, post], lw=0.8, color="steelblue", alpha=0.6)
    plt.xlabel("Neighbour mean  $\\tilde m_{is}$")
    plt.ylabel("Expression value")
    plt.title("Shrinkage arrows  ({} random gene–cell pairs)".format(n_pairs))
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def panel_weight_hist(w_hat, out_png):
    w = w_hat.cpu().numpy().flatten()
    w = w[(w > -0.05) & (w < 1.05)]  # trim crazy outliers
    plt.figure(figsize=(6, 4))
    sns.histplot(w, bins=40, color="skyblue")
    plt.xlim(0, 1)
    plt.xlabel("Empirical shrinkage weight  $\\hat w_{is}$")
    plt.ylabel("Count")
    plt.title("Distribution of shrinkage weights")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def panel_weight_vs_mean(x, w_hat, out_png):
    gene_means = x.mean(1).cpu().numpy()
    gene_weights = w_hat.mean(1).cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.scatter(np.log1p(gene_means), gene_weights, s=6, alpha=0.25)
    # LOWESS
    smoothed = lowess(gene_weights, np.log1p(gene_means), frac=0.3, return_sorted=True)
    plt.plot(smoothed[:, 0], smoothed[:, 1], 'r-', lw=2)
    plt.xlabel("log(1 + mean expression)")
    plt.ylabel("Mean shrinkage weight per gene")
    plt.title("Shrinkage weight vs expression level")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def panel_variance(x, h1, out_png):
    var_before = x.var(1, unbiased=False).cpu().numpy()
    var_after  = h1.var(1, unbiased=False).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(var_before, var_after, s=8, alpha=0.3)
    lims = [var_before.min()*0.8, var_before.max()*1.2]
    plt.plot(lims, lims, 'k--', lw=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Variance before GraphSAGE")
    plt.ylabel("Variance after GraphSAGE")
    plt.title("Per-gene variance shrinkage")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
#                                    Main                                     #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = device()

    # --- I/O checks -------------------------------------------------------- #
    require(args.expression, "Expression matrix")
    require(args.adjacency,  "Adjacency file")
    require(args.model,      "Model checkpoint")
    os.makedirs(args.outdir, exist_ok=True)

    # --- Data -------------------------------------------------------------- #
    x, gene_list = load_expression(args.expression, dev)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    edge_index = load_edges(args.adjacency, gene_to_idx, dev)

    # --- Model (first GraphSAGE layer) ------------------------------------- #
    from regnet.models.graphsage import GraphSAGE  # correct import path
    in_dim = x.size(1)  # number of samples per gene
    # Load checkpoint to infer GraphSAGE output dimension
    ckpt = torch.load(args.model, map_location=dev)
    hidden_key = next(
        (k for k in ckpt.keys() if k.startswith("graphsage.layers.0.agg.mlp.2.weight")),
        None
    )
    hidden_dim = ckpt[hidden_key].shape[0] if hidden_key else in_dim
    # Instantiate model matching checkpoint architecture
    model = GraphSAGE(in_dim, hidden_dim, num_layers=1).to(dev)
    # Load complete first GraphSAGE layer weights strictly
    layer0 = {
        k.replace("graphsage.", ""): v
        for k, v in ckpt.items()
        if k.startswith("graphsage.layers.0.")
    }
    missing, unexpected = model.load_state_dict(layer0, strict=True)
    model.eval()

    with torch.no_grad():
        h1_hidden = model(x, edge_index)  # (n_genes, hidden_dim)

    # Project hidden embeddings back to sample coordinate space using first MLP linear weights
    agg_layer = model.layers[0].agg  # GraphSAGEConv aggregator
    linear1 = agg_layer.mlp[0]       # first Linear(2*in_dim -> hidden_dim)
    W1_x = linear1.weight[:, :in_dim]  # hidden_dim x in_dim (mapping x portion)
    h1 = torch.matmul(h1_hidden, W1_x)  # (n_genes, in_dim)
    # Detach projected embeddings from graph to allow numpy conversion
    h1 = h1.detach()

    # --- Neighbour mean and weights --------------------------------------- #
    row, col = edge_index
    m = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))  # compute neighbor mean over samples
    # Compute shrinkage weights using projected embeddings
    w_hat = (h1 - m) / (x - m + 1e-8)
    w_hat[~torch.isfinite(w_hat)] = 0.0  # clean NaNs/Infs
    # Detach from computation graph so we can convert to numpy
    w_hat = w_hat.detach()

    # --- Plots ------------------------------------------------------------- #
    panel_arrows(x, m, h1, args.n_pairs, os.path.join(args.outdir, "panel_A_arrows.png"))
    panel_weight_hist(w_hat,          os.path.join(args.outdir, "panel_B_hist.png"))
    panel_weight_vs_mean(x, w_hat,    os.path.join(args.outdir, "panel_C_w_vs_mean.png"))
    panel_variance(x, h1,             os.path.join(args.outdir, "panel_D_variance.png"))

    print(f"Saved diagnostic panels to  {args.outdir}")


if __name__ == "__main__":
    main()
