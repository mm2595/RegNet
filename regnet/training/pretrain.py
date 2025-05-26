#!/usr/bin/env python
"""
Pre‑training RegNet – single embedding per gene
"""

import os, argparse, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np, pandas as pd

from regnet.preprocessing import utils
from regnet.models.regnet import RegNet
from regnet.preprocessing.data_loader import create_batches_NC
from regnet.preprocessing.utils import (
    compute_relational_distributions, compute_prototypes,
    save_relational_distributions, save_prototypes)
from regnet.training.loss_functions import vae_reconstruction_loss, attention_entropy_loss

# ------------------------------------------------------------------ #
def build_edge_index(label_path, tf_path, target_path, gene_to_idx):
    labels_df  = pd.read_csv(label_path)
    tf_df      = pd.read_csv(tf_path)
    target_df  = pd.read_csv(target_path)

    tf_map  = dict(zip(tf_df["index"],     tf_df["TF"]))
    tgt_map = dict(zip(target_df["index"], target_df["Gene"]))

    edges, labels = [], []
    for _, row in labels_df.iterrows():
        g1 = tf_map.get(row["TF"]);    g2 = tgt_map.get(row["Target"])
        if g1 in gene_to_idx and g2 in gene_to_idx:
            edges.append([gene_to_idx[g1], gene_to_idx[g2]])
            labels.append(float(row["Label"]))
    edge_index  = torch.tensor(edges,  dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(labels, dtype=torch.float32)
    return edge_index, edge_labels

# ------------------------------------------------------------------ #
def build_edge_index_from_split(label_path, gene_to_idx):
    """Load edges from train_test_split output format.
    The split CSV can contain either:
      1) columns TF / Target with gene symbols (legacy behaviour), or
      2) numeric TF / Target indices **plus** string tf_gene / target_gene with symbols.
    We first look for the symbol columns and fall back to TF/Target.
    """
    labels_df = pd.read_csv(label_path)

    # Decide which columns to use for symbols
    if {"tf_gene", "target_gene"}.issubset(labels_df.columns):
        tf_col, tg_col = "tf_gene", "target_gene"
    else:
        tf_col, tg_col = "TF", "Target"

    edges, labels = [], []
    for _, row in labels_df.iterrows():
        g1, g2 = row[tf_col], row[tg_col]
        if g1 in gene_to_idx and g2 in gene_to_idx:
            edges.append([gene_to_idx[g1], gene_to_idx[g2]])
            labels.append(float(row["Label"]))

    if len(edges) == 0:
        edge_index  = torch.empty((2, 0), dtype=torch.long)
        edge_labels = torch.empty((0,), dtype=torch.float32)
    else:
        edge_index  = torch.tensor(edges,  dtype=torch.long).t().contiguous()
        edge_labels = torch.tensor(labels, dtype=torch.float32)
    return edge_index, edge_labels

# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser("Pretrain RegNet")
    p.add_argument("--expression_data", required=True)
    p.add_argument("--label_data",      required=True)
    p.add_argument("--TF_file",         default=None)
    p.add_argument("--target_file",     default=None)
    p.add_argument("--split_data",      action="store_true", 
                   help="Use data from train_test_split.py output format")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument('--beta_kl', type=float, default=1.0)
    p.add_argument('--recon_weight', type=float, default=0.1, 
                   help='Weight for VAE reconstruction loss')
    p.add_argument('--entropy_weight', type=float, default=0.01,
                   help='Weight for attention row-entropy regularisation')
    p.add_argument('--dropout', type=float, default=0.2,
                   help='Dropout rate for GraphSAGE layers')
    p.add_argument('--patience', type=int, default=8,
                   help='Early stopping patience (epochs without AUPR improvement)')
    p.add_argument('--min_delta', type=float, default=1e-4,
                   help='Minimum AUPR improvement to reset patience')
    p.add_argument('--val_label_data', default=None,
                   help='Path to validation label CSV (same format); if provided, use for early stopping')
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ------------------------------------------------------------------ #
def evaluate_metrics(preds, labels):
    p, y = preds.cpu().numpy(), labels.cpu().numpy()
    # Handle cases with insufficient class variety
    if y.size == 0 or len(np.unique(y)) == 1:
        # Undefined metrics – fallback to NaN or baseline values
        return float('nan'), float('nan')
    return average_precision_score(y, p), roc_auc_score(y, p)

def train_epoch(model, loader, optim_, crit, device, beta_kl=1.0, recon_weight=0.1, entropy_weight=0.01):
    model.train()
    tot = 0; preds_all = []; labels_all = []
    for data in loader:
        data = data.to(device)
        optim_.zero_grad()
        logits, _, row_ent, mu, logvar = model(data.x,
                                        data.edge_index,
                                        data.edge_index.t())
        preds = torch.sigmoid(logits)
        
        # Edge prediction loss (supervised)
        edge_loss = crit(logits, data.edge_attr.float())
        
        # VAE KL loss
        kl_loss = model.vae.loss_function(mu, logvar) * beta_kl
        
        # VAE reconstruction loss (auxiliary task)
        recon_loss = vae_reconstruction_loss(model, data.x, data.edge_index) * recon_weight
        
        # Attention entropy loss (sparsity)
        entropy_loss = attention_entropy_loss(row_ent) * entropy_weight
        
        # Total loss
        loss = edge_loss + kl_loss + recon_loss + entropy_loss
        
        loss.backward()
        optim_.step()
        tot += loss.item()
        preds_all.append(preds.detach()); labels_all.append(data.edge_attr.float())

    preds_all = torch.cat(preds_all); labels_all = torch.cat(labels_all)
    aupr, auroc = evaluate_metrics(preds_all, labels_all)
    return tot / len(loader), aupr, auroc

# ------------------------------------------------------------------ #
def save_outputs(model, loader, device, out_dir, gene_names,
                 edge_index, edge_labels):
    """
    Dumps embeddings, gate values, predictions, prototypes, etc.
    """
    model.eval(); os.makedirs(out_dir, exist_ok=True)
    gsage_dict, gate_dict, vae_dict = {}, {}, {}
    preds_records = []
    row_ent_values = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            idx_global  = data.n_id.cpu().numpy()
            batch_names = [gene_names[i] for i in idx_global]

            gs_emb = model.graphsage(data.x, data.edge_index)
            # attention returns 4 values; we ignore row_entropy here
            att_out, _, gate_val, batch_row_ent = model.attention(gs_emb)
            z, mu, logvar = model.vae(att_out)
            
            # Save row entropy
            row_ent_values.append(batch_row_ent.detach())

            for i, name in enumerate(batch_names):
                gsage_dict[name] = gs_emb[i].cpu().numpy()
                gate_dict[name]  = gate_val[i].cpu().numpy()
                vae_dict[name]   = mu[i].cpu().numpy()

            node_pair = data.edge_index
            pair_feat = torch.cat([mu[node_pair[0]], mu[node_pair[1]]], dim=-1)
            preds = torch.sigmoid(model.edge_classifier(pair_feat)).squeeze().cpu().numpy()

            tf_genes = [batch_names[i] for i in node_pair[0].cpu().numpy()]
            tg_genes = [batch_names[i] for i in node_pair[1].cpu().numpy()]
            
            # Get edge variance
            var_log = model.edge_variance(mu, logvar, node_pair.t()).cpu().numpy()
            preds_records.extend(zip(tf_genes, tg_genes, preds, var_log))


    def df_from(d): return pd.DataFrame([d[g] for g in gene_names],
                                        index=gene_names)
    df_from(gsage_dict).to_csv(f"{out_dir}/graphsage_embeddings.csv")
    df_from(gate_dict ).to_csv(f"{out_dir}/gate_values.csv")
    df_from(vae_dict  ).to_csv(f"{out_dir}/vae_embeddings.csv")

    preds_df = pd.DataFrame(preds_records,
            columns=["TF_gene", "Target_gene", "Prediction", "LogitVar"])
    preds_df.to_csv(f"{out_dir}/predictions_with_gene_names.csv", index=False)

    preds_df[["TF_gene", "Target_gene"]].drop_duplicates() \
           .to_csv(f"{out_dir}/adjacency_matrix.csv", index=False)

    vae_np = np.stack([vae_dict[g] for g in gene_names])
    dist   = compute_relational_distributions(vae_np)
    prot   = compute_prototypes(vae_np,
                                edge_index.cpu().numpy(),
                                edge_labels.cpu().numpy())
    save_relational_distributions(dist, out_dir)
    save_prototypes(prot, out_dir)
    
    # Save row entropy (mean across batches)
    if row_ent_values:
        row_ent = torch.stack([v.view(1) for v in row_ent_values]).mean()
        np.save(f"{out_dir}/row_entropy.npy", row_ent.cpu().numpy())


# ------------------------------------------------------------------ #
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    expr_df = pd.read_csv(args.expression_data, index_col=0)
    gene_to_idx = {g: i for i, g in enumerate(expr_df.index)}

    # Load edge data based on format
    if args.split_data:
        print("Loading data from train_test_split format...")
        edge_index, edge_labels = build_edge_index_from_split(
            args.label_data, gene_to_idx)
    else:
        print("Loading data from regular format...")
        if args.TF_file is None or args.target_file is None:
            raise ValueError("TF_file and target_file must be provided when not using split_data")
        edge_index, edge_labels = build_edge_index(
            args.label_data, args.TF_file, args.target_file, gene_to_idx)

    loader = create_batches_NC(expr_df, edge_index,
                               edge_labels, args.batch_size)

    # Optional validation loader
    val_loader = None
    if args.val_label_data is not None and os.path.isfile(args.val_label_data):
        print(f"Loading validation edges from {args.val_label_data} …")
        if args.split_data:
            val_edge_index, val_edge_labels = build_edge_index_from_split(
                args.val_label_data, gene_to_idx)
        else:
            if args.TF_file is None or args.target_file is None:
                raise ValueError("TF_file and target_file required for val set when not using split_data")
            val_edge_index, val_edge_labels = build_edge_index(
                args.val_label_data, args.TF_file, args.target_file, gene_to_idx)

        val_loader = create_batches_NC(expr_df, val_edge_index, val_edge_labels, args.batch_size)

    model = RegNet(expr_df.shape[1], args.hidden_dim,
                   args.latent_dim, args.num_layers, dropout=args.dropout).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_aupr = -1.0
    wait = 0
    best_state = None
    for ep in range(args.epochs):
        loss, aupr, auroc = train_epoch(
            model, loader, optimizer, criterion, args.device, 
            beta_kl=args.beta_kl, recon_weight=args.recon_weight, entropy_weight=args.entropy_weight)

        # Optionally compute validation metrics
        val_aupr = None
        val_auroc = None
        if val_loader is not None:
            with torch.no_grad():
                preds_v = []; labels_v = []
                for vdata in val_loader:
                    vdata = vdata.to(args.device)
                    logits_v, _, _, _, _ = model(vdata.x, vdata.edge_index, vdata.edge_index.t())
                    preds_v.append(torch.sigmoid(logits_v).cpu()); labels_v.append(vdata.edge_attr.float().cpu())
                preds_v = torch.cat(preds_v); labels_v = torch.cat(labels_v)
                val_aupr, val_auroc = evaluate_metrics(preds_v, labels_v)

        msg = f"Ep{ep+1:02d}  L{loss:.4f}  train AUPR {aupr:.3f}  AUROC {auroc:.3f}"
        if val_aupr is not None:
            msg += f"  val AUPR {val_aupr:.3f}  val AUROC {val_auroc:.3f}"
        print(msg)

        history.append({"epoch": ep+1, "loss": loss, "AUPR": aupr, "val_AUPR": val_aupr, "AUROC": auroc, "val_AUROC": val_auroc})

        metric_to_track = val_aupr if val_aupr is not None else aupr
        if metric_to_track is None or (isinstance(metric_to_track, float) and np.isnan(metric_to_track)):
            metric_to_track = -float('inf')  # skip updating

        # Early stopping check
        if metric_to_track > best_aupr + args.min_delta:
            best_aupr = metric_to_track
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {ep+1} – best AUPR {best_aupr:.3f}")
                break

    pd.DataFrame(history).to_csv(f"{args.output_dir}/training_metrics.csv",
                                 index=False)

    # Save best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{args.output_dir}/regnet_pretrained.pth")

    print("\nComputing Fisher information …")
    fisher_diag = utils.compute_fisher_on_loader(
        model, loader, criterion, torch.device(args.device))
    utils.save_fisher(fisher_diag,
                      os.path.join(args.output_dir, "fisher_diag.pth"))
    print("Saved Fisher diagonals.")

    save_outputs(model, loader, args.device, args.output_dir,
                 expr_df.index.tolist(), edge_index, edge_labels)
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
