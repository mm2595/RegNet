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
from regnet.training.loss_functions import vae_reconstruction_loss

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
    """Load edges from train_test_split output format"""
    labels_df = pd.read_csv(label_path)
    
    edges, labels = [], []
    for _, row in labels_df.iterrows():
        g1, g2 = row["TF"], row["Target"]
        if g1 in gene_to_idx and g2 in gene_to_idx:
            edges.append([gene_to_idx[g1], gene_to_idx[g2]])
            labels.append(float(row["Label"]))
    
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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ------------------------------------------------------------------ #
def evaluate_metrics(preds, labels):
    p, y = preds.cpu().numpy(), labels.cpu().numpy()
    return average_precision_score(y, p), roc_auc_score(y, p)

def train_epoch(model, loader, optim_, crit, device, beta_kl=1.0, recon_weight=0.1):
    model.train()
    tot = 0; preds_all = []; labels_all = []
    for data in loader:
        data = data.to(device)
        optim_.zero_grad()
        preds, _, row_ent, mu, logvar = model(data.x,
                                        data.edge_index,
                                        data.edge_index.t())
        
        # Edge prediction loss (supervised)
        edge_loss = crit(preds, data.edge_attr.float())
        
        # VAE KL loss
        kl_loss = model.vae.loss_function(mu, logvar) * beta_kl
        
        # VAE reconstruction loss (auxiliary task)
        recon_loss = vae_reconstruction_loss(model, data.x, data.edge_index) * recon_weight
        
        # Total loss
        loss = edge_loss + kl_loss + recon_loss
        
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
            preds = model.edge_classifier(pair_feat).squeeze().cpu().numpy()

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
        row_ent = torch.cat(row_ent_values).mean()
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

    model = RegNet(expr_df.shape[1], args.hidden_dim,
                   args.latent_dim, args.num_layers).to(args.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for ep in range(args.epochs):
        loss, aupr, auroc = train_epoch(
            model, loader, optimizer, criterion, args.device, 
            beta_kl=args.beta_kl, recon_weight=args.recon_weight)
        print(f"Ep{ep+1:02d}  L{loss:.4f}  AUPR{aupr:.3f} AUROC{auroc:.3f}")
        history.append({"epoch": ep+1,
                        "loss": loss, "AUPR": aupr, "AUROC": auroc})

    pd.DataFrame(history).to_csv(f"{args.output_dir}/training_metrics.csv",
                                 index=False)
    torch.save(model.state_dict(),
               f"{args.output_dir}/regnet_pretrained.pth")

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
