import argparse, os, numpy as np, pandas as pd, torch, torch.optim as optim
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


from regnet.models.regnet import RegNet
from regnet.preprocessing.data_loader import load_data, create_batches_NC
from regnet.preprocessing.utils import (
    create_finetune_adj_matrix, filter_state_dict,
    load_pretrained_embeddings, load_prototypes,
    load_fisher, centred_pearson)
from regnet.training.pretrain import save_outputs
from regnet.training.loss_functions import (
    info_nce_loss, masked_recon_loss,
    attention_entropy_loss,                        
    layerwise_discrepancy_loss, relational_alignment_loss,
    prototype_alignment_loss, ewc_loss)

# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    # — data —
    p.add_argument('--expression_data', required=True)
    p.add_argument('--TF_file',           required=True)
    p.add_argument('--target_file',       required=True)
    p.add_argument('--pretrain_expression_data', required=True)
    p.add_argument('--pretrain_adj',      required=True)
    p.add_argument('--finetune_full_set', required=True)
    p.add_argument('--pretrain_output_dir', required=True)
    p.add_argument('--pretrain_fisher_file', required=True)
    p.add_argument('--pretrained_model',  required=True)
    # — training —
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--patience',   type=int, default=5)
    p.add_argument('--lr',         type=float, default=2e-4)
    p.add_argument('--lr_min_factor', type=float, default=0.1)
    p.add_argument('--lr_cycle',   type=int, default=5)
    p.add_argument('--lr_mult',    type=int, default=2)
    # — model —
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--latent_dim', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=2)
    # --- new hyper‑params ----------------------------------------------------
    p.add_argument('--w_attn_entropy', type=float, default=0.0,
                   help="Row‑entropy regulariser weight")
    p.add_argument('--sigma_rec', type=float, default=0.1,
                   help="σ for masked‑gene Gaussian NLL")
    p.add_argument('--beta_kl', type=float, default=1.0)
    # — priors / penalties —
    p.add_argument('--ewc_lambda_graphsage', type=float, default=1.0)
    p.add_argument('--ewc_lambda_attention', type=float, default=1.0)
    p.add_argument('--ewc_lambda_vae',       type=float, default=1.0)
    # — self‑sup —
    p.add_argument('--mask_ratio',    type=float, default=0.1)
    p.add_argument('--freeze_epochs', type=int,   default=3)
    # — loss weights —
    p.add_argument('--w_mask', type=float, default=1e-3)
    p.add_argument('--w_con',  type=float, default=1.0)
    p.add_argument('--w_vae',  type=float, default=1.0)
    p.add_argument('--w_disc', type=float, default=1.0)
    p.add_argument('--w_rel',  type=float, default=1.0)
    p.add_argument('--w_prot', type=float, default=10.0)
    p.add_argument('--w_ewc',  type=float, default=1.0)
    # — misc —
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def evaluate(preds, labels):
    return (average_precision_score(labels, preds),
            roc_auc_score(labels, preds))

# ------------------------------------------------------------------ #
def hard_negative_pairs(pos_pairs, num_nodes, k=1):
    """
    Generate hard negatives by swapping targets inside the batch.
    """
    pos_tf, pos_tg = pos_pairs[:, 0], pos_pairs[:, 1]
    shuffle = pos_tg[torch.randperm(len(pos_tg))]
    neg1    = torch.stack([pos_tf, shuffle], dim=1)
    mask    = (shuffle != pos_tg)
    neg1    = neg1[mask]
    if k > 1:
        extra = torch.randint(0, num_nodes,
                              (len(pos_pairs)*(k-1), 2),
                              device=pos_pairs.device)
        neg = torch.cat([neg1, extra], dim=0)[:len(pos_pairs)*k]
    else:
        neg = neg1
    return neg

# ------------------------------------------------------------------ #
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dev = torch.device(args.device)

    # ---------- 1. data ------------------------------------------------------
    expr_df, _ = load_data(args.expression_data, args.TF_file,
                           args.target_file, None)
    finetune_genes = set(expr_df.index)

    pretrain_emb_df = load_pretrained_embeddings(args.pretrain_output_dir)
    pretrain_emb_df = pretrain_emb_df[~pretrain_emb_df.index.duplicated(keep='first')]

    common_genes  = sorted(finetune_genes & set(pretrain_emb_df.index))
    finetune_only = sorted(finetune_genes - set(pretrain_emb_df.index))
    expr_df       = expr_df.loc[common_genes + finetune_only]

    print(f"Genes: {len(common_genes)} common, {len(finetune_only)} new")

    adj = create_finetune_adj_matrix(pd.read_csv(args.pretrain_adj),
                                     common_genes + finetune_only)
    edge_index   = torch.tensor(np.nonzero(adj), dtype=torch.long, device=dev)
    edge_weights = torch.tensor(adj[adj > 0],    dtype=torch.float32)
    dataloader   = create_batches_NC(expr_df, edge_index,
                                     edge_weights, args.batch_size)

    # ---------- 2. models ----------------------------------------------------
    model = RegNet(expr_df.shape[1], args.hidden_dim,
                   args.latent_dim, args.num_layers).to(dev)
    pretrained_model = RegNet(pd.read_csv(args.pretrain_expression_data,
                                          index_col=0).shape[1],
                              args.hidden_dim, args.latent_dim,
                              args.num_layers).to(dev)

    sd = filter_state_dict(torch.load(args.pretrained_model,
                                      map_location=dev),
                           exclude_keywords=['graphsage.layers.0.agg'])
    model.load_state_dict(sd, strict=False)
    pretrained_model.load_state_dict(sd, strict=False)

    # freeze GraphSAGE initially
    for p in model.graphsage.parameters():
        p.requires_grad_(False)

    fisher_diag = load_fisher(args.pretrain_fisher_file, dev)

    decoder = nn.Sequential(
        nn.Linear(args.latent_dim, args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, expr_df.shape[1])
    ).to(dev)

    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()),
                           lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.lr_cycle, T_mult=args.lr_mult,
        eta_min=args.lr * args.lr_min_factor)

    # ---------- 3. tensors ---------------------------------------------------
    pretrained_rel = centred_pearson(
        torch.tensor(pretrain_emb_df.loc[common_genes].values,
                     dtype=torch.float32, device=dev))
    gene_to_idx = {g: i for i, g in enumerate(expr_df.index)}
    common_global2row = {gene_to_idx[g]: r for r, g in enumerate(common_genes)}
    proto = load_prototypes(args.pretrain_output_dir).to(dev)

    labels_df = pd.read_csv(args.finetune_full_set)
    tf_map  = dict(zip(pd.read_csv(args.TF_file)['index'],
                       pd.read_csv(args.TF_file)['TF']))
    tgt_map = dict(zip(pd.read_csv(args.target_file)['index'],
                       pd.read_csv(args.target_file)['Gene']))
    labels_df['TF_gene']     = labels_df['TF'].map(tf_map)
    labels_df['target_gene'] = labels_df['Target'].map(tgt_map)
    labels_df.set_index(['TF_gene', 'target_gene'], inplace=True)

    lambda_ewc = {'graphsage': args.ewc_lambda_graphsage,
                  'attention': args.ewc_lambda_attention,
                  'vae':       args.ewc_lambda_vae,
                  'other':     0.0}

    # ---------- 4. training ---------------------------------------------------
    best_auroc, epochs_no_improve = -1, 0
    history = []

    for ep in range(1, args.epochs + 1):

        if ep == args.freeze_epochs + 1:
            for p in model.graphsage.parameters():
                p.requires_grad_(True)
            print("⇢ Unfroze GraphSAGE layers.")

        model.train()
        sums = dict(mask=0, con=0, vae=0, disc=0,
                    rel=0, prot=0, ewc=0, attn=0, total=0)

        for batch in dataloader:
            batch = batch.to(dev)
            optimizer.zero_grad()

            # forward – returns row_entropy as sixth output
            _, _, row_ent, mu, logvar = model(
                batch.x, batch.edge_index, batch.edge_index.t())

            # masked reconstruction
            mask      = (torch.rand_like(batch.x) < args.mask_ratio).float()
            x_masked  = batch.x * (1 - mask)
            mu_mask   = model.latent_mu(x_masked, batch.edge_index)
            x_rec     = decoder(mu_mask)
            loss_mask = masked_recon_loss(batch.x, x_rec, mask,
                                          sigma_rec=args.sigma_rec)

            # relational alignment
            loc = [i for i, n in enumerate(batch.n_id.tolist())
                   if n in common_global2row]
            if len(loc) > 1:
                mu_sub = mu[loc]
                sel = torch.tensor([common_global2row[int(n)]
                                    for n in batch.n_id[loc]],
                                   dtype=torch.long, device=dev)
                loss_rel = relational_alignment_loss(
                    centred_pearson(mu_sub),
                    pretrained_rel[sel][:, sel])
            else:
                loss_rel = torch.tensor(0., device=dev)

            # InfoNCE with hard negatives
            pos      = batch.edge_index.t()
            neg      = hard_negative_pairs(pos, mu.size(0))
            loss_con = info_nce_loss(mu, pos, neg, temperature=0.07)

            loss_vae = model.vae.loss_function(mu, logvar) * args.beta_kl
            loss_disc = layerwise_discrepancy_loss(
                model, pretrained_model, {'attention': 1, 'vae': 1})
            emb_pairs = torch.cat([mu[pos[:, 0]], mu[pos[:, 1]]], dim=1)
            loss_prot = prototype_alignment_loss(emb_pairs, proto)
            loss_ewc  = args.w_ewc * ewc_loss(
                model, pretrained_model.state_dict(),
                fisher_diag, lambda_ewc)
            loss_attn = args.w_attn_entropy * \
                        attention_entropy_loss(row_ent)

            total = (args.w_mask * loss_mask + args.w_con * loss_con +
                     args.w_vae * loss_vae + args.w_disc * loss_disc +
                     args.w_rel * loss_rel + args.w_prot * loss_prot +
                     loss_ewc + loss_attn)

            total.backward()
            optimizer.step()

            sums['mask'] += loss_mask.item()
            sums['con']  += loss_con.item()
            sums['vae']  += loss_vae.item()
            sums['disc'] += loss_disc.item()
            sums['rel']  += loss_rel.item()
            sums['prot'] += loss_prot.item()
            sums['ewc']  += loss_ewc.item()
            sums['attn'] += loss_attn.item()
            sums['total'] += total.item()

        # prototype EMA
        with torch.no_grad():
            mu_global = model.latent_mu(
                torch.tensor(expr_df.values, dtype=torch.float32, device=dev),
                edge_index)
            mean  = mu_global.mean(0, keepdim=True)
            proto = 0.9 * proto + 0.1 * torch.cat([mean, mean], dim=-1)

        # evaluation ----------------------------------------------------------
        model.eval()
        with torch.no_grad():
            mu_full = model.latent_mu(
                torch.tensor(expr_df.values, dtype=torch.float32, device=dev),
                edge_index)
            preds, lbls = [], []
            for (tf, tg), row in labels_df.iterrows():
                if tf in gene_to_idx and tg in gene_to_idx:
                    pair = torch.cat([mu_full[gene_to_idx[tf]],
                                      mu_full[gene_to_idx[tg]]])
                    preds.append(model.edge_classifier(pair).item())
                    lbls.append(row['Label'])
            aupr, auroc = evaluate(torch.tensor(preds),
                                   torch.tensor(lbls))

        # early stopping ------------------------------------------------------
        if auroc > best_auroc + 1e-3:
            best_auroc, epochs_no_improve = auroc, 0
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early‑stop at epoch {ep}. Best AUROC={best_auroc:.3f}")
                break

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        n_batch = len(dataloader) or 1
        log_line = (f"Ep{ep:02d} "
                    f"M{sums['mask'] / n_batch:.3f} "
                    f"C{sums['con']  / n_batch:.3f} "
                    f"V{sums['vae']  / n_batch:.3f} "
                    f"D{sums['disc'] / n_batch:.3f} "
                    f"R{sums['rel']  / n_batch:.3f} "
                    f"P{sums['prot'] / n_batch:.3f} "
                    f"E{sums['ewc']  / n_batch:.3f} "
                    f"A{sums['attn'] / n_batch:.3f} │ "
                    f"Tot{sums['total'] / n_batch:.1f} "
                    f"AUROC{auroc:.3f} AUPR{aupr:.3f} "
                    f"LR{current_lr:.2e}")
        print(log_line)

        history.append([ep, sums['total'], aupr, auroc, current_lr])

    # ---------- 5. save outputs ---------------------------------------------
    pd.DataFrame(history,
                 columns=['Epoch', 'Loss', 'AUPR', 'AUROC', 'LR']).to_csv(
                     os.path.join(args.output_dir, 'finetune_metrics.csv'),
                     index=False)
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'final_model.pth'))
    save_outputs(model, dataloader, dev, args.output_dir,
                 expr_df.index.tolist(), edge_index, edge_weights)


if __name__ == '__main__':
    main()
