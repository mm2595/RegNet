import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import Dict, Iterable, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F

def create_finetune_adj_matrix(pretrain_adj_df, finetune_genes):
    finetune_gene_to_idx = {gene: idx for idx, gene in enumerate(finetune_genes)}
    
    finetune_adj_matrix = np.zeros((len(finetune_genes), len(finetune_genes)), dtype=int)

    connected_pairs = 0
    potential_pairs = 0

    pretrain_edges = set(zip(pretrain_adj_df['TF_gene'], pretrain_adj_df['Target_gene']))
    print(f"Number of pairs in pretrain_adj: {len(pretrain_edges)}")
    common_genes = set(pretrain_adj_df['TF_gene']).union(set(pretrain_adj_df['Target_gene']))
    common_genes = common_genes.intersection(set(finetune_genes))

    for gene1 in common_genes:
        for gene2 in common_genes:
            if gene1 == gene2:
                continue
            potential_pairs += 1
            if (gene1, gene2) in pretrain_edges:
                idx1, idx2 = finetune_gene_to_idx[gene1], finetune_gene_to_idx[gene2]
                finetune_adj_matrix[idx1, idx2] = 1
                connected_pairs += 1

    print("Connected pairs: {}/{} ({:.2f}%)".format(
        connected_pairs, potential_pairs, (connected_pairs / float(potential_pairs)) * 100
    ))

    return finetune_adj_matrix

def align_expression_data(expr_df, pretrain_genes, output_dir):
    common_genes = [g for g in pretrain_genes if g in expr_df.index]
    new_genes = [g for g in expr_df.index if g not in pretrain_genes]
    expr_df = pd.concat([expr_df.loc[common_genes], expr_df.loc[new_genes]])
    expr_df.to_csv(f"{output_dir}/finetune_gene_order.csv")
    return expr_df, common_genes + new_genes

def filter_state_dict(state_dict, exclude_keywords):
    return {k: v for k, v in state_dict.items() if all(kw not in k for kw in exclude_keywords)}


def compute_relational_distributions(embeddings):
    """
    Compute relational distributions of embeddings using cosine similarity.
    """
    cosine_sim_matrix = cosine_similarity(embeddings)
    distances = pdist(embeddings, metric='euclidean')
    distance_matrix = squareform(distances)
    
    return {
        'cosine_similarity': cosine_sim_matrix,
        'euclidean_distance': distance_matrix
    }

def compute_prototypes(embeddings, edge_index, edge_labels):
    """
    Compute prototype embeddings for positive (connected) and negative (non-connected) pairs.
    """
    positive_pairs = embeddings[edge_index[:, edge_labels == 1]]
    negative_pairs = embeddings[edge_index[:, edge_labels == 0]]

    positive_prototype = np.mean(np.concatenate([positive_pairs[0], positive_pairs[1]], axis=1), axis=0)
    negative_prototype = np.mean(np.concatenate([negative_pairs[0], negative_pairs[1]], axis=1), axis=0)

    return {
        'positive_prototype': positive_prototype,
        'negative_prototype': negative_prototype
    }

def save_relational_distributions(distributions, output_dir):
    np.save(f"{output_dir}/cosine_similarity.npy", distributions['cosine_similarity'])
    np.save(f"{output_dir}/euclidean_distance.npy", distributions['euclidean_distance'])

def save_prototypes(prototypes, output_dir):
    np.save(f"{output_dir}/positive_prototype.npy", prototypes['positive_prototype'])
    np.save(f"{output_dir}/negative_prototype.npy", prototypes['negative_prototype'])

def load_pretrained_relational_distribution(pretrain_output_dir):
    cosine_similarity = np.load(f"{pretrain_output_dir}/cosine_similarity.npy")
    return torch.tensor(cosine_similarity, dtype=torch.float32)

def load_prototypes(pretrain_output_dir):
    positive_prototype = np.load(f"{pretrain_output_dir}/positive_prototype.npy")
    return torch.tensor(positive_prototype, dtype=torch.float32)

def load_pretrained_embeddings(pretrain_output_dir):
    vae_emb = pd.read_csv(f"{pretrain_output_dir}/vae_embeddings.csv", index_col=0)
    return vae_emb

def compute_fisher_on_loader(
        model: torch.nn.Module,
        loader: Iterable,
        criterion,
        device: torch.device,
        blocks: Tuple[str, ...] = ("graphsage", "attention", "vae")
) -> Dict[str, torch.Tensor]:
 
    model.eval()
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters()
              if any(b in n for b in blocks)}

    for data in loader:
        data = data.to(device)
        model.zero_grad(set_to_none=True)
        preds, _, _, mu, logvar = model(
            data.x, data.edge_index, data.edge_index.t())
        loss = criterion(preds, data.edge_attr.float()) \
             + model.vae.loss_function(mu, logvar)
        loss.backward()

        for n, p in model.named_parameters():
            if n in fisher:
                fisher[n] += p.grad.detach()**2

    num_batches = len(loader)
    for n in fisher:
        fisher[n] /= num_batches
    return fisher

def save_fisher(fisher_dict: Dict[str, torch.Tensor], filepath: str) -> None:
    torch.save({k: v.cpu() for k, v in fisher_dict.items()}, filepath)

def load_fisher(filepath: str, device: torch.device) -> Dict[str, torch.Tensor]:
    saved = torch.load(filepath, map_location=device)
    return {k: v.to(device) for k, v in saved.items()}

def centred_pearson(z: torch.Tensor) -> torch.Tensor:
    """
    z : (N, d) latent means
    returns R  (N,N) symmetrical, diag=1
    """
    z = z - z.mean(dim=0, keepdim=True)
    z = F.normalize(z, dim=1)                     # row‑wise / gene‑wise
    return torch.mm(z, z.t()).clamp(-1.0, 1.0)

def get_device():
    """Return available device: use CUDA if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')