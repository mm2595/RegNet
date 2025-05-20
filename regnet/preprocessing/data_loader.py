import argparse
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import NeighborLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Data loader for RegNet")
    parser.add_argument("--expression_data", required=True, type=str, help="Path to expression CSV file")
    parser.add_argument("--label_data", type=str, help="Path to CSV with TF-target edge labels")
    parser.add_argument("--TF_file", required=True, type=str, help="Path to TF genes CSV")
    parser.add_argument("--target_file", required=True, type=str, help="Path to target genes CSV")
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for data loading")
    return parser.parse_args()

def load_data(expression_path, tf_path, target_path, label_path=None):

    expr_df = pd.read_csv(expression_path, index_col=0)

    tf_df = pd.read_csv(tf_path)
    target_df = pd.read_csv(target_path)

    tf_idx_to_name = dict(zip(tf_df["index"], tf_df["TF"]))
    target_idx_to_name = dict(zip(target_df["index"], target_df["Gene"]))

    if label_path is not None:
        labels_df = pd.read_csv(label_path)
        labels_df['TF_gene'] = labels_df['TF'].map(tf_idx_to_name)
        labels_df['target_gene'] = labels_df['Target'].map(target_idx_to_name)
        labels_df.dropna(subset=['TF_gene', 'target_gene'], inplace=True)

        all_genes = set(labels_df['TF_gene']).union(labels_df['target_gene'])
        matched_genes = list(all_genes.intersection(expr_df.index))

        if not matched_genes:
            raise ValueError("No matching genes found between label data and expression data.")

        expr_filtered = expr_df.loc[matched_genes]

        gene_to_idx = {gene: idx for idx, gene in enumerate(expr_filtered.index)}

        edge_index, edge_labels = edges_to_index(labels_df, gene_to_idx)

    else:
        expr_filtered = expr_df
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_labels = torch.empty((0,), dtype=torch.float32)

    return expr_filtered, (edge_index, edge_labels)


def edges_to_index(edges_df, gene_to_idx):
    edges = []
    labels = []

    for _, row in edges_df.iterrows():
        tf_gene = row['TF_gene']
        target_gene = row['target_gene']

        tf_idx = gene_to_idx.get(tf_gene)
        target_idx = gene_to_idx.get(target_gene)

        if tf_idx is not None and target_idx is not None:
            edges.append([tf_idx, target_idx])
            labels.append(float(row['Label']))  

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(labels, dtype=torch.float32)

    return edge_index, edge_labels


def create_batches_NC(expr_df, edge_index, edge_weights, batch_size, num_neighbors=10):
    data = Data(
        x=torch.tensor(expr_df.values, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weights, dtype=torch.float)
    )

    loader = NeighborLoader(
        data,
        num_neighbors=[num_neighbors] * 2,
        batch_size=batch_size,
        shuffle=True
    )

    return loader


def batch_edge_reindex(edge_index, start_idx, end_idx):
    mapping = {original_idx: idx for idx, original_idx in enumerate(range(start_idx, end_idx))}
    mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx) & \
            (edge_index[1] >= start_idx) & (edge_index[1] < end_idx)
    batch_edges = edge_index[:, mask].clone()
    batch_edges[0] -= start_idx
    batch_edges[1] -= start_idx
    return batch_edges.clone().detach().long()


def main():
    args = parse_args()

    expr_df, (edge_index, edge_weights) = load_data(
        expression_path=args.expression_data,
        tf_path=args.TF_file,
        target_path=args.target_file,
        label_path=args.label_data
    )

    batches = create_batches_NC(expr_df, edge_index, edge_weights, args.batch_size)

    return batches


if __name__ == "__main__":
    batches = main()
