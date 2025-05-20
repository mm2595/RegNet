#!/usr/bin/env python
"""
GRN Train-Test Split
--------------------
Splits a gene regulatory network into train and test sets while attempting to:
1. Keep similar edge-to-node ratios between train and test sets
2. Minimize edges crossing between train and test (create "closed" networks)
3. Allow some node overlap between train and test sets
4. Handle GraphSAGE considerations where nodes aggregate features from neighbors

Example usage:
-------------
python scripts/train_test_split.py \
    --expression_data data/expression_matrix.csv \
    --label_data data/full_labels.csv \
    --tf_file data/TF.csv \
    --target_file data/Target.csv \
    --train_ratio 0.8 \
    --output_dir data/split/
"""

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Set
import community as community_louvain  # python-louvain package for Louvain community detection
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_networkx, to_networkx


def parse_args():
    parser = argparse.ArgumentParser(description="Split GRN into train and test sets")
    parser.add_argument("--expression_data", required=True, help="CSV with gene expression data (rows=genes, cols=samples)")
    parser.add_argument("--label_data", required=True, help="CSV with TF, Target, Label columns")
    parser.add_argument("--tf_file", required=True, help="CSV mapping TF indices to names")
    parser.add_argument("--target_file", required=True, help="CSV mapping Target indices to names")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of nodes to include in training set")
    parser.add_argument("--method", choices=["community", "random", "stratified"], default="community", 
                        help="Method to split network: community-based, random, or stratified by degree")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_cross_edges", type=float, default=0.05, 
                        help="Maximum allowed fraction of edges crossing between train and test")
    parser.add_argument("--output_dir", required=True, help="Directory to save split network data")
    return parser.parse_args()


def load_data(args):
    """Load expression data, label data, and gene mappings"""
    expr_df = pd.read_csv(args.expression_data, index_col=0)
    tf_df = pd.read_csv(args.tf_file)
    target_df = pd.read_csv(args.target_file)
    labels_df = pd.read_csv(args.label_data)
    
    # Create mappings
    tf_map = dict(zip(tf_df["index"], tf_df["TF"]))
    target_map = dict(zip(target_df["index"], target_df["Gene"]))
    
    # Map TF and Target indices to gene names
    labels_df["tf_gene"] = labels_df["TF"].map(tf_map)
    labels_df["target_gene"] = labels_df["Target"].map(target_map)
    
    # Filter out rows with missing gene names
    labels_df = labels_df.dropna(subset=["tf_gene", "target_gene"])
    
    # Ensure all genes in the network have expression data
    all_genes = set(labels_df["tf_gene"]).union(set(labels_df["target_gene"]))
    missing_genes = all_genes - set(expr_df.index)
    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes in the network are missing expression data")
        labels_df = labels_df[~labels_df["tf_gene"].isin(missing_genes) & 
                              ~labels_df["target_gene"].isin(missing_genes)]
    
    return expr_df, labels_df, all_genes


def build_network(labels_df, all_genes):
    """Build a NetworkX graph from the labels dataframe"""
    G = nx.DiGraph()
    
    # Add all genes as nodes
    for gene in all_genes:
        G.add_node(gene)
    
    # Add edges based on interactions
    for _, row in labels_df.iterrows():
        tf = row["tf_gene"]
        target = row["target_gene"]
        label = row["Label"]
        
        # Only add edges with positive labels (actual interactions)
        if label > 0:
            G.add_edge(tf, target, weight=label)
    
    return G


def community_based_split(G, train_ratio, max_cross_edges, seed):
    """
    Split network based on community detection to keep clusters together.
    This helps minimize edges crossing between train and test sets.
    """
    # Convert directed graph to undirected for community detection
    G_undir = G.to_undirected()
    
    # Perform community detection
    partition = community_louvain.best_partition(G_undir, random_state=seed)
    
    # Group nodes by community
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
    
    # Sort communities by size (largest first)
    sorted_communities = sorted(communities.items(), key=lambda x: -len(x[1]))
    
    train_nodes, test_nodes = set(), set()
    current_ratio = 0.0
    
    # Allocate whole communities to train set until we get close to the desired ratio
    for comm_id, nodes in sorted_communities:
        if current_ratio < train_ratio:
            train_nodes.update(nodes)
        else:
            test_nodes.update(nodes)
        
        # Recalculate ratio
        current_ratio = len(train_nodes) / len(G.nodes())
    
    # Calculate crossing edges
    crossing_edges = 0
    for u, v in G.edges():
        if (u in train_nodes and v in test_nodes) or (u in test_nodes and v in train_nodes):
            crossing_edges += 1
    
    cross_edge_ratio = crossing_edges / len(G.edges())
    
    # If too many crossing edges, try an alternative approach with node overlap
    if cross_edge_ratio > max_cross_edges:
        print(f"Initial split has {cross_edge_ratio:.1%} crossing edges, above threshold of {max_cross_edges:.1%}")
        print("Attempting to create overlapping communities to reduce crossing edges...")
        
        # Find boundary nodes that have many connections across train/test boundary
        boundary_nodes = set()
        for u, v in G.edges():
            if (u in train_nodes and v in test_nodes):
                boundary_nodes.add(v)
            elif (u in test_nodes and v in train_nodes):
                boundary_nodes.add(u)
        
        # Add boundary nodes to both sets (create overlap)
        train_nodes.update(boundary_nodes)
        test_nodes.update(boundary_nodes)
        
        # Recalculate crossing edges
        crossing_edges = sum(1 for u, v in G.edges() 
                           if (u in train_nodes and v not in train_nodes) or 
                              (u not in train_nodes and v in train_nodes))
        
        cross_edge_ratio = crossing_edges / len(G.edges())
        print(f"After creating node overlap, crossing edges reduced to {cross_edge_ratio:.1%}")
        
        # Calculate node overlap
        overlap = len(train_nodes.intersection(test_nodes))
        print(f"Node overlap: {overlap} genes ({overlap/len(G.nodes()):.1%} of all nodes)")
    
    return train_nodes, test_nodes


def degree_stratified_split(G, train_ratio, seed):
    """
    Split network while maintaining the degree distribution between train and test.
    This helps ensure that both subnetworks have similar properties.
    """
    np.random.seed(seed)
    
    # Calculate node degrees
    degrees = dict(G.degree())
    
    # Group nodes by degree
    degree_groups = {}
    for node, degree in degrees.items():
        if degree not in degree_groups:
            degree_groups[degree] = []
        degree_groups[degree].append(node)
    
    train_nodes, test_nodes = set(), set()
    
    # For each degree, split nodes according to the train ratio
    for degree, nodes in degree_groups.items():
        train_size = int(len(nodes) * train_ratio)
        
        # Shuffle nodes of this degree
        shuffled_nodes = np.random.permutation(nodes)
        
        # Add to train and test sets
        train_nodes.update(shuffled_nodes[:train_size])
        test_nodes.update(shuffled_nodes[train_size:])
    
    return train_nodes, test_nodes


def random_split(G, train_ratio, seed):
    """Simple random split of nodes"""
    nodes = list(G.nodes())
    train_nodes, test_nodes = train_test_split(
        nodes, train_size=train_ratio, random_state=seed
    )
    return set(train_nodes), set(test_nodes)


def create_subnetworks(G, train_nodes, test_nodes, expr_df, labels_df):
    """Create train and test subnetworks based on node assignments"""
    # Extract subgraphs
    train_G = G.subgraph(train_nodes).copy()
    test_G = G.subgraph(test_nodes).copy()
    
    # Create label dataframes for train and test
    train_labels = labels_df[
        (labels_df["tf_gene"].isin(train_nodes)) & 
        (labels_df["target_gene"].isin(train_nodes))
    ].copy()
    
    test_labels = labels_df[
        (labels_df["tf_gene"].isin(test_nodes)) & 
        (labels_df["target_gene"].isin(test_nodes))
    ].copy()
    
    # Create expression dataframes for train and test
    train_expr = expr_df.loc[list(train_nodes)].copy()
    test_expr = expr_df.loc[list(test_nodes)].copy()
    
    return train_G, test_G, train_labels, test_labels, train_expr, test_expr


def prepare_graphsage_data(G, train_nodes, test_nodes, expr_df, labels_df):
    """
    Prepare data for GraphSAGE by creating a special format that handles node overlap.
    For GraphSAGE, we need to ensure that test nodes can access features of their neighbors,
    even when those neighbors are in the training set.
    """
    # Calculate 1-hop neighbors of test nodes
    test_neighbors = set()
    for node in test_nodes:
        test_neighbors.update(G.predecessors(node))
        test_neighbors.update(G.successors(node))
    
    # Create combined node set for GraphSAGE
    graphsage_nodes = set(train_nodes).union(test_nodes)
    
    # Create node indices
    node_to_idx = {node: i for i, node in enumerate(graphsage_nodes)}
    
    # Create feature matrix for all nodes
    all_features = torch.tensor(expr_df.loc[list(graphsage_nodes)].values, dtype=torch.float)
    
    # Create edge indices for train and test sets
    train_edges = []
    for u, v in G.subgraph(train_nodes).edges():
        train_edges.append([node_to_idx[u], node_to_idx[v]])
    
    test_edges = []
    for u, v in G.subgraph(test_nodes).edges():
        test_edges.append([node_to_idx[u], node_to_idx[v]])
    
    # Convert to tensor
    if train_edges:
        train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    else:
        train_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
    if test_edges:
        test_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()
    else:
        test_edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Create train and test masks
    train_mask = torch.zeros(len(graphsage_nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(graphsage_nodes), dtype=torch.bool)
    
    for node in train_nodes:
        train_mask[node_to_idx[node]] = True
    
    for node in test_nodes:
        test_mask[node_to_idx[node]] = True
    
    return {
        'node_to_idx': node_to_idx,
        'idx_to_node': {idx: node for node, idx in node_to_idx.items()},
        'features': all_features,
        'train_edge_index': train_edge_index,
        'test_edge_index': test_edge_index,
        'train_mask': train_mask,
        'test_mask': test_mask
    }


def visualize_split(G, train_nodes, test_nodes, output_dir):
    """Visualize the train-test split"""
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 10))
    
    # Draw train nodes
    nx.draw_networkx_nodes(G, pos, nodelist=list(train_nodes), 
                           node_color='blue', node_size=50, alpha=0.8, label='Train')
    
    # Draw test nodes 
    nx.draw_networkx_nodes(G, pos, nodelist=list(test_nodes), 
                           node_color='red', node_size=50, alpha=0.8, label='Test')
    
    # Draw edges within train set
    train_edges = [(u, v) for u, v in G.edges() if u in train_nodes and v in train_nodes]
    nx.draw_networkx_edges(G, pos, edgelist=train_edges, width=0.5, 
                           alpha=0.5, edge_color='blue')
    
    # Draw edges within test set
    test_edges = [(u, v) for u, v in G.edges() if u in test_nodes and v in test_nodes]
    nx.draw_networkx_edges(G, pos, edgelist=test_edges, width=0.5, 
                           alpha=0.5, edge_color='red')
    
    # Draw crossing edges
    cross_edges = [(u, v) for u, v in G.edges() 
                  if (u in train_nodes and v in test_nodes) or
                     (u in test_nodes and v in train_nodes)]
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, width=0.5, 
                           alpha=0.5, edge_color='gray', style='dashed')
    
    plt.legend()
    plt.title('Train-Test Split Visualization')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_split.png'), dpi=300)
    plt.close()


def evaluate_split(G, train_nodes, test_nodes):
    """Evaluate the quality of the train-test split"""
    train_G = G.subgraph(train_nodes)
    test_G = G.subgraph(test_nodes)
    
    # Calculate edge-to-node ratios
    if len(train_nodes) > 0:
        train_edge_node_ratio = len(train_G.edges()) / len(train_nodes)
    else:
        train_edge_node_ratio = 0
        
    if len(test_nodes) > 0:
        test_edge_node_ratio = len(test_G.edges()) / len(test_nodes)
    else:
        test_edge_node_ratio = 0
    
    # Calculate node ratio
    train_node_ratio = len(train_nodes) / len(G.nodes())
    test_node_ratio = len(test_nodes) / len(G.nodes())
    
    # Calculate edge ratio
    train_edge_ratio = len(train_G.edges()) / len(G.edges()) if len(G.edges()) > 0 else 0
    test_edge_ratio = len(test_G.edges()) / len(G.edges()) if len(G.edges()) > 0 else 0
    
    # Calculate crossing edges
    crossing_edges = 0
    for u, v in G.edges():
        if (u in train_nodes and v in test_nodes) or (u in test_nodes and v in train_nodes):
            crossing_edges += 1
    
    cross_edge_ratio = crossing_edges / len(G.edges()) if len(G.edges()) > 0 else 0
    
    # Calculate node overlap
    overlap = len(train_nodes.intersection(test_nodes))
    overlap_ratio = overlap / len(G.nodes())
    
    results = {
        "nodes": {
            "train": len(train_nodes),
            "test": len(test_nodes),
            "train_ratio": train_node_ratio,
            "test_ratio": test_node_ratio,
            "overlap": overlap,
            "overlap_ratio": overlap_ratio
        },
        "edges": {
            "train": len(train_G.edges()),
            "test": len(test_G.edges()),
            "train_ratio": train_edge_ratio,
            "test_ratio": test_edge_ratio,
            "crossing": crossing_edges,
            "crossing_ratio": cross_edge_ratio
        },
        "edge_node_ratio": {
            "train": train_edge_node_ratio,
            "test": test_edge_node_ratio,
            "ratio_similarity": min(train_edge_node_ratio, test_edge_node_ratio) / 
                                max(train_edge_node_ratio, test_edge_node_ratio) if 
                                max(train_edge_node_ratio, test_edge_node_ratio) > 0 else 0
        }
    }
    
    return results


def save_split_data(output_dir, train_labels, test_labels, train_expr, test_expr, graphsage_data, evaluation):
    """Save all split data to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save label data
    train_labels.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
    test_labels.to_csv(os.path.join(output_dir, 'test_labels.csv'), index=False)
    
    # Save expression data
    train_expr.to_csv(os.path.join(output_dir, 'train_expression.csv'))
    test_expr.to_csv(os.path.join(output_dir, 'test_expression.csv'))
    
    # Save GraphSAGE data
    torch.save(graphsage_data, os.path.join(output_dir, 'graphsage_data.pt'))
    
    # Save node mappings
    with open(os.path.join(output_dir, 'node_mapping.txt'), 'w') as f:
        for node, idx in graphsage_data['node_to_idx'].items():
            f.write(f"{node}\t{idx}\n")
    
    # Save evaluation metrics
    with open(os.path.join(output_dir, 'evaluation.txt'), 'w') as f:
        f.write("Split Evaluation:\n")
        f.write("-----------------\n")
        f.write(f"Nodes: {evaluation['nodes']['train']} train, {evaluation['nodes']['test']} test (ratio: {evaluation['nodes']['train_ratio']:.2f}:{evaluation['nodes']['test_ratio']:.2f})\n")
        f.write(f"Node overlap: {evaluation['nodes']['overlap']} ({evaluation['nodes']['overlap_ratio']:.2%})\n\n")
        
        f.write(f"Edges: {evaluation['edges']['train']} train, {evaluation['edges']['test']} test (ratio: {evaluation['edges']['train_ratio']:.2f}:{evaluation['edges']['test_ratio']:.2f})\n")
        f.write(f"Crossing edges: {evaluation['edges']['crossing']} ({evaluation['edges']['crossing_ratio']:.2%})\n\n")
        
        f.write(f"Edge-to-node ratio: {evaluation['edge_node_ratio']['train']:.2f} train, {evaluation['edge_node_ratio']['test']:.2f} test\n")
        f.write(f"Ratio similarity: {evaluation['edge_node_ratio']['ratio_similarity']:.2f} (1.0 is perfect)\n")


def main():
    args = parse_args()
    
    # Load data
    expr_df, labels_df, all_genes = load_data(args)
    print(f"Loaded data: {len(all_genes)} genes, {len(labels_df)} interactions")
    
    # Build the network
    G = build_network(labels_df, all_genes)
    print(f"Built network with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Split the network based on chosen method
    if args.method == "community":
        train_nodes, test_nodes = community_based_split(G, args.train_ratio, args.max_cross_edges, args.seed)
    elif args.method == "stratified":
        train_nodes, test_nodes = degree_stratified_split(G, args.train_ratio, args.seed)
    else:  # random
        train_nodes, test_nodes = random_split(G, args.train_ratio, args.seed)
    
    # Create subnetworks
    train_G, test_G, train_labels, test_labels, train_expr, test_expr = create_subnetworks(
        G, train_nodes, test_nodes, expr_df, labels_df
    )
    
    # Prepare GraphSAGE data
    graphsage_data = prepare_graphsage_data(G, train_nodes, test_nodes, expr_df, labels_df)
    
    # Evaluate the split
    evaluation = evaluate_split(G, train_nodes, test_nodes)
    
    # Print results
    print("\nSplit Results:")
    print(f"Train set: {len(train_nodes)} nodes, {len(train_G.edges())} edges")
    print(f"Test set: {len(test_nodes)} nodes, {len(test_G.edges())} edges")
    print(f"Node overlap: {len(train_nodes.intersection(test_nodes))} genes")
    print(f"Crossing edges: {evaluation['edges']['crossing']} ({evaluation['edges']['crossing_ratio']:.2%})")
    print(f"Train edge-node ratio: {evaluation['edge_node_ratio']['train']:.2f}")
    print(f"Test edge-node ratio: {evaluation['edge_node_ratio']['test']:.2f}")
    print(f"Ratio similarity: {evaluation['edge_node_ratio']['ratio_similarity']:.2f} (1.0 is perfect)")
    
    # Save data
    save_split_data(args.output_dir, train_labels, test_labels, train_expr, test_expr, graphsage_data, evaluation)
    print(f"Saved split data to {args.output_dir}")
    
    # Visualize the split
    visualize_split(G, train_nodes, test_nodes, args.output_dir)
    print(f"Saved visualization to {args.output_dir}/network_split.png")


if __name__ == "__main__":
    main() 