import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def split(label_file, tf_file, target_file, output_dir, test_size=0.2, hub_ratio=0.1, random_state=42):
    np.random.seed(random_state)
    
    # Load data files
    label_df = pd.read_csv(label_file)
    tf_df = pd.read_csv(tf_file)
    target_df = pd.read_csv(target_file)

    # Check for common column issues and fix them
    if 'index' not in tf_df.columns and 'index' in tf_df.columns[1:]:
        tf_df.rename(columns={tf_df.columns[1]: 'index'}, inplace=True)
    
    if 'TF' not in tf_df.columns and 'TF' in tf_df.columns[1:]:
        tf_df.rename(columns={tf_df.columns[1]: 'TF'}, inplace=True)
    
    if 'Gene' not in target_df.columns and 'Gene' in target_df.columns[1:]:
        target_df.rename(columns={target_df.columns[1]: 'Gene'}, inplace=True)
    
    if 'index' not in target_df.columns and 'index' in target_df.columns[1:]:
        target_df.rename(columns={target_df.columns[1]: 'index'}, inplace=True)
    
    # Create mappings from ID to gene name
    tf_id_to_name = dict(zip(tf_df['index'], tf_df['TF']))
    target_id_to_name = dict(zip(target_df['index'], target_df['Gene']))
    
    # Add gene names to label dataframe if needed
    if 'TF' in label_df.columns and not label_df['TF'].apply(lambda x: isinstance(x, str)).any():
        # If TF column exists but contains numeric indices
        label_df['TF'] = label_df['TF'].map(tf_id_to_name)
    
    if 'Target' in label_df.columns and not label_df['Target'].apply(lambda x: isinstance(x, str)).any():
        # If Target column exists but contains numeric indices
        label_df['Target'] = label_df['Target'].map(target_id_to_name)
    
    # Count the number of positive edges per TF
    tf_pos_counts = label_df[label_df['Label'] == 1]['TF'].value_counts()
    
    # Identify hub TFs (those with many connections)
    hub_threshold = np.quantile(tf_pos_counts, 1 - hub_ratio)
    hub_tfs = set(tf_pos_counts[tf_pos_counts >= hub_threshold].index)
    
    # All other TFs are regular
    regular_tfs = set(tf_df['TF']) - hub_tfs
    
    print(f"Identified {len(hub_tfs)} hub TFs out of {len(tf_df)} total TFs")
    print(f"Hub threshold: {hub_threshold} edges")

    train_edges, test_edges = [], []

    def split_edges(tf_group):
        pos_edges = tf_group[tf_group['Label'] == 1]
        neg_edges = tf_group[tf_group['Label'] == 0]

        if len(pos_edges) > 0:
        pos_train, pos_test = train_test_split(
            pos_edges, test_size=test_size, random_state=random_state, shuffle=True
        )
            train_edges.append(pos_train)
            test_edges.append(pos_test)

        if len(neg_edges) > 0:
        neg_train, neg_test = train_test_split(
            neg_edges, test_size=test_size, random_state=random_state, shuffle=True
        )
            train_edges.append(neg_train)
            test_edges.append(neg_test)

    # Split edges for hub TFs
    print("Splitting edges for hub TFs...")
    for tf in hub_tfs:
        tf_edges = label_df[label_df['TF'] == tf]
        if len(tf_edges) > 0:
            split_edges(tf_edges)

    # Split edges for regular TFs
    print("Splitting edges for regular TFs...")
    for tf in regular_tfs:
        tf_edges = label_df[label_df['TF'] == tf]
        if len(tf_edges) > 0:
            split_edges(tf_edges)

    # Combine all edges
    if train_edges and test_edges:
    train_df = pd.concat(train_edges).reset_index(drop=True)
    test_df = pd.concat(test_edges).reset_index(drop=True)
    else:
        print("Error: No edges found for splitting!")
        return

    # Check for genes that appear only in the test set
    train_genes = set(train_df['TF']).union(train_df['Target'])
    test_genes = set(test_df['TF']).union(test_df['Target'])
    test_only_genes = test_genes - train_genes

    if test_only_genes:
        print(f"Warning: Found {len(test_only_genes)} genes appearing only in test set")
        print("Reallocating some edges explicitly to maintain gene consistency...")
        for gene in test_only_genes:
            gene_edges = test_df[(test_df['TF'] == gene) | (test_df['Target'] == gene)]
            realloc_edges = gene_edges.sample(frac=0.5, random_state=random_state)
            train_df = pd.concat([train_df, realloc_edges])
            test_df = test_df.drop(realloc_edges.index)

    # Save files
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_labels.csv"), index=False)

    print("Sophisticated train-test split completed.")
    print(f"Final Train edges: {len(train_df)}, Final Test edges: {len(test_df)}")
    print(f"Positive train edges: {train_df['Label'].sum()}, Positive test edges: {test_df['Label'].sum()}")
    
    # Print class balance
    train_pos_pct = (train_df['Label'].sum() / len(train_df)) * 100
    test_pos_pct = (test_df['Label'].sum() / len(test_df)) * 100
    print(f"Train positive %: {train_pos_pct:.2f}%, Test positive %: {test_pos_pct:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced train-test splitting for GRNs")
    parser.add_argument("--label_file", required=True, help="Path to label CSV")
    parser.add_argument("--tf_file", required=True, help="Path to TF CSV")
    parser.add_argument("--target_file", required=True, help="Path to Target CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory for splits")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--hub_ratio", type=float, default=0.1, help="Proportion of hub TFs")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    split(
        args.label_file, args.tf_file, args.target_file, 
        args.output_dir, args.test_size, args.hub_ratio, args.random_state
    )
