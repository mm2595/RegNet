#!/usr/bin/env python
"""
Evaluation script for RegNet models
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

from regnet.models.regnet import RegNet
from regnet.preprocessing.data_loader import create_batches_NC
from regnet.training.pretrain import build_edge_index_from_split, build_edge_index

def parse_args():
    p = argparse.ArgumentParser("Evaluate RegNet")
    p.add_argument("--model_path", required=True, help="Path to pretrained model weights")
    p.add_argument("--expression_data", required=True, help="Expression data (CSV)")
    p.add_argument("--label_data", required=True, help="Label data (CSV)")
    p.add_argument("--TF_file", default=None, help="TF mapping file (regular format)")
    p.add_argument("--target_file", default=None, help="Target mapping file (regular format)")
    p.add_argument("--split_data", action="store_true", help="Use data from train_test_split.py format")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--hidden_dim", type=int, default=128, help="Must match pretrained model")
    p.add_argument("--latent_dim", type=int, default=64, help="Must match pretrained model")
    p.add_argument("--num_layers", type=int, default=2, help="Must match pretrained model")
    p.add_argument("--output_dir", default="evaluation_results")
    p.add_argument("--apply_temperature", action="store_true", help="Apply temperature scaling")
    p.add_argument("--temp_value", type=float, default=1.0, help="Temperature value (if not auto-optimized)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def evaluate_model(model, loader, device):
    """Evaluate model on a data loader"""
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    all_var = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits, _, _, mu, logvar = model(data.x, data.edge_index, data.edge_index.t())
            
            # Get edge variance
            node_pair = data.edge_index
            var_log = model.edge_variance(mu, logvar, node_pair.t())
            
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_labels.append(data.edge_attr.float().cpu())
            all_logits.append(logits.detach().cpu())
            all_var.append(var_log.detach().cpu())
    
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    logits = torch.cat(all_logits)
    variances = torch.cat(all_var)
    
    return preds, labels, logits, variances

def compute_metrics(preds, labels):
    """Compute performance metrics"""
    p, y = preds.numpy(), labels.numpy()
    
    # Basic metrics
    aupr = average_precision_score(y, p)
    auroc = roc_auc_score(y, p)
    
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, p)
    
    # Find optimal F1 threshold
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    binary_preds = (p >= optimal_threshold).astype(float)
    
    # True/False positives/negatives
    tp = ((binary_preds == 1) & (y == 1)).sum()
    fp = ((binary_preds == 1) & (y == 0)).sum()
    tn = ((binary_preds == 0) & (y == 0)).sum()
    fn = ((binary_preds == 0) & (y == 1)).sum()
    
    # Calculate metrics from confusion matrix
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_at_threshold = 2 * precision_at_threshold * recall_at_threshold / (precision_at_threshold + recall_at_threshold) if (precision_at_threshold + recall_at_threshold) > 0 else 0
    
    metrics = {
        "AUPR": aupr,
        "AUROC": auroc,
        "Optimal_Threshold": optimal_threshold,
        "Accuracy": accuracy,
        "Precision": precision_at_threshold,
        "Recall": recall_at_threshold,
        "F1": f1_at_threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }
    
    curve_data = {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds
    }
    
    return metrics, curve_data

def apply_temperature_scaling(logits, temperature):
    """Apply temperature scaling to logits"""
    return torch.sigmoid(logits / temperature)

def plot_pr_curve(precision, recall, metrics, output_path):
    """Plot precision-recall curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'AUPR = {metrics["AUPR"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def save_predictions(preds, labels, variances, gene_names, edge_index, out_dir):
    """Save predictions and variance to file"""
    
    # Get gene pairs from edge_index
    tf_idx = edge_index[0].cpu().numpy()
    target_idx = edge_index[1].cpu().numpy()
    
    # Ensure all arrays have the same length
    min_length = min(len(tf_idx), len(target_idx), len(preds), len(labels), len(variances))
    
    # Use consistent lengths for all arrays
    tf_idx = tf_idx[:min_length]
    target_idx = target_idx[:min_length]
    preds_np = preds.numpy()[:min_length]
    labels_np = labels.numpy()[:min_length]
    variances_np = variances.numpy()[:min_length]
    
    # Map indices to gene names
    tf_genes = [gene_names[i] for i in tf_idx]
    target_genes = [gene_names[i] for i in target_idx]
    
    # Create dataframe with predictions
    df = pd.DataFrame({
        "TF": tf_genes,
        "Target": target_genes,
        "Label": labels_np,
        "Prediction": preds_np,
        "Variance": variances_np
    })
    
    # Save to CSV
    df.to_csv(os.path.join(out_dir, "evaluation_predictions.csv"), index=False)
    
    return df

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load expression data
    expr_df = pd.read_csv(args.expression_data, index_col=0)
    gene_to_idx = {g: i for i, g in enumerate(expr_df.index)}
    gene_names = expr_df.index.tolist()
    
    # Load edge data based on format
    if args.split_data:
        print("Loading data from train_test_split format...")
        edge_index, edge_labels = build_edge_index_from_split(args.label_data, gene_to_idx)
    else:
        print("Loading data from regular format...")
        if args.TF_file is None or args.target_file is None:
            raise ValueError("TF_file and target_file must be provided when not using split_data")
        edge_index, edge_labels = build_edge_index(args.label_data, args.TF_file, args.target_file, gene_to_idx)
    
    # Create data loader
    loader = create_batches_NC(expr_df, edge_index, edge_labels, args.batch_size)
    
    # Initialize model
    model = RegNet(expr_df.shape[1], args.hidden_dim, args.latent_dim, args.num_layers).to(args.device)
    
    # Load pretrained weights
    print(f"Loading pretrained model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    
    # Evaluate model
    print("Evaluating model...")
    preds, labels, logits, variances = evaluate_model(model, loader, args.device)
    
    # Apply temperature scaling if requested
    if args.apply_temperature:
        print(f"Applying temperature scaling (T={args.temp_value})...")
        preds = apply_temperature_scaling(logits, args.temp_value)
    
    # Compute metrics
    print("Computing evaluation metrics...")
    metrics, curve_data = compute_metrics(preds, labels)
    
    # Save results
    print("Saving evaluation results...")
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
            print(f"{key}: {value}")
    
    # Plot PR curve
    plot_pr_curve(curve_data["precision"], curve_data["recall"], metrics, 
                 os.path.join(args.output_dir, "pr_curve.png"))
    
    # Save predictions
    save_predictions(preds, labels, variances, gene_names, edge_index, args.output_dir)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 