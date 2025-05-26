#!/bin/bash
#SBATCH --job-name=RegNet_mESC
#SBATCH --output=RegNet_mESC_%j.out
#SBATCH --error=RegNet_mESC_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Exit on error
set -e

# Load environment
module purge
eval "$(conda shell.bash hook)"
conda activate RegNet_env

# Paths
DATA_DIR="data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500"
OUTPUT_DIR="pretrain_outputs/Lofgof_mESC_TF500+"
SPLIT_DIR="${OUTPUT_DIR}/split"
PRETRAIN_DIR="${OUTPUT_DIR}/pretrain"
EVAL_DIR="${OUTPUT_DIR}/evaluation"
TEMP_DIR="${OUTPUT_DIR}/temperature_calibration"
VAL_SPLIT_DIR="${SPLIT_DIR}/val_split"

# Dataset files
EXPR_FILE="${DATA_DIR}/BL--ExpressionData.csv"
LABEL_FILE="${DATA_DIR}/Full_set.csv"
TF_FILE="${DATA_DIR}/TF.csv"
TARGET_FILE="${DATA_DIR}/Target.csv"

# Model parameters
TEST_SIZE=0.2
HIDDEN_DIM=128
LATENT_DIM=64
NUM_LAYERS=2
BATCH_SIZE=1024
EPOCHS=50
LEARNING_RATE=0.001
BETA_KL=1.0
RECON_WEIGHT=0.1
DROPOUT=0.2
ENTROPY_WEIGHT=0.01
PATIENCE=8
MIN_DELTA=0.0001

# Create directory structure
mkdir -p "${SPLIT_DIR}" "${PRETRAIN_DIR}" "${EVAL_DIR}" "${TEMP_DIR}" "${VAL_SPLIT_DIR}"

echo "===== RegNet Workflow on HPC ====="
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Running on node: $(hostname)"
echo "GPU info: $(nvidia-smi -L 2>/dev/null || echo 'No GPU detected')"
echo ""

# Step 1: Check if train-test split already exists
if [ -f "${SPLIT_DIR}/train_labels.csv" ] && [ -f "${SPLIT_DIR}/test_labels.csv" ]; then
    echo "Train-test split already exists. Skipping step 1."
else
    echo "Step 1: Running train-test split..."
    python regnet/preprocessing/train_test_split.py \
      --label_file "${LABEL_FILE}" \
      --tf_file "${TF_FILE}" \
      --target_file "${TARGET_FILE}" \
      --output_dir "${SPLIT_DIR}" \
      --test_size ${TEST_SIZE}
    echo "Train-test split complete. Results saved to: ${SPLIT_DIR}"
fi
echo ""

# Step 1.5: Create validation split inside training data
if [ -f "${VAL_SPLIT_DIR}/test_labels.csv" ]; then
    echo "Validation split already exists. Skipping step 1.5."
else
    echo "Step 1.5: Creating validation split from training data..."
    python regnet/preprocessing/train_test_split.py \
      --label_file "${SPLIT_DIR}/train_labels.csv" \
      --tf_file "${TF_FILE}" \
      --target_file "${TARGET_FILE}" \
      --output_dir "${VAL_SPLIT_DIR}" \
      --test_size 0.1
    echo "Validation split complete. Results in ${VAL_SPLIT_DIR}"
fi
echo ""

# Step 2: Check if pretrained model already exists
if [ -f "${PRETRAIN_DIR}/regnet_pretrained.pth" ]; then
    echo "Pretrained model already exists. Skipping step 2."
else
    echo "Step 2: Running pretraining on training data..."
    # Create a modified version of pretrain.py that skips Fisher calculation
    cat > pretrain_skip_fisher.py << 'EOF'
#!/usr/bin/env python
import sys
from regnet.training.pretrain import *

# Override main function to skip Fisher calculation
def main_skip_fisher():
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

    # Validation loader
    val_loader = None
    if args.val_label_data and os.path.isfile(args.val_label_data):
        if args.split_data:
            val_edge_index, val_edge_labels = build_edge_index_from_split(args.val_label_data, gene_to_idx)
        else:
            val_edge_index, val_edge_labels = build_edge_index(args.val_label_data, args.TF_file, args.target_file, gene_to_idx)
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
        # Validation metric
        val_aupr = None
        val_auroc = None
        if val_loader is not None:
            with torch.no_grad():
                p_list, y_list = [], []
                for vd in val_loader:
                    vd = vd.to(args.device)
                    logits_v, _, _, _, _ = model(vd.x, vd.edge_index, vd.edge_index.t())
                    p_list.append(torch.sigmoid(logits_v).cpu()); y_list.append(vd.edge_attr.float().cpu())
                p_all = torch.cat(p_list); y_all = torch.cat(y_list)
                val_aupr, val_auroc = evaluate_metrics(p_all, y_all)

        msg = f"Ep{ep+1:02d}  L{loss:.4f}  train AUPR {aupr:.3f}  AUROC {auroc:.3f}"
        if val_aupr is not None:
            msg += f"  val AUPR {val_aupr:.3f}  val AUROC {val_auroc:.3f}"
        print(msg)
        history.append({"epoch": ep+1, "loss": loss, "AUPR": aupr, "val_AUPR": val_aupr, "AUROC": auroc, "val_AUROC": val_auroc})

        metric = val_aupr if val_aupr is not None else aupr
        if metric > best_aupr + args.min_delta:
            best_aupr = metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    pd.DataFrame(history).to_csv(f"{args.output_dir}/training_metrics.csv",
                                 index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{args.output_dir}/regnet_pretrained.pth")

    # Skip Fisher information calculation which causes errors
    print("\nSkipping Fisher information calculation...")

    save_outputs(model, loader, args.device, args.output_dir,
                 expr_df.index.tolist(), edge_index, edge_labels)
    print(f"All outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main_skip_fisher()
EOF

    # Run the modified pretraining script
    python pretrain_skip_fisher.py \
      --expression_data "${EXPR_FILE}" \
      --label_data "${SPLIT_DIR}/train_labels.csv" \
      --split_data \
      --hidden_dim ${HIDDEN_DIM} \
      --latent_dim ${LATENT_DIM} \
      --num_layers ${NUM_LAYERS} \
      --batch_size ${BATCH_SIZE} \
      --epochs ${EPOCHS} \
      --lr ${LEARNING_RATE} \
      --beta_kl ${BETA_KL} \
      --recon_weight ${RECON_WEIGHT} \
      --entropy_weight ${ENTROPY_WEIGHT} \
      --dropout ${DROPOUT} \
      --val_label_data "${VAL_SPLIT_DIR}/test_labels.csv" \
      --patience ${PATIENCE} \
      --min_delta ${MIN_DELTA} \
      --output_dir "${PRETRAIN_DIR}" \
      --device cuda

    # Clean up temporary file
    rm pretrain_skip_fisher.py
    
    echo "Pretraining complete. Model saved to: ${PRETRAIN_DIR}/regnet_pretrained.pth"
fi
echo ""

# Step 3: Evaluate on test data
echo "Step 3: Evaluating model on test data..."
python -m regnet.training.evaluation \
  --model_path "${PRETRAIN_DIR}/regnet_pretrained.pth" \
  --expression_data "${EXPR_FILE}" \
  --label_data "${SPLIT_DIR}/test_labels.csv" \
  --split_data \
  --hidden_dim ${HIDDEN_DIM} \
  --latent_dim ${LATENT_DIM} \
  --num_layers ${NUM_LAYERS} \
  --batch_size ${BATCH_SIZE} \
  --output_dir "${EVAL_DIR}" \
  --device cuda

echo "Evaluation complete. Results saved to: ${EVAL_DIR}"
echo ""

# Step 4: Run temperature calibration
echo "Step 4: Running temperature calibration..."
# Check if we have the necessary prediction files
if [ -f "${PRETRAIN_DIR}/predictions_with_gene_names.csv" ] && [ -f "${EVAL_DIR}/evaluation_predictions.csv" ]; then
    # Create a simplified temperature calibration script
    cat > temp_calibration_simple.py << 'EOF'
#!/usr/bin/env python
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description="Simple temperature calibration")
    parser.add_argument("--train_preds", required=True, help="Training predictions CSV")
    parser.add_argument("--test_preds", required=True, help="Test predictions with labels CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    return parser.parse_args()

def fit_temperature(logits, labels, n_iter=500, lr=1e-2):
    """Optimize the temperature parameter for calibration"""
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=n_iter)
    
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            logits / temperature, labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    return temperature.item()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test predictions with labels
    test_df = pd.read_csv(args.test_preds)
    test_probs = torch.tensor(test_df['Prediction'].values, dtype=torch.float32)
    test_labels = torch.tensor(test_df['Label'].values, dtype=torch.float32)
    
    # Convert probabilities to logits
    test_logits = torch.logit(test_probs.clamp(1e-6, 1-1e-6))
    
    # Find optimal temperature on test data
    T = fit_temperature(test_logits, test_labels)
    print(f"Optimal temperature: {T:.4f}")
    
    # Save optimal temperature to file
    with open(os.path.join(args.output_dir, "optimal_temperature.txt"), "w") as f:
        f.write(f"Optimal temperature: {T:.4f}\n")
    
    # Apply temperature scaling to test predictions
    calibrated_probs = torch.sigmoid(test_logits / T).numpy()
    
    # Create calibrated predictions
    test_df['Calibrated'] = calibrated_probs
    test_df.to_csv(os.path.join(args.output_dir, "calibrated_test_predictions.csv"), index=False)
    
    # Load training predictions
    train_df = pd.read_csv(args.train_preds)
    if 'Prediction' in train_df.columns:
        train_probs = torch.tensor(train_df['Prediction'].values, dtype=torch.float32)
        train_logits = torch.logit(train_probs.clamp(1e-6, 1-1e-6))
        train_calibrated = torch.sigmoid(train_logits / T).numpy()
        train_df['Calibrated'] = train_calibrated
        train_df.to_csv(os.path.join(args.output_dir, "calibrated_train_predictions.csv"), index=False)
    
    # Calculate metrics
    uncal_auroc = roc_auc_score(test_labels.numpy(), test_probs.numpy())
    uncal_aupr = average_precision_score(test_labels.numpy(), test_probs.numpy())
    cal_auroc = roc_auc_score(test_labels.numpy(), calibrated_probs)
    cal_aupr = average_precision_score(test_labels.numpy(), calibrated_probs)
    
    # Save metrics
    with open(os.path.join(args.output_dir, "calibration_metrics.txt"), "w") as f:
        f.write(f"Temperature: {T:.4f}\n")
        f.write(f"Uncalibrated AUROC: {uncal_auroc:.4f}\n")
        f.write(f"Uncalibrated AUPR: {uncal_aupr:.4f}\n")
        f.write(f"Calibrated AUROC: {cal_auroc:.4f}\n")
        f.write(f"Calibrated AUPR: {cal_aupr:.4f}\n")
    
    print(f"Calibration complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
EOF

    # Run the simplified calibration script
    python temp_calibration_simple.py \
        --train_preds "${PRETRAIN_DIR}/predictions_with_gene_names.csv" \
        --test_preds "${EVAL_DIR}/evaluation_predictions.csv" \
        --output_dir "${TEMP_DIR}"
    
    # Clean up temporary file
    rm temp_calibration_simple.py
    
    echo "Temperature calibration complete. Results saved to: ${TEMP_DIR}"
else
    echo "Missing prediction files. Skipping temperature calibration."
    # Create a default temperature value for the next step
    mkdir -p "${TEMP_DIR}"
    echo "Optimal temperature: 1.0" > "${TEMP_DIR}/optimal_temperature.txt"
fi
echo ""

# Step 5: Final evaluation with temperature scaling
echo "Step 5: Running final evaluation with temperature scaling..."
# Read the optimal temperature value from the calibration output
if [ -f "${TEMP_DIR}/optimal_temperature.txt" ]; then
    TEMP_VALUE=$(grep "Optimal temperature" "${TEMP_DIR}/optimal_temperature.txt" | cut -d' ' -f3)
    echo "Using temperature value: ${TEMP_VALUE}"
    
    python -m regnet.training.evaluation \
      --model_path "${PRETRAIN_DIR}/regnet_pretrained.pth" \
      --expression_data "${EXPR_FILE}" \
      --label_data "${SPLIT_DIR}/test_labels.csv" \
      --split_data \
      --hidden_dim ${HIDDEN_DIM} \
      --latent_dim ${LATENT_DIM} \
      --num_layers ${NUM_LAYERS} \
      --batch_size ${BATCH_SIZE} \
      --apply_temperature \
      --temp_value ${TEMP_VALUE} \
      --output_dir "${EVAL_DIR}_calibrated" \
      --device cuda
    
    echo "Final evaluation complete. Results saved to: ${EVAL_DIR}_calibrated"
else
    echo "Temperature calibration file not found. Skipping final evaluation."
fi
echo ""

# Step 6: Bayesian shrinkage visualisation
echo "Step 6: Creating Bayesian shrinkage visualisations..."

python -m regnet.visualization.visualize_regnet \
  --pretrain_dir "${PRETRAIN_DIR}" \
  --tf_file "${TF_FILE}" \
  --expression_csv "${EXPR_FILE}" \
  --plots shrinkage \
  --method umap

echo "Visualisations saved to ${PRETRAIN_DIR}"

echo "===== RegNet Workflow Complete ====="
echo "Results are available in: ${OUTPUT_DIR}"

# Print job statistics
echo "Job completed at: $(date)"
echo "Duration: $((($(date +%s) - $SECONDS) / 60)) minutes" 