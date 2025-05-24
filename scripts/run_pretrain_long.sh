#!/bin/bash
#SBATCH --job-name=RegNet_mESC_long
#SBATCH --output=RegNet_mESC_long_%j.out
#SBATCH --error=RegNet_mESC_long_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e
module purge
eval "$(conda shell.bash hook)"
conda activate RegNet_env

# ensure project root is on PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PWD"

DATA_DIR="data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500"
OUTPUT_DIR="pretrain_outputs/Lofgof_mESC_TF500+"
PRETRAIN_DIR="${OUTPUT_DIR}/pretrain"
SPLIT_DIR="${OUTPUT_DIR}/split"
VAL_SPLIT_DIR="${SPLIT_DIR}/val_split"
mkdir -p "${SPLIT_DIR}" "${PRETRAIN_DIR}" "${VAL_SPLIT_DIR}"

EXPR_FILE="${DATA_DIR}/BL--ExpressionData.csv"
LABEL_FILE="${DATA_DIR}/Full_set.csv"
TF_FILE="${DATA_DIR}/TF.csv"
TARGET_FILE="${DATA_DIR}/Target.csv"

# create split if missing
if [ ! -f "${SPLIT_DIR}/train_labels.csv" ]; then
  python regnet/preprocessing/train_test_split.py \
    --label_file "${LABEL_FILE}" \
    --tf_file "${TF_FILE}" \
    --target_file "${TARGET_FILE}" \
    --output_dir "${SPLIT_DIR}" \
    --test_size 0.2
fi

if [ ! -f "${VAL_SPLIT_DIR}/test_labels.csv" ]; then
  python regnet/preprocessing/train_test_split.py \
    --label_file "${SPLIT_DIR}/train_labels.csv" \
    --tf_file "${TF_FILE}" \
    --target_file "${TARGET_FILE}" \
    --output_dir "${VAL_SPLIT_DIR}" \
    --test_size 0.1
fi

echo "Starting long pretrain..."
python regnet/training/pretrain.py \
  --expression_data "${EXPR_FILE}" \
  --label_data "${SPLIT_DIR}/train_labels.csv" \
  --val_label_data "${VAL_SPLIT_DIR}/test_labels.csv" \
  --split_data \
  --TF_file "${TF_FILE}" \
  --target_file "${TARGET_FILE}" \
  --output_dir "${PRETRAIN_DIR}" \
  --hidden_dim 128 \
  --latent_dim 64 \
  --num_layers 2 \
  --batch_size 1024 \
  --epochs 300 \
  --lr 0.001 \
  --beta_kl 1.0 \
  --recon_weight 0.1 \
  --entropy_weight 0.01 \
  --dropout 0.2 \
  --patience 15 \
  --min_delta 0.0001 \
  --device cuda

echo "Long pretraining finished." 