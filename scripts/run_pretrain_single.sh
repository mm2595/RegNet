#!/bin/bash
#SBATCH --job-name=RegNet_single
#SBATCH --output=RegNet_%x_%j.out
#SBATCH --error=RegNet_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e
module purge
eval "$(conda shell.bash hook)"
conda activate RegNet_env
export PYTHONPATH="$(pwd):$PYTHONPATH"

python - <<'PY'
try:
    import community
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'python-louvain'])
PY

# ------------------------------------------------------------------
# Usage: sbatch scripts/run_pretrain_single.sh \
#            data/Benchmark\ Dataset/Lofgof\ Dataset/mESC/TFs+500
# ------------------------------------------------------------------
DATA_DIR="$1"
if [ -z "${DATA_DIR}" ]; then
  echo "Usage: $0 <DATA_DIR> (directory containing BL--ExpressionData.csv, TF.csv, Target.csv, Full_set.csv)" >&2
  exit 1
fi

# Derive a compact dataset identifier, e.g.
# data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500  ->  Lofgof_mESC_TF500+
DATASET_ID=$(echo "${DATA_DIR#"data/Benchmark Dataset/"}" | sed 's#/#_#g' )
OUTPUT_DIR="pretrain_outputs/${DATASET_ID}"
SPLIT_DIR="${OUTPUT_DIR}/split"
PRETRAIN_DIR="${OUTPUT_DIR}/pretrain"
VAL_SPLIT_DIR="${SPLIT_DIR}/val_split"

EXPR_FILE="${DATA_DIR}/BL--ExpressionData.csv"
LABEL_FILE="${DATA_DIR}/Full_set.csv"
TF_FILE="${DATA_DIR}/TF.csv"
TARGET_FILE="${DATA_DIR}/Target.csv"

# Create needed directories
mkdir -p "${SPLIT_DIR}" "${PRETRAIN_DIR}" "${VAL_SPLIT_DIR}"

# --- 1. Split ------------------------------------------------------
if [ ! -f "${SPLIT_DIR}/train_labels.csv" ]; then
  python scripts/train_test_split.py \
     --expression_data "${EXPR_FILE}" \
     --label_data "${LABEL_FILE}" \
     --tf_file "${TF_FILE}" \
     --target_file "${TARGET_FILE}" \
     --method random \
     --train_ratio 0.8 \
     --output_dir "${SPLIT_DIR}"  || { echo "Split failed"; exit 1; }
fi

if [ ! -f "${VAL_SPLIT_DIR}/test_labels.csv" ]; then
  python scripts/train_test_split.py \
     --expression_data "${EXPR_FILE}" \
     --label_data "${SPLIT_DIR}/train_labels.csv" \
     --tf_file "${TF_FILE}" \
     --target_file "${TARGET_FILE}" \
     --method random \
     --train_ratio 0.9 \
     --output_dir "${VAL_SPLIT_DIR}" \
     || { echo "Val split failed"; exit 1; }
fi

# quick sanity: ensure train_labels yields at least 50 mapped edges
EDGE_CNT=$(python - <<PY
import pandas as pd, os
expr = "${EXPR_FILE}"
labels = "${SPLIT_DIR}/train_labels.csv"
# Determine which columns contain gene symbols
SCORE_COLUMNS = None
try:
    sample_df = pd.read_csv(labels, nrows=1)
    if {'tf_gene', 'target_gene'}.issubset(sample_df.columns):
        TF_COL, TG_COL = 'tf_gene', 'target_gene'
    else:
        TF_COL, TG_COL = 'TF', 'Target'
except Exception:
    TF_COL, TG_COL = 'TF', 'Target'

genes = pd.read_csv(expr, index_col=0).index
gene_set = set(genes)
df = pd.read_csv(labels)
cnt = sum((row[TF_COL] in gene_set and row[TG_COL] in gene_set) for _, row in df.iterrows())
print(cnt)
PY
)

echo "Mapped edges: $EDGE_CNT"
if [ "$EDGE_CNT" -lt 50 ]; then
  echo "Dataset ${DATASET_ID}: only ${EDGE_CNT} mapped edges after split. Skipping training." >&2
  exit 0
fi

# --- 2. Pretrain ---------------------------------------------------
if [ ! -f "${PRETRAIN_DIR}/regnet_pretrained.pth" ]; then
  python -m regnet.training.pretrain \
    --expression_data "${EXPR_FILE}" \
    --label_data "${SPLIT_DIR}/train_labels.csv" \
    --val_label_data "${VAL_SPLIT_DIR}/test_labels.csv" \
    --split_data \
    --output_dir "${PRETRAIN_DIR}" \
    --epochs 50 --batch_size 1024 --hidden_dim 128 --latent_dim 64 --dropout 0.2 \
    --entropy_weight 0.01 --patience 8 --min_delta 0.0001 || { echo "Pretrain failed"; exit 1; }
fi

# --- 3. Evaluate on test set --------------------------------------
EVAL_DIR="${OUTPUT_DIR}/evaluation"
mkdir -p "${EVAL_DIR}"

if [ ! -f "${EVAL_DIR}/evaluation_predictions.csv" ]; then
  python -m regnet.training.evaluation \
    --model_path "${PRETRAIN_DIR}/regnet_pretrained.pth" \
    --expression_data "${EXPR_FILE}" \
    --label_data "${SPLIT_DIR}/test_labels.csv" \
    --split_data \
    --output_dir "${EVAL_DIR}" || { echo "Evaluation failed"; exit 1; }
fi

# --- 4. Temperature calibration -----------------------------------
TEMP_DIR="${OUTPUT_DIR}/temperature_calibration"
mkdir -p "${TEMP_DIR}"

if [ ! -f "${TEMP_DIR}/calibrated_test_predictions.csv" ]; then
  python -m regnet.tools.temperature_calibration \
     --edge_logit_csv "${EVAL_DIR}/evaluation_predictions.csv" \
     --label_csv "${SPLIT_DIR}/test_labels.csv" \
     --tf_file "${TF_FILE}" \
     --target_file "${TARGET_FILE}" \
     --output_dir "${TEMP_DIR}" || echo "Calib failed" 
fi

# --- 3. Visualise --------------------------------------------------
bash scripts/run_visualize.sh "${OUTPUT_DIR}" "${TF_FILE}" "${EXPR_FILE}"

echo "Dataset ${DATASET_ID} submitted." 