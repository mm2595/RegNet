#!/bin/bash
#SBATCH --job-name=RegNet_viz
#SBATCH --output=RegNet_viz_%j.out
#SBATCH --error=RegNet_viz_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

set -e
module purge
eval "$(conda shell.bash hook)"
conda activate RegNet_env

OUTPUT_DIR="pretrain_outputs/Lofgof_mESC_TF500+/pretrain"
TF_FILE="data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500/TF.csv"
DATA_DIR="data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500"

python -m regnet.visualization.visualize_regnet \
  --pretrain_dir "${OUTPUT_DIR}" \
  --tf_file "${TF_FILE}" \
  --expression_csv "${DATA_DIR}/BL--ExpressionData.csv" \
  --method umap

echo "Visualisation complete. Figures are in ${OUTPUT_DIR}" 