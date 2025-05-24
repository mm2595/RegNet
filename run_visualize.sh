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
TF_FILE="pretrain_outputs/Lofgof_mESC_TF500+/TF.csv"
EXPRESSION_CSV="data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500/BL--ExpressionData.csv"

# Make sure module can be found
export PYTHONPATH=.:$PYTHONPATH

python -m regnet.visualization.visualize_regnet \
  --pretrain_dir "${OUTPUT_DIR}" \
  --tf_file "${TF_FILE}" \
  --expression_csv "${EXPRESSION_CSV}" \
  --method umap

echo "Visualisation complete. Figures are in ${OUTPUT_DIR}" 