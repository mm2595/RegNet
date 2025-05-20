#!/bin/bash
#SBATCH --job-name=Lofgof_mESC
#SBATCH --output=Lofgof_mESC.out
#SBATCH --error=Lofgof_mESC.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tcmmyoung@outlook.com
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

DATA_PATH="data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+1000/"
OUTPUT_DIR="outputs"

python regnet/training/pretrain.py \
    --expression_data "$DATA_PATH/BL--ExpressionData.csv" \
    --label_data "$DATA_PATH/Full_set.csv" \
    --TF_file "$DATA_PATH/TF.csv" \
    --target_file "$DATA_PATH/Target.csv" \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --hidden_dim 128 \
    --latent_dim 64 \
    --num_layers 2 \
    --output_dir "$OUTPUT_DIR"

