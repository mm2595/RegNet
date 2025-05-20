#!/bin/bash

declare -A PRETRAIN_TASKS=(
    ["Specific_hESC"]="Non-specific_hESC_TF500+"

)

declare -A FINETUNE_PATHS=(
    ["Specific_hESC"]="Specific Dataset/hESC/TFs+500"

)

declare -A PRETRAIN_PATHS=(
    ["Non-specific_hESC_TF500+"]="Non-Specific Dataset/hESC/TFs+500"

)

for FT_NAME in "${!PRETRAIN_TASKS[@]}"
do
    PT_NAME=${PRETRAIN_TASKS[$FT_NAME]}
    FT_REL_PATH=${FINETUNE_PATHS[$FT_NAME]}
    PT_REL_PATH=${PRETRAIN_PATHS[$PT_NAME]}

    FT_OUTPUT_DIR="finetune_outputs/${FT_NAME}"
    PT_OUTPUT_DIR="pretrain_outputs/${PT_NAME}"

    mkdir -p "$FT_OUTPUT_DIR"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${FT_NAME}_finetune
#SBATCH --output=${FT_OUTPUT_DIR}/${FT_NAME}.out
#SBATCH --error=${FT_OUTPUT_DIR}/${FT_NAME}.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00  

python3 -m regnet.training.finetune2 \
    --expression_data "data/Benchmark Dataset/${FT_REL_PATH}/BL--ExpressionData.csv" \
    --TF_file "data/Benchmark Dataset/${FT_REL_PATH}/TF.csv" \
    --target_file "data/Benchmark Dataset/${FT_REL_PATH}/Target.csv" \
    --pretrain_expression_data "data/Benchmark Dataset/${PT_REL_PATH}/BL--ExpressionData.csv" \
    --pretrain_adj "${PT_OUTPUT_DIR}/adjacency_matrix.csv" \
    --finetune_full_set "data/Benchmark Dataset/${FT_REL_PATH}/Full_set.csv" \
    --pretrain_fisher_file "${PT_OUTPUT_DIR}/fisher_diag.pth" \
    --pretrained_model "${PT_OUTPUT_DIR}/regnet_pretrained.pth" \
    --pretrain_output_dir "${PT_OUTPUT_DIR}" \
    --output_dir "${FT_OUTPUT_DIR}" \
    --batch_size 1024 \
    --epochs 50 \
    --hidden_dim 128 \
    --latent_dim 64 \
    --num_layers 2 \

EOT

    echo "Submitted fine-tune job for ${FT_NAME}"
done
