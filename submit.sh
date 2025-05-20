#!/bin/bash

declare -A DATASETS=(
    # ─── Lofgof ────────────────────────────────────────────────────────────────
    ["Lofgof_mESC_TF1000+"]="data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+1000"
    ["Lofgof_mESC_TF500+"]="data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500"

    # ─── Non‑specific ──────────────────────────────────────────────────────────
    ["Non-specific_hESC_TF1000+"]="data/Benchmark Dataset/Non-Specific Dataset/hESC/TFs+1000"
    ["Non-specific_hESC_TF500+"]="data/Benchmark Dataset/Non-Specific Dataset/hESC/TFs+500"
    ["Non-specific_hHEP_TF1000+"]="data/Benchmark Dataset/Non-Specific Dataset/hHEP/TFs+1000"
    ["Non-specific_hHEP_TF500+"]="data/Benchmark Dataset/Non-Specific Dataset/hHEP/TFs+500"
    ["Non-specific_mDC_TF1000+"]="data/Benchmark Dataset/Non-Specific Dataset/mDC/TFs+1000"
    ["Non-specific_mDC_TF500+"]="data/Benchmark Dataset/Non-Specific Dataset/mDC/TFs+500"
    ["Non-specific_mESC_TF1000+"]="data/Benchmark Dataset/Non-Specific Dataset/mESC/TFs+1000"
    ["Non-specific_mESC_TF500+"]="data/Benchmark Dataset/Non-Specific Dataset/mESC/TFs+500"
    ["Non-specific_mHSC-E_TF1000+"]="data/Benchmark Dataset/Non-Specific Dataset/mHSC-E/TFs+1000"
    ["Non-specific_mHSC-E_TF500+"]="data/Benchmark Dataset/Non-Specific Dataset/mHSC-E/TFs+500"
    ["Non-specific_mHSC-GM_TF1000+"]="data/Benchmark Dataset/Non-Specific Dataset/mHSC-GM/TFs+1000"
    ["Non-specific_mHSC-GM_TF500+"]="data/Benchmark Dataset/Non-Specific Dataset/mHSC-GM/TFs+500"
    ["Non-specific_mHSC-L_TF1000+"]="data/Benchmark Dataset/Non-Specific Dataset/mHSC-L/TFs+1000"
    ["Non-specific_mHSC-L_TF500+"]="data/Benchmark Dataset/Non-Specific Dataset/mHSC-L/TFs+500"

    # ─── Sample ────────────────────────────────────────────────────────────────
    ["Sample_hESC_TF1000+"]="data/Benchmark Dataset/Sample Dataset/hESC/TFs+1000"
    ["Sample_hESC_TF500+"]="data/Benchmark Dataset/Sample Dataset/hESC/TFs+500"

    # ─── Specific ──────────────────────────────────────────────────────────────
    ["Specific_hESC_TF1000+"]="data/Benchmark Dataset/Specific Dataset/hESC/TFs+1000"
    ["Specific_hESC_TF500+"]="data/Benchmark Dataset/Specific Dataset/hESC/TFs+500"
    ["Specific_hHEP_TF1000+"]="data/Benchmark Dataset/Specific Dataset/hHEP/TFs+1000"
    ["Specific_hHEP_TF500+"]="data/Benchmark Dataset/Specific Dataset/hHEP/TFs+500"
    ["Specific_mDC_TF1000+"]="data/Benchmark Dataset/Specific Dataset/mDC/TFs+1000"
    ["Specific_mDC_TF500+"]="data/Benchmark Dataset/Specific Dataset/mDC/TFs+500"
    ["Specific_mESC_TF1000+"]="data/Benchmark Dataset/Specific Dataset/mESC/TFs+1000"
    ["Specific_mESC_TF500+"]="data/Benchmark Dataset/Specific Dataset/mESC/TFs+500"
    ["Specific_mHSC-E_TF1000+"]="data/Benchmark Dataset/Specific Dataset/mHSC-E/TFs+1000"
    ["Specific_mHSC-E_TF500+"]="data/Benchmark Dataset/Specific Dataset/mHSC-E/TFs+500"
    ["Specific_mHSC-GM_TF1000+"]="data/Benchmark Dataset/Specific Dataset/mHSC-GM/TFs+1000"
    ["Specific_mHSC-GM_TF500+"]="data/Benchmark Dataset/Specific Dataset/mHSC-GM/TFs+500"
    ["Specific_mHSC-L_TF1000+"]="data/Benchmark Dataset/Specific Dataset/mHSC-L/TFs+1000"
    ["Specific_mHSC-L_TF500+"]="data/Benchmark Dataset/Specific Dataset/mHSC-L/TFs+500"

    # ─── STRING ────────────────────────────────────────────────────────────────
    ["STRING_hESC_TF1000+"]="data/Benchmark Dataset/STRING Dataset/hESC/TFs+1000"
    ["STRING_hESC_TF500+"]="data/Benchmark Dataset/STRING Dataset/hESC/TFs+500"
    ["STRING_hHEP_TF1000+"]="data/Benchmark Dataset/STRING Dataset/hHEP/TFs+1000"
    ["STRING_hHEP_TF500+"]="data/Benchmark Dataset/STRING Dataset/hHEP/TFs+500"
    ["STRING_mDC_TF1000+"]="data/Benchmark Dataset/STRING Dataset/mDC/TFs+1000"
    ["STRING_mDC_TF500+"]="data/Benchmark Dataset/STRING Dataset/mDC/TFs+500"
    ["STRING_mESC_TF1000+"]="data/Benchmark Dataset/STRING Dataset/mESC/TFs+1000"
    ["STRING_mESC_TF500+"]="data/Benchmark Dataset/STRING Dataset/mESC/TFs+500"
    ["STRING_mHSC-E_TF1000+"]="data/Benchmark Dataset/STRING Dataset/mHSC-E/TFs+1000"
    ["STRING_mHSC-E_TF500+"]="data/Benchmark Dataset/STRING Dataset/mHSC-E/TFs+500"
    ["STRING_mHSC-GM_TF1000+"]="data/Benchmark Dataset/STRING Dataset/mHSC-GM/TFs+1000"
    ["STRING_mHSC-GM_TF500+"]="data/Benchmark Dataset/STRING Dataset/mHSC-GM/TFs+500"
    ["STRING_mHSC-L_TF1000+"]="data/Benchmark Dataset/STRING Dataset/mHSC-L/TFs+1000"
    ["STRING_mHSC-L_TF500+"]="data/Benchmark Dataset/STRING Dataset/mHSC-L/TFs+500"
)

# Loop through datasets and submit a job for each
for JOB_NAME in "${!DATASETS[@]}"
do
    DATA_PATH=${DATASETS[$JOB_NAME]}
    OUTPUT_DIR="pretrain_outputs/${JOB_NAME}"

    mkdir -p "$OUTPUT_DIR"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUTPUT_DIR}/${JOB_NAME}.out
#SBATCH --error=${OUTPUT_DIR}/${JOB_NAME}.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tcmmyoung@outlook.com
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

python -m regnet.training.pretrain \
    --expression_data "${DATA_PATH}/BL--ExpressionData.csv" \
    --label_data "${DATA_PATH}/Full_set.csv" \
    --TF_file "${DATA_PATH}/TF.csv" \
    --target_file "${DATA_PATH}/Target.csv" \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --hidden_dim 128 \
    --latent_dim 64 \
    --num_layers 2 \
    --output_dir "$OUTPUT_DIR"

EOT

    echo "Submitted job for ${JOB_NAME}"
done