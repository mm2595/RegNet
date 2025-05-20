# RegNet: Graph-Based Gene Regulatory Network Inference

RegNet is a modular graph neural network framework for inferring gene regulatory networks (GRNs) by integrating gene expression data and known regulatory priors.

## Key Features

- Modular architecture combining:
  - **GraphSAGE + MLP** encoder for local neighborhood aggregation
  - **Attention** module for weighted message passing
  - **Variational Autoencoder (VAE)** for capturing nonlinear latent structures
- End-to-end pipeline from raw expression to edge probability predictions
- Two-stage training:
  1. **Pretraining** (supervised): uses known labels to learn initial representations
  2. **Fine-tuning** (self-supervised): refines embeddings without labels
- Visualization utilities for embeddings, attention weights, and VAE reconstructions

## Directory Structure

```
regnet/
├── preprocessing/           # Data loading, normalization, and dataset splitting
│   ├── data_loader.py
│   ├── train_test_split.py
│   └── utils.py
├── models/                  # Core model definitions
│   ├── graphsage.py         # GraphSAGE + MLP encoder
│   ├── attention.py         # Graph attention layer
│   ├── vae.py               # Variational Autoencoder module
│   └── regnet.py            # High-level model integrating all submodules
├── training/                # Training scripts and loss functions
│   ├── pretrain.py          # Supervised pretraining pipeline
│   ├── finetune.py          # Self-supervised fine-tuning pipeline
│   ├── finetune2.py         # Alternate fine-tuning variant
│   └── loss_functions.py    # Custom loss definitions
├── visualization/           # Plotting and analysis scripts
│   ├── plot_embeddings.py
│   ├── plot_attention.py
│   └── plot_vae.py
└── README.md                # This file
```

## Model Architecture Overview

1. **GraphSAGE + MLP Encoder**
   - Aggregates local neighborhood features from the adjacency matrix
   - Projects aggregated features through an MLP to generate initial node embeddings

2. **Attention Module**
   - Learns attention coefficients for each edge to emphasize important interactions
   - Refines node embeddings via weighted neighbor aggregation

3. **Variational Autoencoder (VAE)**
   - Encodes embeddings into a probabilistic latent space
   - Decodes to reconstruct original features, enforcing robust representations

4. **Readout Layer**
   - Computes pairwise combinations of node embeddings
   - Outputs edge probability scores for all gene pairs

## Training Pipeline

### 1. Pretraining (Supervised)

- Script: `training/pretrain.py`
- Objective: Binary cross-entropy between predicted edge probabilities and ground-truth labels (from `Full_set.csv`)
- Inputs:
  - Gene expression matrix (`BL--ExpressionData.csv`)
  - Adjacency labels (`Full_set.csv`)
  - TF/Target index mappings (`TF.csv`, `Target.csv`)
- Outputs: Saved model checkpoints in `pretrain_outputs/`
- Note: Pretrained weights and training logs are already available in `pretrain_outputs/`; you can skip re-running pretraining if desired.
- **Pretraining Outputs Structure**:
  The `pretrain_outputs/` directory contains subfolders named `<Dataset>_<CellType>_TF<size>+/` (e.g., `Specific_hESC_TF1000+/`). Each subfolder includes:
    - `adjacency_matrix.csv`: processed adjacency matrix used for training
    - `regnet_pretrained.pth`: pretrained model checkpoint (PyTorch .pth file)
    - `training_metrics.csv`: epoch-wise training loss and accuracy logs
    - `predictions_with_gene_names.csv`: predicted edge probabilities mapped to gene names
    - `graphsage_embeddings.csv`, `vae_embeddings.csv`, `gate_values.csv`: saved intermediate embeddings and gate values
    - Optional prototype files (`positive_prototype.npy`, `negative_prototype.npy`) and distance metrics (`euclidean_distance.npy`, `cosine_similarity.npy`)
  Example: For Specific hESC TFs+1000, see:
  ```
  pretrain_outputs/Specific_hESC_TF1000+/adjacency_matrix.csv
  pretrain_outputs/Specific_hESC_TF1000+/regnet_pretrained.pth
  ```

### 2. Fine-Tuning (Self-Supervised)

- Script: `training/finetune.py` (or `finetune2.py`)
- Objective: Self-supervised reconstruction or contrastive loss (no label usage during training)
- Inputs:
  - Pretrained model weights
  - Gene expression matrix
- Evaluation: Ground-truth labels are only used to assess model performance post-training
- Outputs: Fine-tuned checkpoints in `finetune_outputs/`

## Usage Examples

### Pretrain RegNet
```bash
python regnet/training/pretrain.py \
  --expr_path path/to/BL--ExpressionData.csv \
  --edge_path path/to/Full_set.csv \
  --tf_map path/to/TF.csv \
  --target_map path/to/Target.csv \
  --output_dir pretrain_outputs/
```

### Fine-Tune RegNet
```bash
python regnet/training/finetune.py \
  --pretrained_model pretrain_outputs/best_model.pth \
  --expr_path path/to/BL--ExpressionData.csv \
  --output_dir finetune_outputs/
```

### Visualize Embeddings
```bash
python regnet/visualization/plot_embeddings.py \
  --embedding_path finetune_outputs/embeddings.npy \
  --save_path embedding_plot.png
```

## Dependencies

- Python 3.x
- PyTorch
- DGL or PyTorch Geometric
- pandas, numpy, scikit-learn
- matplotlib, seaborn

Install via:
```bash
pip install -r requirements.txt
```
or create environment:
```bash
conda env create -f environment.yml
```

## Contributing

Contributions welcome! Please follow repository guidelines, open an issue, or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For detailed implementation and parameter tuning, refer to docstrings within each script. For questions or issues, please open an issue on the repository. 