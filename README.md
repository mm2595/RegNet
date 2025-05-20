# RegNet: Graph-Based Gene Regulatory Network Inference

RegNet is a modular graph neural network framework for inferring gene regulatory networks (GRNs) by integrating gene expression data and known regulatory priors. It leverages a hybrid architecture combining GraphSAGE, self-attention, and a variational autoencoder (VAE) to learn robust node embeddings and predict regulatory interactions.

## Key Features

- **GraphSAGE encoder**: aggregates local neighborhood information via MLP-based aggregation.
- **Self-Attention module**: computes global attention coefficients over all genes and fuses them via a gating mechanism.
- **Variational Autoencoder (VAE)**: imposes a probabilistic latent structure with Kullback–Leibler divergence regularization and includes reconstruction capability as an auxiliary task.
- **Edge Classifier**: MLP that scores pairwise combinations of latent means \(\mu_i \oplus \mu_j\) to predict regulatory links.
- **Edge Uncertainty**: Calculation of edge variance to quantify prediction uncertainty.
- **Temperature Calibration**: Scaling of logits to produce better-calibrated probabilities.
- **Complete Pipeline**:
  1. **Data Splitting**: Partitions data into train/test sets while preserving network properties.
  2. **Pretraining (supervised)** on training data with ground-truth edges using combined loss functions.
  3. **Evaluation** on test data using metrics like AUROC, AUPR, and F1 scores.
  4. **Temperature Calibration** to improve probability estimates.
  5. **Fine-tuning (self-supervised)** on unlabeled datasets via multiple self-supervised objectives.
- **Visualization utilities** for embeddings, attention maps, and latent space analyses.

## Installation

```bash
# Clone repository
git clone <repo_url> && cd RegNet

# (Optional) Create a conda environment
conda env create -f environment.yml
conda activate regnet

# Or install via pip
pip install torch torchvision torch-scatter torch-sparse torch-geometric
pip install -r requirements.txt
```

## Repository Structure

```
.
├── README.md                 # Project overview and usage (this file)
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment specification
├── regnet/                   # Core Python package
│   ├── preprocessing/        # Data loading & preprocessing
│   │   └── train_test_split.py  # Script to split data into train/test
│   ├── models/               # GraphSAGE, Attention, VAE, and RegNet
│   ├── training/             # Training and evaluation scripts
│   │   ├── pretrain.py       # Pretraining on labeled data
│   │   ├── evaluation.py     # Evaluation on test data
│   │   ├── finetune.py       # Fine-tuning with self-supervised methods
│   │   └── loss_functions.py # Loss function implementations
│   └── visualization/        # Plotting utilities
├── scripts/                  # Helper shell scripts
│   └── run_pretrain.sh       # Complete pipeline execution script
├── tools/                    # Auxiliary tools
├── data/                     # Example datasets (expression matrices, mappings)
├── pretrain_outputs/         # Outputs from pretraining runs
├── finetune_outputs/         # Outputs from fine-tuning runs
└── tests/                    # Unit tests
```

## Model Architecture

1. **Input**: gene expression matrix \(X \in \mathbb{R}^{N\times F}\), each row is a gene with an \(F\)-dimensional expression vector.
2. **GraphSAGE**: MLP-based neighbor aggregation
   
   \[ h^{(k+1)}_i = \mathrm{ReLU}(\mathrm{MLP}([h^{(k)}_i || \mathrm{mean}_{j \in \mathcal{N}(i)} h^{(k)}_j])) \]

3. **Self-Attention Gate Fusion**:
   - Compute \(Q,K,V = W_Q h, W_K h, W_V h\)
   - Attention scores \(A = \mathrm{softmax}(QK^T / \sqrt{d_k})\)
   - Row entropy regularizer: \(-\sum_j A_{ij} \log A_{ij}\)
   - Fuse via gate \(g = \sigma(W_g[h || AV])\), output \(h' = W_o(g \odot AV + (1-g) \odot h)\)

4. **VAE Encoder/Decoder**:
   - Encoder: \(\mu = W_{\mu} h'\), \(\log\sigma^2 = W_{\log\sigma} h'\)
   - Reparameterization: \(z = \mu + \sigma \odot \epsilon\)
   - Decoder: \(h_{recon} = \mathrm{MLP}(z)\)
   - KL loss: \(\mathcal{L}_{\mathrm{KLD}} = -\tfrac12 \mathbb{E}[1 + \log\sigma^2 - \mu^2 - \sigma^2]\)
   - Reconstruction loss: \(\mathcal{L}_{\mathrm{recon}} = \mathrm{MSE}(h, h_{recon})\)

5. **Edge Classifier**:
   - Concatenate latent means: \([\mu_i || \mu_j]\)
   - MLP → Sigmoid to predict \(p_{ij} = P(\text{edge}_{ij}=1)\)

6. **Edge Uncertainty**:
   - Compute variance for each edge prediction based on latent distributions of nodes

## Complete Pipeline

The RegNet pipeline implements a comprehensive workflow for GRN inference:

### 1. Train-Test Split

Before model training, data is partitioned into train and test sets while preserving network properties:

```bash
python regnet/preprocessing/train_test_split.py \
  --label_file data/labels.csv \
  --tf_file data/TFs.csv \
  --target_file data/targets.csv \
  --output_dir output/split \
  --test_size 0.2
```

- Splits interaction data preserving hub TF distribution
- Handles multiple dataset formats (standard and split formats)
- Creates balanced train/test datasets for robust evaluation

### 2. Pretraining (Supervised)

The model is pretrained on the training portion of ground-truth interactions:

```bash
python regnet/training/pretrain.py \
  --expression_data data/expression.csv \
  --label_data output/split/train_labels.csv \
  --split_data \
  --batch_size 1024 \
  --epochs 50 \
  --lr 1e-3 \
  --hidden_dim 128 \
  --latent_dim 64 \
  --num_layers 2 \
  --beta_kl 1.0 \
  --recon_weight 0.1 \
  --device cuda \
  --output_dir pretrain_outputs/
```

- **Loss**: \(\mathcal{L} = \mathrm{BCE}(p_{ij}, y_{ij}) + \beta_{KL}\,\mathcal{L}_{\mathrm{KLD}} + w_{recon}\,\mathcal{L}_{\mathrm{recon}}\)
- **Outputs**: Model weights (`regnet_pretrained.pth`), training metrics, embeddings, predictions

### 3. Evaluation

The pretrained model is evaluated on the held-out test set:

```bash
python regnet/training/evaluation.py \
  --model_path pretrain_outputs/regnet_pretrained.pth \
  --expression_data data/expression.csv \
  --label_data output/split/test_labels.csv \
  --split_data \
  --hidden_dim 128 \
  --latent_dim 64 \
  --num_layers 2 \
  --batch_size 1024 \
  --output_dir evaluation_outputs/ \
  --device cuda
```

- **Metrics**: AUROC, AUPR, Precision, Recall, F1, Accuracy
- **Outputs**: Evaluation metrics, precision-recall curves, predictions with confidence scores

### 4. Temperature Calibration

Logits are scaled by an optimal temperature parameter to improve probability calibration:

```bash
# Built into the pipeline to optimize and apply temperature scaling
# Typically improves the reliability of predicted probabilities
```

- Finds optimal temperature value to minimize negative log-likelihood
- Rescales logits: \(p_{ij} = \sigma(z_{ij} / T)\) where \(T\) is the temperature
- Produces better-calibrated probability estimates

### 5. Fine-Tuning (Self-Supervised)

For datasets without ground-truth labels:

```bash
python regnet/training/finetune.py \
  --expression_data data/new_expression.csv \
  --TF_file data/TFs.csv \
  --target_file data/targets.csv \
  --pretrained_model pretrain_outputs/regnet_pretrained.pth \
  --batch_size 1024 \
  --epochs 100 \
  --patience 5 \
  --lr 2e-4 \
  --output_dir finetune_outputs/
```

- **Combined Loss Functions**:
  - Masked Reconstruction, Contrastive Loss, VAE KL
  - Relational & Prototype Alignment, Discrepancy/EWC
  - Attention Entropy regularization (optional)

## Automated Pipeline Execution

For convenience, a complete pipeline script is provided:

```bash
bash scripts/run_pretrain.sh
```

This script automates the entire workflow:
1. Splits data into train/test sets
2. Pretrains the model on training data
3. Evaluates on test data
4. Performs temperature calibration
5. Conducts final evaluation with calibrated probabilities

## Results on Benchmark Datasets

When evaluated on the Lofgof mESC TFs+500 dataset, the model achieves:
- AUROC: ~0.84
- AUPR: ~0.81
- After temperature calibration (~1.2), calibrated AUROC: ~0.84, AUPR: ~0.81

## Inference

Use the `RegNet` API directly:

```python
from regnet.models.regnet import RegNet
import torch

# Load model
model = RegNet(input_dim=F, hidden_dim=H, latent_dim=D, num_layers=K)
state = torch.load("pretrain_outputs/regnet_pretrained.pth")
model.load_state_dict(state)
model.eval()

# Prepare data
x = torch.tensor(expr_df.values)            # (N,F)
edge_index = torch.tensor([...], dtype=torch.long)  # (2,E)

# Get predicted probabilities
node_pairs = edge_index.t()                 # (E,2)
preds, att_w, row_ent, mu, logvar = model(x, edge_index, node_pairs)

# Extract latent means
mu = model.latent_mu(x, edge_index)

# Reconstruct hidden representation (auxiliary task)
h_recon, z, mu, logvar, att_w, row_ent = model.reconstruct(x, edge_index)

# Get prediction uncertainty
uncertainty = model.edge_variance(mu, logvar, node_pairs)
```

## Dependencies

- Python 3.8+
- PyTorch
- PyTorch Geometric
- torch-scatter, torch-sparse (needed by PyG)
- pandas, numpy, scikit-learn
- matplotlib, seaborn (for visualization)

Install with:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. 