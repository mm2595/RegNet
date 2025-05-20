# Benchmark Dataset for Gene Regulatory Network Inference

This directory contains benchmark datasets for gene regulatory network (GRN) inference.

## Folder Structure

```
data/
├── Benchmark Dataset/
│   ├── Lofgof Dataset/
│   │   ├── <cell_typeX>/
│   │   │   ├── TFs+500/
│   │   │   │   ├── BL--ExpressionData.csv
│   │   │   │   ├── Full_set.csv
│   │   │   │   ├── TF.csv
│   │   │   │   └── Target.csv
│   │   │   └── TFs+1000/
│   │   │       ├── BL--ExpressionData.csv
│   │   │       ├── Full_set.csv
│   │   │       ├── TF.csv
│   │   │       └── Target.csv
│   │   └── <cell_typeY>/
│   ├── Non-Specific Dataset/
│   │   └── <cell_type>/
│   │       ├── TFs+500/
│   │       └── TFs+1000/
│   ├── Specific Dataset/
│   │   └── <cell_type>/
│   │       ├── TFs+500/
│   │       └── TFs+1000/
│   ├── STRING Dataset/
│   │   └── <cell_type>/
│   │       ├── TFs+500/
│   │       └── TFs+1000/
│   └── Sample Dataset/
│       └── <cell_type>/
└── sample_data/
```

- Replace `<cell_typeX>`, `<cell_typeY>`, and `<cell_type>` with actual cell types under each folder

## Data Organization

- **Network Types**:
  - Lofgof Dataset
  - Non-Specific Dataset
  - Specific Dataset
  - STRING Dataset
  - Sample Dataset (example/training/validation/test splits)

- **Cell Types**: Varies by network type (e.g., `hHEP`, `mESC`). See subfolders for details.

- **Network Sizes**:
  - `TFs+500` – network containing TFs plus 500 target genes
  - `TFs+1000` – network containing TFs plus 1000 target genes

## File Descriptions

1. **BL--ExpressionData.csv**  
   - Gene expression matrix.  
   - Rows: gene names  
   - Columns: sample IDs

2. **Full_set.csv**  
   - Specifies regulatory relationships (edges) between genes.  
   - Columns:  
     - `TF`: regulator gene index (integer; see `TF.csv` for mapping)  
     - `Target`: target gene index (integer; see `Target.csv` for mapping)  
     - `Label`: 1 = regulatory edge exists; 0 = no edge  
   - Genes are specified by index rather than name.

3. **TF.csv**  
   - Maps transcription factor (TF) gene names to the integer indices used in `Full_set.csv`.  
   - Columns:  
     - `TF`: gene name  
     - `index`: gene index (integer)

4. **Target.csv**  
   - Maps target gene names to the integer indices used in `Full_set.csv`.  
   - Columns:  
     - `Gene`: gene name  
     - `index`: gene index (integer)

## Usage Example

```python
import pandas as pd

# Example: load data for the mESC network with TFs+500 (Lofgof Dataset)
base_path = "data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500"

expr = pd.read_csv(f"{base_path}/BL--ExpressionData.csv", index_col=0)
edges = pd.read_csv(f"{base_path}/Full_set.csv")
tf_map = pd.read_csv(f"{base_path}/TF.csv")
target_map = pd.read_csv(f"{base_path}/Target.csv")
```

## Notes

- Gene names in the expression data should correspond to the names in `TF.csv` and `Target.csv`.  
- Directory names contain spaces; ensure your code or shell handles spaces properly or adjust paths accordingly.

## Contact

For questions or issues, please reach out to the data maintainers or open an issue in this repository.

---

Feel free to reach out if you have any questions or need further clarification.
