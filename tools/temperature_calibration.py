# regnet/tools/temperature_calibration.py
"""
Post‑hoc temperature scaling for RegNet edge probabilities
----------------------------------------------------------

Fits a scalar T on a *small* label set via negative‑log‑likelihood (BCE),
then outputs calibrated probabilities, per‑edge entropy and calibration
metrics (AUROC, AUPR, ECE, R_shift).

Example
-------
python -m regnet.tools.temperature_calibration \
       --edge_logit_csv outputs/predictions_with_gene_names.csv \
       --label_csv      data/.../Label.csv \
       --tf_file        data/.../TF.csv \
       --target_file    data/.../Target.csv \
       --output_dir     outputs
"""
from __future__ import annotations
import argparse, os, json
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .ece_utils import (entropy_binary,
                        expected_calibration_error,
                        r_shift_index,
                        dump_metrics)

# --------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser("RegNet temperature calibration")
    p.add_argument("--edge_logit_csv", required=True,
                   help="CSV with columns: TF_gene, Target_gene, Prediction (logit or prob)")
    p.add_argument("--label_csv",      required=True,
                   help="CSV with TF, Target, Label columns (indices)")
    p.add_argument("--tf_file",        required=True,
                   help="TF index→name mapping CSV")
    p.add_argument("--target_file",    required=True,
                   help="Target index→name mapping CSV")
    p.add_argument("--max_samples",    type=int, default=50000,
                   help="Random subsample for fitting temperature")
    p.add_argument("--output_dir",     required=True)
    return p.parse_args()

# --------------------------------------------------------------------- #
def map_indices(label_df, tf_map, tgt_map):
    label_df['TF_gene']     = label_df['TF'].map(tf_map)
    label_df['Target_gene'] = label_df['Target'].map(tgt_map)
    label_df = label_df.dropna(subset=['TF_gene', 'Target_gene'])
    return label_df.set_index(['TF_gene', 'Target_gene'])['Label']

# --------------------------------------------------------------------- #
def fit_temperature(logits: torch.Tensor,
                    labels: torch.Tensor,
                    n_iter: int = 500,
                    lr: float = 1e-2) -> float:
    """
    Optimises BCE(logit/T, label) over scalar T > 0.
    """
    T = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=n_iter,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            logits / T.clamp_min(1e-4), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.data)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load edge predictions
    edge_df = pd.read_csv(args.edge_logit_csv)
    if edge_df['Prediction'].max() <= 1:  # probabilities
        logits = torch.logit(torch.tensor(edge_df['Prediction'].values,
                                          dtype=torch.float32))
    else:
        logits = torch.tensor(edge_df['Prediction'].values,
                              dtype=torch.float32)

    # Load ground‑truth labels
    tf_df = pd.read_csv(args.tf_file)
    tgt_df = pd.read_csv(args.target_file)
    tf_map  = dict(tf_df.iloc[:, :2].values)   # index -> TF
    tgt_map = dict(tgt_df.iloc[:, :2].values)  # index -> Gene
    label_series = map_indices(pd.read_csv(args.label_csv),
                               tf_map, tgt_map)
    # align
    idx = edge_df.set_index(['TF_gene', 'Target_gene']).index
    mask = label_series.reindex(idx)
    valid = mask.notna().to_numpy()

    logits = logits[valid]
    labels = torch.tensor(mask[valid].astype(float).values,
                          dtype=torch.float32)

    # Subsample for speed if desired
    if args.max_samples and len(logits) > args.max_samples:
        sel = torch.randperm(len(logits))[:args.max_samples]
        logits, labels = logits[sel], labels[sel]

    # Fit temperature
    T_star = fit_temperature(logits, labels)
    print(f"Optimal temperature T = {T_star:.3f}")

    # Calibrate ALL predictions
    all_logits = torch.logit(torch.tensor(edge_df['Prediction'].values,
                                          dtype=torch.float32)) \
                 if edge_df['Prediction'].max() <= 1 else \
                 torch.tensor(edge_df['Prediction'].values,
                              dtype=torch.float32)

    probs_cal = torch.sigmoid(all_logits / T_star).numpy()
    edge_df['Calibrated'] = probs_cal
    edge_df.to_csv(os.path.join(args.output_dir,
                                "calibrated_test_predictions.csv"),
                   index=False)

    # Metrics (on valid labelled subset)
    p_valid  = probs_cal[valid]
    y_valid  = labels.numpy()
    aupr     = average_precision_score(y_valid, p_valid)
    auroc    = roc_auc_score(y_valid, p_valid)
    ece      = expected_calibration_error(p_valid, y_valid)
    entropy  = entropy_binary(p_valid).mean()
    r_shift  = r_shift_index(ece, float(entropy))

    dump_metrics(os.path.join(args.output_dir, "calibration_metrics.json"),
                 temperature=T_star,
                 AUROC=auroc, AUPR=aupr,
                 ECE=ece, MeanEntropy=float(entropy),
                 R_shift=r_shift)

    np.save(os.path.join(args.output_dir, "edge_entropy.npy"),
            entropy_binary(probs_cal))
    print(f"Saved calibrated scores and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
