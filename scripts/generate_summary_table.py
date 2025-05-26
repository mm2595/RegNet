#!/usr/bin/env python
"""Generate summary table of RegNet pretraining results for README.
Scans every dataset directory under pretrain_outputs/ and extracts:
 - Best train AUPR & AUROC (max across epochs)
 - Best validation AUPR (if available)
 - Calibrated AUPR (if calibrated_predictions.csv exists)
"""
import os, glob, pandas as pd, numpy as np
from pathlib import Path

def extract_metrics(pretrain_dir: Path):
    metrics_csv = pretrain_dir / 'training_metrics.csv'
    if not metrics_csv.is_file():
        return None
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return None
    # Best training metrics – max across epochs
    idx_train_best = df['AUPR'].idxmax()
    best_train_aupr  = df.at[idx_train_best, 'AUPR']
    best_train_auroc = df.at[idx_train_best, 'AUROC'] if 'AUROC' in df.columns else np.nan

    # Best validation metrics – take column-wise maxima, ignoring NaNs
    best_val_aupr  = df['val_AUPR'].max(skipna=True) if 'val_AUPR' in df.columns else np.nan
    best_val_auroc = df['val_AUROC'].max(skipna=True) if 'val_AUROC' in df.columns else np.nan

    # --------------------------------------------------------------
    # Test metrics (evaluation_predictions.csv)
    test_aupr  = np.nan; test_auroc = np.nan
    eval_csv = pretrain_dir.parent / 'evaluation' / 'evaluation_predictions.csv'
    if eval_csv.is_file():
        try:
            df_eval = pd.read_csv(eval_csv)
            from sklearn.metrics import average_precision_score, roc_auc_score
            test_aupr  = average_precision_score(df_eval['Label'], df_eval['Prediction'])
            test_auroc = roc_auc_score      (df_eval['Label'], df_eval['Prediction'])
        except Exception:
            pass

    # --------------------------------------------------------------
    # Calibrated metrics – both AUPR and AUROC if available
    calib_auroc = np.nan
    calib_paths = [pretrain_dir.parent / 'temperature_calibration' / 'calibrated_test_predictions.csv',
                   pretrain_dir / 'calibrated_predictions.csv']
    calib_aupr = np.nan
    for cp in calib_paths:
        if cp.is_file():
            try:
                df_cal = pd.read_csv(cp)
                if {'Label','Calibrated'}.issubset(df_cal.columns):
                    from sklearn.metrics import average_precision_score, roc_auc_score
                    calib_aupr  = average_precision_score(df_cal['Label'], df_cal['Calibrated'])
                    calib_auroc = roc_auc_score      (df_cal['Label'], df_cal['Calibrated'])
                    break
            except Exception:
                pass

    return (best_train_aupr, best_train_auroc,
            test_aupr, test_auroc,
            calib_aupr, calib_auroc)

def main():
    root = Path('pretrain_outputs')
    rows = []
    for dataset_dir in sorted(root.glob('*')):
        pretrain_dir = dataset_dir / 'pretrain'
        m = extract_metrics(pretrain_dir)
        if m is None:
            continue
        rows.append((dataset_dir.name,) + m)
    if not rows:
        print("No metrics found.")
        return
    # Markdown table
    header = ['Dataset','Train AUPR','Train AUROC','Test AUPR','Test AUROC','Calibrated AUPR','Calibrated AUROC']
    lines = ['| ' + ' | '.join(header) + ' |',
             '| ' + ' | '.join(['---']*len(header)) + ' |']
    for r in rows:
        vals = []
        for v in r:
            if isinstance(v,float) and not np.isnan(v):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    print('\n'.join(lines))

if __name__ == '__main__':
    main() 