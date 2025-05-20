# regnet/tools/ece_utils.py
"""
Utility functions for probabilistic calibration of RegNet edge scores.
----------------------------------------------------------------------
* expected_calibration_error
* entropy_binary
* r_shift_index
"""

from __future__ import annotations
import json, numpy as np


def entropy_binary(p: np.ndarray) -> np.ndarray:
    """Binary entropy h_bin(p) = -p log p - (1-p) log(1-p)."""
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p * np.log(p) + (1-p) * np.log(1-p))


def expected_calibration_error(pred: np.ndarray,
                               label: np.ndarray,
                               n_bins: int = 15) -> float:
    """
    Vectorised ECE over equally‑spaced probability bins.
    """
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask   = (pred >= lo) & (pred < hi)
        if mask.any():
            acc   = label[mask].mean()
            conf  = pred [mask].mean()
            ece  += np.abs(acc - conf) * mask.mean()
    return float(ece)


def r_shift_index(ece: float,
                  var_logit: float,
                  kappa: float = .5) -> float:
    """
    Simple scalar: mis‑calibration + κ · predictive‑variance.
    """
    return float(ece + kappa * var_logit)


def dump_metrics(path: str, **kwargs) -> None:
    with open(path, "w") as fp:
        json.dump(kwargs, fp, indent=2)
