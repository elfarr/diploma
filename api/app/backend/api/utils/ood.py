from __future__ import annotations

import numpy as np


def mahalanobis(x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    inv_cov = np.asarray(inv_cov, dtype=np.float64)

    if x.shape != mu.shape:
        raise ValueError()
    d = x.shape[0]
    if inv_cov.shape != (d, d):
        raise ValueError()

    diff = x - mu
    return float(np.sqrt(diff @ inv_cov @ diff))


def is_ood(score: float, threshold: float) -> bool:
    return bool(score > threshold)
