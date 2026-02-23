from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def validate_input(df: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    required = {"y_true", "p_cal"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError()

    data = df[["y_true", "p_cal"]].copy()
    if data.isna().any().any():
        raise ValueError()

    try:
        y = data["y_true"].astype(int).to_numpy()
    except Exception as exc:  # noqa: BLE001
        raise ValueError() from exc

    uniq = sorted(np.unique(y).tolist())
    if len(uniq) > 2:
        raise ValueError()
    if any(v not in (0, 1) for v in uniq):
        if len(uniq) != 2:
            raise ValueError()
        mapping = {uniq[0]: 0, uniq[1]: 1}
        y = np.array([mapping[int(v)] for v in y], dtype=int)

    try:
        p = data["p_cal"].astype(float).to_numpy()
    except Exception as exc:  # noqa: BLE001
        raise ValueError() from exc

    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError()

    if len(y) == 0:
        raise ValueError()

    return y, p


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return float(np.mean((p - y) ** 2))


def ece_fixed_bins(y_true: Iterable[int], p_pred: Iterable[float], n_bins: int = 10) -> float:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(p_pred), dtype=float)
    if len(y) != len(p):
        raise ValueError()
    if len(y) == 0:
        raise ValueError()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=False)

    ece = 0.0
    n = len(y)
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        weight = float(mask.sum() / n)
        ece += abs(acc - conf) * weight
    return float(ece)


def bootstrap_metric_samples(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_boot <= 0:
        raise ValueError()

    rng = np.random.default_rng(seed)
    n = len(y_true)
    brier_vals = np.empty(n_boot, dtype=float)
    ece_vals = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = p_pred[idx]
        brier_vals[i] = brier_score(y_b, p_b)
        ece_vals[i] = ece_fixed_bins(y_b, p_b, n_bins=10)

    return brier_vals, ece_vals


def summarize_bootstrap(metric_name: str, values: np.ndarray, n_boot: int) -> dict:
    return {
        "metric": metric_name,
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "n_boot": int(n_boot),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.preds.exists():
        raise FileNotFoundError()

    df = pd.read_csv(args.preds)
    y_true, p_pred = validate_input(df, args.preds)

    brier_boot, ece_boot = bootstrap_metric_samples(
        y_true=y_true,
        p_pred=p_pred,
        n_boot=args.n,
        seed=args.seed,
    )

    out_df = pd.DataFrame(
        [
            summarize_bootstrap("brier", brier_boot, args.n),
            summarize_bootstrap("ece", ece_boot, args.n),
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Сохранено: {args.out}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
