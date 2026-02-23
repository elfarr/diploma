from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def validate_input(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"y_true", "p_cal"}

    out = df[["y_true", "p_cal"]].copy()
    if out["y_true"].isna().any() or out["p_cal"].isna().any():
        raise ValueError()

    try:
        y = out["y_true"].astype(int)
    except Exception as exc:  # noqa: BLE001
        raise ValueError() from exc

    uniq = sorted(y.unique().tolist())
    if any(v not in (0, 1) for v in uniq):
        if len(uniq) == 2:
            y = y.map({uniq[0]: 0, uniq[1]: 1}).astype(int)
        else:
            raise ValueError()

    try:
        p = out["p_cal"].astype(float)
    except Exception as exc:  # noqa: BLE001
        raise ValueError() from exc

    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError()

    out["y_true"] = y
    out["p_cal"] = p
    return out


def calc_defined_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    if (tp + fp) > 0:
        precision = float(tp / (tp + fp))
        denom_f1 = 2 * tp + fp + fn
        f1 = float(2 * tp / denom_f1) if denom_f1 > 0 else 0.0
    else:
        precision = float("nan")
        f1 = 0.0

    return {
        "sens": sens,
        "spec": spec,
        "precision": precision,
        "f1": f1,
    }


def build_grid(df: pd.DataFrame) -> pd.DataFrame:
    p = df["p_cal"].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=int)
    n_total = len(df)

    t_low_list = np.arange(0.30, 0.50 + 1e-9, 0.05)
    t_high_list = np.arange(0.50, 0.70 + 1e-9, 0.05)

    rows: List[Dict[str, float | int]] = []
    for t_low in t_low_list:
        for t_high in t_high_list:
            if not (t_low < t_high):
                continue

            defined = (p < t_low) | (p > t_high)
            n_defined = int(defined.sum())
            coverage = float(defined.mean()) if n_total > 0 else float("nan")

            if n_defined == 0:
                sens = float("nan")
                spec = float("nan")
                precision = float("nan")
                f1 = 0.0
            else:
                y_def = y[defined]
                pred_def = np.where(p[defined] > t_high, 1, 0).astype(int)
                m = calc_defined_metrics(y_def, pred_def)
                sens = m["sens"]
                spec = m["spec"]
                precision = m["precision"]
                f1 = m["f1"]

            rows.append(
                {
                    "t_low": float(t_low),
                    "t_high": float(t_high),
                    "coverage": coverage,
                    "sens": sens,
                    "spec": spec,
                    "precision": precision,
                    "f1": f1,
                    "n_defined": n_defined,
                    "n_total": int(n_total),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(by=["f1", "coverage"], ascending=[False, False], na_position="last").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.preds)
    df = validate_input(df, args.preds)
    out_df = build_grid(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Сохранено: {args.out}")


if __name__ == "__main__":
    main()

