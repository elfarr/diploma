from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"t_low", "t_high", "coverage", "sens", "f1"}


def load_grid(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError()

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError() from exc

    if df.empty:
        raise ValueError()
    return df.copy()


def pick_best_thresholds(df: pd.DataFrame, min_coverage: float) -> pd.Series:
    if not (0.0 <= float(min_coverage) <= 1.0):
        raise ValueError()

    work = df.copy()
    for col in ["t_low", "t_high", "coverage", "sens", "f1"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if work[["t_low", "t_high"]].isna().any().any():
        raise ValueError()

    work = work[work["t_low"] < work["t_high"]].copy()
    if work.empty:
        raise ValueError()

    filtered = work[work["coverage"] >= float(min_coverage)].copy()
    if filtered.empty:
        raise ValueError()

    filtered["f1_cmp"] = filtered["f1"].fillna(-1.0)
    filtered["sens_cmp"] = filtered["sens"].fillna(-1.0)
    filtered["coverage_cmp"] = filtered["coverage"].fillna(-1.0)

    filtered = filtered.sort_values(
        by=["f1_cmp", "sens_cmp", "coverage_cmp"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return filtered.iloc[0]


def save_thresholds(path: Path, row: pd.Series) -> None:
    payload = {
        "t_low": float(row["t_low"]),
        "t_high": float(row["t_high"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.90,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    grid_df = load_grid(args.grid)
    best = pick_best_thresholds(grid_df, min_coverage=args.min_coverage)
    save_thresholds(args.out, best)
    cols_to_print = ["t_low", "t_high", "coverage", "sens", "spec", "precision", "f1", "n_defined", "n_total"]
    available_cols = [c for c in cols_to_print if c in best.index]
    print(best[available_cols].to_string())
    print(f"Сохранено: {args.out}")


if __name__ == "__main__":
    main()
