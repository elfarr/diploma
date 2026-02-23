from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


MODEL_NAMES = ("svm_rbf", "catboost", "mlp")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError() from exc


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def select_model_for_bin(
    bin_id: int,
    competence: Dict[str, Any],
    fallback_model: str = "catboost",
) -> str:
    ece_map = competence.get("ece", {}) if isinstance(competence, dict) else {}
    brier_map = competence.get("brier", {}) if isinstance(competence, dict) else {}
    candidates: list[Tuple[str, float, float]] = []

    for model_name in MODEL_NAMES:
        ece_arr = ece_map.get(model_name)
        brier_arr = brier_map.get(model_name)
        if not isinstance(ece_arr, list) or bin_id >= len(ece_arr):
            continue
        if not isinstance(brier_arr, list) or bin_id >= len(brier_arr):
            continue

        ece_val = safe_float(ece_arr[bin_id])
        if ece_val is None:
            continue
        brier_val = safe_float(brier_arr[bin_id])
        if brier_val is None:
            brier_val = float("inf")
        candidates.append((model_name, ece_val, brier_val))

    if not candidates:
        return fallback_model
    return min(candidates, key=lambda x: (x[1], x[2]))[0]


def validate_probabilities(series: pd.Series, file_path: Path, col: str = "p_cal") -> None:
    if series.isna().any():
        raise ValueError()
    if ((series < 0.0) | (series > 1.0)).any():
        raise ValueError()


def load_fold_df(preds_dir: Path, model_name: str, fold: int) -> pd.DataFrame:
    path = preds_dir / f"preds_{model_name}_fold{fold}.csv"
    if not path.exists():
        raise FileNotFoundError()
    df = pd.read_csv(path)
    required_cols = {"y_true", "p_cal"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError()
    return df


def resolve_preds_dir(preds_dir: Path) -> Path:
    probe = preds_dir / "preds_svm_rbf_fold1.csv"
    if probe.exists():
        return preds_dir

    fallback = Path("reports/tables")
    fallback_probe = fallback / "preds_svm_rbf_fold1.csv"
    if fallback_probe.exists():
        return fallback

    return preds_dir


def validate_targets_equal(df_svm: pd.DataFrame, df_cat: pd.DataFrame, df_mlp: pd.DataFrame, fold: int) -> None:
    n_svm, n_cat, n_mlp = len(df_svm), len(df_cat), len(df_mlp)
    if not (n_svm == n_cat == n_mlp):
        raise ValueError()

    y_svm = df_svm["y_true"].reset_index(drop=True)
    y_cat = df_cat["y_true"].reset_index(drop=True)
    y_mlp = df_mlp["y_true"].reset_index(drop=True)

    if not y_svm.equals(y_cat) or not y_svm.equals(y_mlp):
        raise ValueError()


def export_fold(
    preds_dir: Path,
    fold: int,
    competence: Dict[str, Any],
) -> Path:
    df_svm = load_fold_df(preds_dir, "svm_rbf", fold)
    df_cat = load_fold_df(preds_dir, "catboost", fold)
    df_mlp = load_fold_df(preds_dir, "mlp", fold)

    validate_targets_equal(df_svm, df_cat, df_mlp, fold)
    validate_probabilities(df_svm["p_cal"], preds_dir / f"preds_svm_rbf_fold{fold}.csv")
    validate_probabilities(df_cat["p_cal"], preds_dir / f"preds_catboost_fold{fold}.csv")
    validate_probabilities(df_mlp["p_cal"], preds_dir / f"preds_mlp_fold{fold}.csv")

    y_true = df_svm["y_true"].reset_index(drop=True)
    p_svm = df_svm["p_cal"].astype(float).reset_index(drop=True)
    p_cat = df_cat["p_cal"].astype(float).reset_index(drop=True)
    p_mlp = df_mlp["p_cal"].astype(float).reset_index(drop=True)

    p_avg = (p_svm + p_cat + p_mlp) / 3.0
    bin_id = np.floor(p_avg * 10.0).astype(int).clip(0, 9)

    model_used: list[str] = []
    p_final: list[float] = []

    for i in range(len(y_true)):
        b = int(bin_id.iloc[i])
        winner = select_model_for_bin(bin_id=b, competence=competence, fallback_model="catboost")
        model_used.append(winner)
        if winner == "svm_rbf":
            p_final.append(float(p_svm.iloc[i]))
        elif winner == "mlp":
            p_final.append(float(p_mlp.iloc[i]))
        else:
            p_final.append(float(p_cat.iloc[i]))

    out_df = pd.DataFrame(
        {
            "y_true": y_true,
            "p_cal": p_final,
            "bin_id": bin_id.astype(int),
            "model_used": model_used,
            "p_avg": p_avg.astype(float),
        }
    )
    validate_probabilities(out_df["p_cal"], preds_dir / f"preds_ensemble_fold{fold}.csv")

    out_path = preds_dir / f"preds_ensemble_fold{fold}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=Path, default=Path("reports/preds"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/v2.0.0"))
    args = parser.parse_args()

    preds_dir: Path = resolve_preds_dir(args.preds_dir)
    model_dir: Path = args.model_dir
    preds_dir.mkdir(parents=True, exist_ok=True)

    competence_path = model_dir / "competence_by_risk_bin.json"
    competence = load_json(competence_path)

    thresholds_path = model_dir / "thresholds.json"
    if thresholds_path.exists():
        thresholds = load_json(thresholds_path)
        t_low = float(thresholds.get("t_low", 0.45))
        t_high = float(thresholds.get("t_high", 0.65))
    else:
        t_low = 0.45
        t_high = 0.65


    for fold in range(1, 6):
        out_path = export_fold(preds_dir=preds_dir, fold=fold, competence=competence)
        print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
