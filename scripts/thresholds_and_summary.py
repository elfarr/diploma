from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PRED_FILE_RE = re.compile(r"^preds_(?P<name>.+)_fold(?P<fold>\d+)\.csv$")
REQUIRED_MODELS = ["svm_rbf", "catboost", "mlp"]
SUMMARY_MODELS = ["svm_rbf", "catboost", "mlp", "ensemble"]


def _to_binary(y: pd.Series) -> pd.Series:
    if y.isna().any():
        raise ValueError()
    try:
        y_int = y.astype(int)
    except Exception as exc:  # noqa: BLE001
        raise ValueError() from exc

    uniq = sorted(y_int.unique().tolist())
    if len(uniq) != 2:
        raise ValueError()
    if uniq != [0, 1]:
        y_int = y_int.map({uniq[0]: 0, uniq[1]: 1}).astype(int)
    return y_int


def _validate_pred_columns(df: pd.DataFrame, file_path: Path) -> None:
    required = {"y_true", "p_cal"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError()


def collect_prediction_files(preds_dir: Path) -> Dict[str, List[Tuple[int, Path]]]:
    if not preds_dir.exists():
        raise FileNotFoundError()

    mapping: Dict[str, List[Tuple[int, Path]]] = {}
    for path in preds_dir.glob("preds_*_fold*.csv"):
        m = PRED_FILE_RE.match(path.name)
        if not m:
            continue
        name = m.group("name")
        fold = int(m.group("fold"))
        mapping.setdefault(name, []).append((fold, path))

    if not mapping:
        raise ValueError()
    return mapping


def resolve_preds_dir(preds_dir: Path) -> Path:
    if any(preds_dir.glob("preds_*_fold*.csv")):
        return preds_dir
    fallback = Path("reports/tables")
    if any(fallback.glob("preds_*_fold*.csv")):
        return fallback
    return preds_dir


def load_pooled_predictions(pred_files: List[Tuple[int, Path]]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for fold, path in sorted(pred_files, key=lambda x: x[0]):
        df = pd.read_csv(path)
        _validate_pred_columns(df, path)
        df = df[["y_true", "p_cal"]].copy()
        df["outer_fold"] = fold
        parts.append(df)

    pooled = pd.concat(parts, axis=0, ignore_index=True)
    if pooled.empty:
        raise ValueError()
    return pooled


def compute_ece(y_true: pd.Series, p_pred: pd.Series, n_bins: int = 10) -> float:
    y = _to_binary(y_true).to_numpy(dtype=float)
    p = p_pred.astype(float).to_numpy()

    if np.isnan(p).any():
        raise ValueError()
    if (p < 0).any() or (p > 1).any():
        raise ValueError()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=False)

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        weight = float(mask.mean())
        ece += abs(acc - conf) * weight
    return float(ece)


def compute_basic_metrics(y_true: pd.Series, p_pred: pd.Series) -> Dict[str, float]:
    y = _to_binary(y_true)
    p = p_pred.astype(float)
    if p.isna().any():
        raise ValueError()
    if (p < 0).any() or (p > 1).any():
        raise ValueError()

    try:
        roc_auc = float(roc_auc_score(y, p))
    except Exception:  # noqa: BLE001
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y, p))
    except Exception:  # noqa: BLE001
        pr_auc = float("nan")
    try:
        brier = float(brier_score_loss(y, p))
    except Exception:  # noqa: BLE001
        brier = float("nan")
    ece = compute_ece(y, p, n_bins=10)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece": ece,
    }


def compute_threshold_row(y_true: pd.Series, p_pred: pd.Series, t_low: float, t_high: float) -> Dict[str, float]:
    y = _to_binary(y_true)
    p = p_pred.astype(float)

    pred_label = pd.Series(np.where(p < t_low, 0, np.where(p > t_high, 1, -1)), index=p.index)
    defined_mask = pred_label != -1
    n_total = int(len(pred_label))
    n_defined = int(defined_mask.sum())
    coverage = float(n_defined / n_total) if n_total > 0 else float("nan")

    sens = float("nan")
    spec = float("nan")
    precision = float("nan")
    f1 = float("nan")

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if n_defined > 0:
        y_d = y[defined_mask].to_numpy()
        p_d = pred_label[defined_mask].astype(int).to_numpy()

        tp = int(((p_d == 1) & (y_d == 1)).sum())
        fp = int(((p_d == 1) & (y_d == 0)).sum())
        tn = int(((p_d == 0) & (y_d == 0)).sum())
        fn = int(((p_d == 0) & (y_d == 1)).sum())

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")

    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "coverage": coverage,
        "sens": sens,
        "spec": spec,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_defined": n_defined,
        "n_total": n_total,
    }


def build_threshold_grid(y_true: pd.Series, p_pred: pd.Series, model_name: str) -> pd.DataFrame:
    t_low_grid = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    t_high_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    rows: List[Dict[str, float]] = []
    for t_low in t_low_grid:
        for t_high in t_high_grid:
            if t_low >= t_high:
                continue
            rows.append(compute_threshold_row(y_true=y_true, p_pred=p_pred, t_low=t_low, t_high=t_high))
    grid_df = pd.DataFrame(rows)
    grid_df.insert(0, "model", model_name)
    return grid_df


def choose_threshold_pair(grid_df: pd.DataFrame) -> Tuple[float, float]:
    if grid_df.empty:
        raise ValueError("Таблица grid_thresholds пуста.")

    filtered = grid_df.copy()
    if "fn" in filtered.columns:
        fn_target = 1
        fn_ok = pd.to_numeric(filtered["fn"], errors="coerce") <= fn_target
        if fn_ok.fillna(False).any():
            filtered = filtered[fn_ok].copy()
        else:
            min_fn = pd.to_numeric(filtered["fn"], errors="coerce").min()
            filtered = filtered[pd.to_numeric(filtered["fn"], errors="coerce") == min_fn].copy()

    coverage_ok = filtered["coverage"] >= 0.50
    if coverage_ok.fillna(False).any():
        filtered = filtered[coverage_ok].copy()
    else:
        max_cov = filtered["coverage"].max()
        filtered = filtered[filtered["coverage"] == max_cov].copy()

    filtered["sens_cmp"] = filtered["sens"].fillna(-1.0)
    filtered["spec_cmp"] = filtered["spec"].fillna(-1.0)
    filtered["f1_cmp"] = filtered["f1"].fillna(-1.0)
    filtered["youden_cmp"] = filtered["sens_cmp"] + filtered["spec_cmp"] - 1.0
    filtered["width"] = filtered["t_high"] - filtered["t_low"]

    if "fn" in filtered.columns and "fp" in filtered.columns:
        filtered["fn_cmp"] = pd.to_numeric(filtered["fn"], errors="coerce").fillna(float("inf"))
        filtered["fp_cmp"] = pd.to_numeric(filtered["fp"], errors="coerce").fillna(float("inf"))
        filtered = filtered.sort_values(
            by=["fn_cmp", "fp_cmp", "coverage", "f1_cmp", "youden_cmp", "sens_cmp", "width", "t_low", "t_high"],
            ascending=[True, True, False, False, False, False, True, True, True],
        )
    else:
        filtered = filtered.sort_values(
            by=["f1_cmp", "youden_cmp", "sens_cmp", "width", "t_low", "t_high"],
            ascending=[False, False, False, True, True, True],
        )
    best = filtered.iloc[0]
    return float(best["t_low"]), float(best["t_high"])


def read_ci_file(path: Path) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    if not path.exists():
        return {"brier": (None, None), "ece": (None, None)}

    df = pd.read_csv(path)
    required = {"metric", "ci_low", "ci_high"}
    if not required.issubset(df.columns):
        raise ValueError()

    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {"brier": (None, None), "ece": (None, None)}
    for metric in ("brier", "ece"):
        row = df[df["metric"] == metric]
        if row.empty:
            continue
        ci_low = row.iloc[0]["ci_low"]
        ci_high = row.iloc[0]["ci_high"]
        out[metric] = (
            None if pd.isna(ci_low) else float(ci_low),
            None if pd.isna(ci_high) else float(ci_high),
        )
    return out


def read_calibration(model_dir: Path, model_name: str) -> str:
    cfg_path = model_dir / f"cfg_{model_name}.json"
    if not cfg_path.exists():
        return ""
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    for key in ("calibration", "calibration_method", "type"):
        value = cfg.get(key)
        if value is not None:
            return str(value)
    return ""


def build_final_summary(
    pooled_by_model: Dict[str, pd.DataFrame],
    tables_dir: Path,
    model_dir: Path,
) -> pd.DataFrame:
    ci_paths = {
        "svm_rbf": tables_dir / "brier_ece_ci_svm_rbf.csv",
        "catboost": tables_dir / "brier_ece_ci_catboost.csv",
        "mlp": tables_dir / "brier_ece_ci_mlp.csv",
        "ensemble": tables_dir / "brier_ece_ci_ensemble.csv",
    }
    ci_data = {model: read_ci_file(path) for model, path in ci_paths.items()}

    rows: List[Dict[str, object]] = []
    for model_name in SUMMARY_MODELS:
        if model_name not in pooled_by_model:
            raise ValueError()

        pooled = pooled_by_model[model_name]
        metrics = compute_basic_metrics(pooled["y_true"], pooled["p_cal"])
        brier_ci_low, brier_ci_high = ci_data[model_name]["brier"]
        ece_ci_low, ece_ci_high = ci_data[model_name]["ece"]


        calibration = read_calibration(model_dir, model_name) if model_name in REQUIRED_MODELS else ""

        rows.append(
            {
                "model": model_name,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "brier_ci_low": brier_ci_low if brier_ci_low is not None else "",
                "brier_ci_high": brier_ci_high if brier_ci_high is not None else "",
                "ece_ci_low": ece_ci_low if ece_ci_low is not None else "",
                "ece_ci_high": ece_ci_high if ece_ci_high is not None else "",
                "calibration": calibration,
            }
        )
    return pd.DataFrame(rows)


def choose_best_single_model_for_thresholds(pooled_by_model: Dict[str, pd.DataFrame]) -> str:
    rows: List[Tuple[str, int, int, float, float]] = []
    for model_name in REQUIRED_MODELS:
        if model_name not in pooled_by_model:
            raise ValueError()
        pooled = pooled_by_model[model_name]
        y = _to_binary(pooled["y_true"]).to_numpy(dtype=int)
        p = pooled["p_cal"].astype(float).to_numpy()
        pred = (p >= 0.5).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        m = compute_basic_metrics(pooled["y_true"], pooled["p_cal"])
        rows.append((model_name, fn, fp, m["pr_auc"], m["brier"]))

    rows_sorted = sorted(rows, key=lambda x: (x[1], x[2], -x[3], x[4], x[0]))
    return rows_sorted[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/v2.0.0"))
    parser.add_argument("--preds-dir", type=Path, default=Path("reports/preds"))
    parser.add_argument("--tables-dir", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    preds_dir: Path = resolve_preds_dir(args.preds_dir)
    tables_dir: Path = args.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    pred_file_map = collect_prediction_files(preds_dir)

    for model_name in REQUIRED_MODELS:
        if model_name not in pred_file_map:
            raise ValueError()
    if "ensemble" not in pred_file_map:
        raise ValueError()

    pooled_by_model: Dict[str, pd.DataFrame] = {}
    for model_name, files in pred_file_map.items():
        pooled_by_model[model_name] = load_pooled_predictions(files)

    grid_by_model: Dict[str, pd.DataFrame] = {}
    for grid_model_name in SUMMARY_MODELS:
        pooled = pooled_by_model[grid_model_name]
        grid_df_model = build_threshold_grid(
            y_true=pooled["y_true"],
            p_pred=pooled["p_cal"],
            model_name=grid_model_name,
        )
        grid_by_model[grid_model_name] = grid_df_model
        grid_model_path = tables_dir / f"grid_thresholds_{grid_model_name}.csv"
        grid_df_model.to_csv(grid_model_path, index=False, encoding="utf-8")

    threshold_model = choose_best_single_model_for_thresholds(pooled_by_model)
    selected_grid_df = grid_by_model[threshold_model]

    grid_path = tables_dir / "grid_thresholds.csv"
    selected_grid_df.to_csv(grid_path, index=False, encoding="utf-8")

    best_t_low, best_t_high = choose_threshold_pair(selected_grid_df)
    thresholds_path = model_dir / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "t_low": best_t_low,
                "t_high": best_t_high,
                "source_model": threshold_model,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_df = build_final_summary(pooled_by_model=pooled_by_model, tables_dir=tables_dir, model_dir=model_dir)
    summary_path = tables_dir / "final_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    for grid_model_name in SUMMARY_MODELS:
        print(f"Сохранено: {tables_dir / f'grid_thresholds_{grid_model_name}.csv'}")
    print(f"Сохранено: {grid_path}")
    print(f"Сохранено: {thresholds_path}")
    print(f"Сохранено: {summary_path}")


if __name__ == "__main__":
    main()
