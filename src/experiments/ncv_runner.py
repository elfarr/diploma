from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TypedDict

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.calibration import build_calibrator
from src.pipeline.models import build_catboost, build_mlp, build_svm
from src.pipeline.preprocess import build_preprocess

LOGGER = logging.getLogger("ncv_runner")


class MetricDict(TypedDict):
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float


@dataclass
class FoldSelection:
    best_params: Dict[str, Any]
    best_calibrator: str
    best_inner_brier: float


@dataclass
class OuterFoldResult:
    outer_fold: int
    best_params: str
    best_calibrator: str
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float


@dataclass
class CalibratedModelWrapper:
    model: Any
    calibrator: Any
    calibrator_type: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_score = _extract_raw_score(self.model, X)
        if self.calibrator_type == "platt":
            return self.calibrator.predict_proba(raw_score.reshape(-1, 1))[:, 1]
        if self.calibrator_type == "isotonic":
            return np.clip(self.calibrator.predict(raw_score), 0.0, 1.0)
        raise ValueError()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    sep = ";" if ";" in first_line else ","
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep=sep, encoding_errors="replace")


def infer_categorical_features(df: pd.DataFrame, features: Sequence[str]) -> List[str]:
    categorical: List[str] = []
    for col in features:
        if str(df[col].dtype) in {"object", "category", "bool"}:
            categorical.append(col)
    return categorical


def compute_ece(y_true: Sequence[int], p_cal: Sequence[float], n_bins: int = 10) -> float:
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(p_cal, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p_arr, bin_edges[1:-1], right=False)

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        acc = float(y_arr[mask].mean())
        conf = float(p_arr[mask].mean())
        weight = float(mask.mean())
        ece += abs(acc - conf) * weight
    return float(ece)


def assign_decile_bins_with_fallback(p_cal: Iterable[float]) -> tuple[pd.Series, str]:
    p_series = pd.Series(p_cal).astype(float)
    if p_series.isna().any():
        raise ValueError()
    if (p_series < 0).any() or (p_series > 1).any():
        raise ValueError()

    q_bins = pd.qcut(p_series, q=10, labels=False, duplicates="drop")
    n_unique_bins = int(pd.Series(q_bins).nunique(dropna=True))
    if n_unique_bins == 10:
        return pd.Series(q_bins, index=p_series.index, dtype=int), "qcut"

    fixed_bins = np.floor(p_series.to_numpy() * 10).astype(int)
    fixed_bins = np.clip(fixed_bins, 0, 9)
    return pd.Series(fixed_bins, index=p_series.index, dtype=int), "fixed_0.1"


def compute_bin_metrics_table(
    y_true: Iterable[Any],
    p_cal: Iterable[float],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    y_series = pd.Series(y_true)
    p_series = pd.Series(p_cal).astype(float)
    if len(y_series) != len(p_series):
        raise ValueError()
    if len(y_series) == 0:
        raise ValueError()

    y_bin = to_binary_target(y_series)
    bins, binning_mode = assign_decile_bins_with_fallback(p_series)

    tmp = pd.DataFrame({"y_true": y_bin, "p_cal": p_series, "bin": bins})
    n_total = len(tmp)
    rows: List[Dict[str, Any]] = []
    weighted_ece = 0.0

    for bin_id in sorted(tmp["bin"].unique().tolist()):
        part = tmp[tmp["bin"] == bin_id]
        n_bin = len(part)
        mean_y = float(part["y_true"].mean())
        mean_p = float(part["p_cal"].mean())
        brier_bin = float(((part["p_cal"] - part["y_true"]) ** 2).mean())
        ece_bin = float(abs(mean_y - mean_p))
        weighted_ece += ece_bin * (n_bin / n_total)
        rows.append(
            {
                "bin": int(bin_id),
                "n_bin": int(n_bin),
                "mean_y_true": mean_y,
                "mean_p_cal": mean_p,
                "brier_bin": brier_bin,
                "ece_bin": ece_bin,
            }
        )

    summary = {
        "n_total": int(n_total),
        "n_bins_actual": int(tmp["bin"].nunique()),
        "binning_mode": binning_mode,
        "ece_weighted": float(weighted_ece),
    }
    return pd.DataFrame(rows), summary


def compute_metrics(y_true: Iterable[Any], p_pred: Iterable[float]) -> MetricDict:
    y_series = pd.Series(y_true)
    p_series = pd.Series(p_pred)

    if len(y_series) != len(p_series):
        raise ValueError()
    if len(y_series) == 0:
        raise ValueError()
    if y_series.isna().any() or p_series.isna().any():
        raise ValueError()

    try:
        y_bin = y_series.astype(int)
    except Exception as exc:
        raise ValueError() from exc

    uniq = sorted(y_bin.unique().tolist())
    if len(uniq) != 2:
        raise ValueError()
    if uniq != [0, 1]:
        y_bin = y_bin.map({uniq[0]: 0, uniq[1]: 1}).astype(int)

    try:
        p_float = p_series.astype(float)
    except Exception as exc:
        raise ValueError() from exc

    if (p_float < 0).any() or (p_float > 1).any():
        raise ValueError()

    try:
        return {
            "roc_auc": float(roc_auc_score(y_bin, p_float)),
            "pr_auc": float(average_precision_score(y_bin, p_float)),
            "brier": float(brier_score_loss(y_bin, p_float)),
            "ece": float(compute_ece(y_bin, p_float, n_bins=10)),
        }
    except Exception as exc:
        raise ValueError() from exc


def aggregate_metric_mean(items: List[MetricDict]) -> Dict[str, float]:
    df = pd.DataFrame(items)
    return {
        "roc_auc_mean": float(df["roc_auc"].mean()),
        "pr_auc_mean": float(df["pr_auc"].mean()),
        "brier_mean": float(df["brier"].mean()),
        "ece_mean": float(df["ece"].mean()),
    }


def build_param_grid(model_name: str, raw_grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    def _as_float_list(values: Iterable[Any]) -> List[float]:
        return [float(v) for v in values]

    def _as_int_list(values: Iterable[Any]) -> List[int]:
        return [int(v) for v in values]

    if model_name == "svm_rbf":
        keys = ["C", "gamma"]
        values = [
            _as_float_list(raw_grid.get("C", [1.0])),
            raw_grid.get("gamma", ["scale"]),
        ]
    elif model_name == "catboost":
        keys = ["depth", "n_estimators", "learning_rate", "class_weights"]
        values = [
            _as_int_list(raw_grid.get("depth", [6])),
            _as_int_list(raw_grid.get("n_estimators", [500])),
            _as_float_list(raw_grid.get("learning_rate", [0.03])),
            [raw_grid.get("class_weights", "balanced")],
        ]
    elif model_name == "mlp":
        keys = ["layers", "dropout", "weight_decay", "early_stopping"]
        values = [
            raw_grid.get("layers", [[16]]),
            _as_float_list(raw_grid.get("dropout", [0.0])),
            _as_float_list(raw_grid.get("weight_decay", [1e-4])),
            [raw_grid.get("early_stopping", True)],
        ]
    else:
        raise ValueError()

    combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    if model_name != "mlp":
        return combos

    unique: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for combo in combos:
        key = (
            tuple(combo["layers"]),
            float(combo["weight_decay"]),
            bool(combo["early_stopping"]),
        )
        if key not in unique:
            unique[key] = combo
    deduped = list(unique.values())
    return deduped


def get_builder(model_name: str):
    if model_name == "svm_rbf":
        return build_svm
    if model_name == "catboost":
        return build_catboost
    if model_name == "mlp":
        return build_mlp
    raise ValueError()


def to_binary_target(y_raw: pd.Series) -> pd.Series:
    if y_raw.dtype == "object":
        s = y_raw.astype(str).str.strip()
        s_lower = s.str.lower()
        common_map = {
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "positive": 1,
            "negative": 0,
            "favorable": 1,
            "unfavorable": 0,
        }
        y_mapped = s_lower.map(common_map)
        if y_mapped.isna().any():
            uniq = sorted(s.dropna().unique().tolist())
            if len(uniq) != 2:
                raise ValueError()
            label_map = {uniq[0]: 0, uniq[1]: 1}
            return s.map(label_map).astype(int)
        return y_mapped.astype(int)

    y = y_raw.astype(int)
    uniq = sorted(y.unique().tolist())
    if len(uniq) != 2:
        raise ValueError()
    if uniq != [0, 1]:
        y = y.map({uniq[0]: 0, uniq[1]: 1}).astype(int)
    return y


def _extract_raw_score(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError()
        return proba[:, 1].astype(float)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        return np.asarray(decision, dtype=float).reshape(-1)
    raise ValueError()


def fit_calibrated_model(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    calibrator_type: str,
    inner_cv: StratifiedKFold,
) -> CalibratedModelWrapper:
    if calibrator_type not in {"platt", "isotonic"}:
        raise ValueError()
    if len(X_train) != len(y_train):
        raise ValueError()
    if len(X_train) == 0:
        raise ValueError()

    LOGGER.info("Обучение калибратора (%s)", calibrator_type)
    y_arr = np.asarray(y_train, dtype=int)
    oof_raw = np.full(shape=len(y_arr), fill_value=np.nan, dtype=float)

    for tr_idx, va_idx in inner_cv.split(X_train, y_train):
        model_fold = clone(base_model)
        model_fold.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_raw[va_idx] = _extract_raw_score(model_fold, X_train.iloc[va_idx])

    if np.isnan(oof_raw).any():
        raise RuntimeError()

    final_model = clone(base_model)
    final_model.fit(X_train, y_train)

    if calibrator_type == "platt":
        calibrator = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        calibrator.fit(oof_raw.reshape(-1, 1), y_arr)
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof_raw, y_arr)

    return CalibratedModelWrapper(
        model=final_model,
        calibrator=calibrator,
        calibrator_type=calibrator_type,
    )


def select_best_params_and_calibrator(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    categorical: List[str],
    candidates: List[Dict[str, Any]],
    calib_methods: List[str],
    inner_cv: StratifiedKFold,
) -> FoldSelection:
    builder = get_builder(model_name)
    best_selection: FoldSelection | None = None

    for params in candidates:
        by_calibrator: Dict[str, List[MetricDict]] = {method: [] for method in calib_methods}
        for tr_idx, va_idx in inner_cv.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            estimator = builder(params)
            if estimator is None:
                raise RuntimeError()

            base_estimator = Pipeline(
                steps=[
                    ("preprocess", build_preprocess(features=features, categorical=categorical)),
                    ("model", estimator),
                ]
            )

            for method in calib_methods:
                calibrator = build_calibrator(base_estimator=base_estimator, method=method)
                calibrator.fit(X_tr, y_tr)
                p_cal = calibrator.predict_proba(X_va)[:, 1]
                by_calibrator[method].append(compute_metrics(y_va, p_cal))

        mean_by_calibrator = {
            method: aggregate_metric_mean(items) for method, items in by_calibrator.items()
        }
        best_method = min(mean_by_calibrator.keys(), key=lambda m: mean_by_calibrator[m]["brier_mean"])
        candidate_best_brier = mean_by_calibrator[best_method]["brier_mean"]

        if best_selection is None or candidate_best_brier < best_selection.best_inner_brier:
            best_selection = FoldSelection(
                best_params=params,
                best_calibrator=best_method,
                best_inner_brier=float(candidate_best_brier),
            )

    if best_selection is None:
        raise RuntimeError()
    return best_selection


def run_model_nested_cv(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    categorical: List[str],
    model_grid: Dict[str, Any],
    calib_methods: List[str],
    outer_cv: StratifiedKFold,
    inner_cv: StratifiedKFold,
    preds_dir: Path,
) -> List[OuterFoldResult]:
    builder = get_builder(model_name)
    candidates = build_param_grid(model_name, model_grid)
    if not candidates:
        raise ValueError()

    if model_name == "catboost" and builder({}) is None:
        raise RuntimeError()

    results: List[OuterFoldResult] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
        LOGGER.info("Модель %s: внешний фолд %s/5", model_name, fold_idx)
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

        selection = select_best_params_and_calibrator(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            features=features,
            categorical=categorical,
            candidates=candidates,
            calib_methods=calib_methods,
            inner_cv=inner_cv,
        )

        final_estimator = builder(selection.best_params)
        if final_estimator is None:
            raise RuntimeError()

        base_estimator = Pipeline(
            steps=[
                ("preprocess", build_preprocess(features=features, categorical=categorical)),
                ("model", final_estimator),
            ]
        )
        final_model = fit_calibrated_model(
            base_model=base_estimator,
            X_train=X_train,
            y_train=y_train,
            calibrator_type=selection.best_calibrator,
            inner_cv=inner_cv,
        )
        p_cal = final_model.predict_proba(X_test)

        fold_metrics = compute_metrics(y_test, p_cal)
        results.append(
            OuterFoldResult(
                outer_fold=fold_idx,
                best_params=json.dumps(selection.best_params, ensure_ascii=False),
                best_calibrator=selection.best_calibrator,
                roc_auc=fold_metrics["roc_auc"],
                pr_auc=fold_metrics["pr_auc"],
                brier=fold_metrics["brier"],
                ece=fold_metrics["ece"],
            )
        )

        preds_path = preds_dir / f"preds_{model_name}_fold{fold_idx}.csv"
        pd.DataFrame({"y_true": y_test.to_numpy(), "p_cal": p_cal}).to_csv(
            preds_path,
            index=False,
            encoding="utf-8",
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск вложенной кросс-валидации для SVM/CatBoost/MLP с калибровкой")
    parser.add_argument("--data", required=True, type=Path, help="Путь к файлу датасета")
    parser.add_argument(
        "--config_features",
        default=Path("configs/kstar_features.json"),
        type=Path,
        help="Путь к JSON с признаками и целевой переменной",
    )
    parser.add_argument(
        "--config_grids",
        default=Path("configs/grids.yaml"),
        type=Path,
        help="Путь к YAML с сетками гиперпараметров",
    )
    args = parser.parse_args()

    setup_logging()
    LOGGER.info("Загрузка данных из %s", args.data)

    df = load_tabular(args.data)
    features_cfg = json.loads(args.config_features.read_text(encoding="utf-8"))
    grids_cfg = yaml.safe_load(args.config_grids.read_text(encoding="utf-8"))

    features = features_cfg.get("features", [])
    target = features_cfg.get("target", "target")
    if not features:
        raise ValueError("В config_features не найден список признаков")
    if target not in df.columns:
        preprocess_path = Path("ml/inference_pack/preprocess.json")
        if preprocess_path.exists():
            prep = json.loads(preprocess_path.read_text(encoding="utf-8"))
            fallback_target = prep.get("target_col")
            if fallback_target in df.columns:
                LOGGER.warning(
                    "Целевая переменная '%s' отсутствует в данных; используем target_col='%s' из %s",
                    target,
                    fallback_target,
                    preprocess_path,
                )
                target = fallback_target
            else:
                raise ValueError()
        else:
            raise ValueError()

    feature_pool = features
    missing = [col for col in feature_pool if col not in df.columns]
    if missing:
        preprocess_path = Path("ml/inference_pack/preprocess.json")
        if preprocess_path.exists():
            prep = json.loads(preprocess_path.read_text(encoding="utf-8"))
            raw_features = prep.get("raw_feature_cols", [])
            raw_missing = [col for col in raw_features if col not in df.columns]
            if raw_features and not raw_missing:
                LOGGER.warning(
                    "Сконфигурированные признаки не совпадают с датасетом (отсутствует: %s). "
                    "Для обучения используем raw_feature_cols из %s.",
                    len(missing),
                    preprocess_path,
                )
                feature_pool = raw_features
            else:
                raise ValueError()
        else:
            raise ValueError()

    data = df[feature_pool + [target]].copy()
    data = data.dropna(subset=[target])
    X = data[feature_pool]
    y = to_binary_target(data[target])
    categorical = infer_categorical_features(X, feature_pool)
    LOGGER.info("Данные подготовлены: %s строк, %s признаков.", X.shape[0], X.shape[1])

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    calib_methods = grids_cfg.get("calibration", {}).get("methods", ["platt", "isotonic"])

    tables_dir = Path("reports/tables")
    preds_dir = Path("reports/preds")
    tables_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    model_to_table = {
        "svm_rbf": tables_dir / "ncv_results_svm.csv",
        "catboost": tables_dir / "ncv_results_cat.csv",
        "mlp": tables_dir / "ncv_results_mlp.csv",
    }

    for model_name in ["svm_rbf", "catboost", "mlp"]:
        LOGGER.info("Старт nested-CV для модели: %s", model_name)
        model_results = run_model_nested_cv(
            model_name=model_name,
            X=X,
            y=y,
            features=feature_pool,
            categorical=categorical,
            model_grid=grids_cfg.get(model_name, {}),
            calib_methods=calib_methods,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            preds_dir=preds_dir,
        )
        out_df = pd.DataFrame([asdict(row) for row in model_results])
        out_df.to_csv(model_to_table[model_name], index=False, encoding="utf-8")
        LOGGER.info("Результаты модели %s сохранены в %s", model_name, model_to_table[model_name])

    LOGGER.info("Вложенная кросс-валидация завершена")


if __name__ == "__main__":
    main()
