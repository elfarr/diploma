from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.calibration import build_calibrator
from src.pipeline.models import build_catboost, build_mlp, build_svm
from src.pipeline.preprocess import build_preprocess

LOGGER = logging.getLogger("warmup")


def oversample_to_ratio(
    X: pd.DataFrame,
    y: pd.Series,
    target_pos_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if not (0.0 < target_pos_ratio < 1.0):
        return X, y
    y_arr = y.to_numpy(dtype=int)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return X, y

    desired_pos = int(np.ceil((target_pos_ratio / (1.0 - target_pos_ratio)) * n_neg))
    if desired_pos <= n_pos:
        return X, y

    add_n = desired_pos - n_pos
    rng = np.random.default_rng(random_state)
    sampled = rng.choice(pos_idx, size=add_n, replace=True)
    all_idx = np.concatenate([np.arange(len(y_arr)), sampled])
    rng.shuffle(all_idx)
    return X.iloc[all_idx].reset_index(drop=True), y.iloc[all_idx].reset_index(drop=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


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


def infer_categorical_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    categorical: List[str] = []
    for col in features:
        dtype = df[col].dtype
        if str(dtype) in {"object", "category", "bool"}:
            categorical.append(col)
    return categorical


def metric_dict(y_true: pd.Series, proba: Iterable[float], threshold: float = 0.5) -> Dict[str, float]:
    proba_arr = pd.Series(proba).astype(float)
    pred = (proba_arr >= threshold).astype(int)
    y_bin = y_true.astype(int)
    tp = int(((pred == 1) & (y_bin == 1)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    tn = int(((pred == 0) & (y_bin == 0)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {
        "roc_auc": float(roc_auc_score(y_true, proba_arr)),
        "pr_auc": float(average_precision_score(y_true, proba_arr)),
        "brier": float(brier_score_loss(y_true, proba_arr)),
        "sens": sens,
        "spec": spec,
    }


def aggregate_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    df = pd.DataFrame(items)
    return {
        "roc_auc_mean": float(df["roc_auc"].mean()),
        "pr_auc_mean": float(df["pr_auc"].mean()),
        "brier_mean": float(df["brier"].mean()),
        "sens_mean": float(df["sens"].mean()),
        "spec_mean": float(df["spec"].mean()),
        "roc_auc_std": float(df["roc_auc"].std(ddof=0)),
        "pr_auc_std": float(df["pr_auc"].std(ddof=0)),
        "brier_std": float(df["brier"].std(ddof=0)),
        "sens_std": float(df["sens"].std(ddof=0)),
        "spec_std": float(df["spec"].std(ddof=0)),
    }


def build_param_grid(model_name: str, raw_grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    def _as_float_list(values: Iterable[Any]) -> List[float]:
        return [float(v) for v in values]

    def _as_int_list(values: Iterable[Any]) -> List[int]:
        return [int(v) for v in values]

    if model_name == "svm_rbf":
        keys = ["C", "gamma", "class_weight"]
        values = [
            _as_float_list(raw_grid.get("C", [1.0])),
            raw_grid.get("gamma", ["scale"]),
            raw_grid.get("class_weight", ["balanced"]),
        ]
    elif model_name == "catboost":
        keys = ["depth", "n_estimators", "learning_rate", "class_weights"]
        values = [
            _as_int_list(raw_grid.get("depth", [6])),
            _as_int_list(raw_grid.get("n_estimators", [500])),
            _as_float_list(raw_grid.get("learning_rate", [0.03])),
            raw_grid.get("class_weights", ["balanced"]),
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
        raise ValueError(f"Неизвестная модель: {model_name}")

    combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    if model_name == "mlp":
        unique: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for c in combos:
            key = (
                tuple(c["layers"]),
                float(c["weight_decay"]),
                bool(c["early_stopping"]),
            )
            if key not in unique:
                unique[key] = c
        deduped = list(unique.values())
        if len(deduped) < len(combos):
            LOGGER.info(
                "Параметр dropout для MLP из конфига игнорируется в sklearn MLPClassifier; "
                "комбинации сокращены: %s -> %s.",
                len(combos),
                len(deduped),
            )
        return deduped

    return combos


def get_builder(model_name: str) -> Callable[[Dict[str, Any]], Any]:
    if model_name == "svm_rbf":
        return build_svm
    if model_name == "catboost":
        return build_catboost
    if model_name == "mlp":
        return build_mlp
    raise ValueError(f"Неизвестная модель: {model_name}")


def evaluate_inner_cv(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    categorical: List[str],
    builder: Callable[[Dict[str, Any]], Any],
    params: Dict[str, Any],
    seed: int,
    oversample_pos_ratio: float,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    fold_metrics: List[Dict[str, float]] = []

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        X_tr_fit, y_tr_fit = oversample_to_ratio(
            X_tr.reset_index(drop=True),
            y_tr.reset_index(drop=True),
            target_pos_ratio=oversample_pos_ratio,
            random_state=seed + fold_i,
        )

        estimator = builder(params)
        if estimator is None:
            raise RuntimeError("Оценщик недоступен.")

        pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocess(features=features, categorical=categorical)),
                ("model", estimator),
            ]
        )
        pipe.fit(X_tr_fit, y_tr_fit)
        proba = pipe.predict_proba(X_va)[:, 1]
        fold_metrics.append(metric_dict(y_va, proba))

    return aggregate_metrics(fold_metrics)


def evaluate_calibration_inner_cv(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    categorical: List[str],
    builder: Callable[[Dict[str, Any]], Any],
    best_params: Dict[str, Any],
    method: str,
    seed: int,
    oversample_pos_ratio: float,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    fold_metrics: List[Dict[str, float]] = []

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        X_tr_fit, y_tr_fit = oversample_to_ratio(
            X_tr.reset_index(drop=True),
            y_tr.reset_index(drop=True),
            target_pos_ratio=oversample_pos_ratio,
            random_state=seed + 100 + fold_i,
        )

        base_estimator = Pipeline(
            steps=[
                ("preprocess", build_preprocess(features=features, categorical=categorical)),
                ("model", builder(best_params)),
            ]
        )
        calibrator = build_calibrator(base_estimator=base_estimator, method=method)
        calibrator.fit(X_tr_fit, y_tr_fit)
        proba = calibrator.predict_proba(X_va)[:, 1]
        fold_metrics.append(metric_dict(y_va, proba))

    return aggregate_metrics(fold_metrics)


def fit_and_eval_on_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    categorical: List[str],
    builder: Callable[[Dict[str, Any]], Any],
    best_params: Dict[str, Any],
    best_calibrator: str | None,
    oversample_pos_ratio: float,
    seed: int,
) -> Dict[str, float]:
    base_estimator = Pipeline(
        steps=[
            ("preprocess", build_preprocess(features=features, categorical=categorical)),
            ("model", builder(best_params)),
        ]
    )
    if best_calibrator:
        final_model = build_calibrator(base_estimator=base_estimator, method=best_calibrator)
    else:
        final_model = clone(base_estimator)

    X_train_fit, y_train_fit = oversample_to_ratio(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        target_pos_ratio=oversample_pos_ratio,
        random_state=seed + 999,
    )
    final_model.fit(X_train_fit, y_train_fit)
    proba = final_model.predict_proba(X_test)[:, 1]
    return metric_dict(y_test, proba)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="Путь к файлу датасета")
    parser.add_argument(
        "--config_features",
        default=Path("configs/kstar_features.json"),
        type=Path,
        help="Путь к конфигу k_star/features",
    )
    parser.add_argument(
        "--config_grids",
        default=Path("configs/grids.yaml"),
        type=Path,
        help="Путь к конфигу сеток моделей",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--oversample-pos-ratio",
        default=0.45,
        type=float,
    )
    args = parser.parse_args()

    setup_logging()

    df = load_tabular(args.data)
    features_cfg = json.loads(args.config_features.read_text(encoding="utf-8"))
    grids_cfg = yaml.safe_load(args.config_grids.read_text(encoding="utf-8"))

    features = features_cfg.get("features", [])
    configured_k_star = features_cfg.get("k_star")
    target = features_cfg.get("target", "target")
    if not features:
        raise ValueError("В config_features не найдены признаки.")

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
    missing = [f for f in feature_pool if f not in df.columns]
    if missing:
        preprocess_path = Path("ml/inference_pack/preprocess.json")
        if preprocess_path.exists():
            prep = json.loads(preprocess_path.read_text(encoding="utf-8"))
            raw_features = prep.get("raw_feature_cols", [])
            raw_missing = [f for f in raw_features if f not in df.columns]
            if raw_features and not raw_missing:
                LOGGER.warning(
                    "Сконфигурированные признаки не совпадают с датасетом (отсутствует: %s)"
                    "Для обучения используем raw_feature_cols из %s",
                    len(missing),
                    preprocess_path,
                )
                feature_pool = raw_features
            else:
                raise ValueError(f"В датасете отсутствуют столбцы признаков: {missing[:5]}...")
        else:
            raise ValueError(f"В датасете отсутствуют столбцы признаков: {missing[:5]}...")

    data = df[feature_pool + [target]].copy()
    data = data.dropna(subset=[target])
    y_raw = data[target]
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
            y = s.map(label_map).astype(int)
        else:
            y = y_mapped.astype(int)
    else:
        y = y_raw.astype(int)
        uniq = sorted(y.unique().tolist())
        if len(uniq) != 2:
            raise ValueError()
        if uniq != [0, 1]:
            y = y.map({uniq[0]: 0, uniq[1]: 1}).astype(int)

    candidate_k_stars = [k for k in [10, 20, 30, 40, 50] if k <= len(feature_pool)]
    if not candidate_k_stars:
        candidate_k_stars = [len(feature_pool)]

    model_order = ["svm_rbf", "catboost", "mlp"]
    calib_methods = grids_cfg.get("calibration", {}).get("methods", ["platt", "isotonic"])
    leaderboard_rows: List[Dict[str, Any]] = []
    kstar_comparison_rows: List[Dict[str, Any]] = []
    kstar_results: Dict[str, Any] = {}

    best_k_star: int | None = None
    best_k_score = float("-inf")
    best_summary_for_models: Dict[str, Any] = {}
    best_features_used: List[str] = []

    for k_star in candidate_k_stars:
        features_used = feature_pool[:k_star]
        X = data[features_used]
        categorical = infer_categorical_features(X, features_used)

        LOGGER.info("Запуск для k_star=%s: %s признаков", k_star, len(features_used))
        LOGGER.info("Размер данных: %s строк, %s признаков", X.shape[0], X.shape[1])
        LOGGER.info("Обнаружено категориальных признаков: %s", len(categorical))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=args.seed,
            stratify=y,
        )

        per_k_models: Dict[str, Any] = {}
        best_test_sens_for_k = float("-inf")

        for model_name in model_order:
            model_grid = grids_cfg.get(model_name, {})
            builder = get_builder(model_name)
            LOGGER.info("Запуск модели: %s (k_star=%s)", model_name, k_star)

            if model_name == "catboost" and build_catboost({}) is None:
                LOGGER.warning("Модель catboost пропущена")
                per_k_models[model_name] = {"status": "skipped_no_catboost"}
                continue

            candidates = build_param_grid(model_name, model_grid)
            if not candidates:
                LOGGER.warning("Нет кандидатов параметров для %s. Пропуск", model_name)
                per_k_models[model_name] = {"status": "skipped_no_candidates"}
                continue

            candidate_results: List[Dict[str, Any]] = []
            for params in candidates:
                metrics = evaluate_inner_cv(
                    X=X_train,
                    y=y_train,
                    features=features_used,
                    categorical=categorical,
                    builder=builder,
                    params=params,
                    seed=args.seed,
                    oversample_pos_ratio=args.oversample_pos_ratio,
                )
                row = {
                    "k_star": k_star,
                    "model": model_name,
                    "stage": "param_search",
                    "params": json.dumps(params, ensure_ascii=False),
                    **metrics,
                }
                candidate_results.append(row)
                leaderboard_rows.append(row)

            best_row = max(
                candidate_results,
                key=lambda r: (
                    r["sens_mean"],
                    -r["brier_mean"],
                    r["roc_auc_mean"],
                ),
            )
            best_params = json.loads(best_row["params"])
            LOGGER.info(
                "Лучшие параметры %s по чувствительности: %s (sens=%.4f, brier=%.4f)",
                model_name,
                best_params,
                best_row["sens_mean"],
                best_row["brier_mean"],
            )

            calib_results: List[Dict[str, Any]] = []
            for method in calib_methods:
                calib_metrics = evaluate_calibration_inner_cv(
                    X=X_train,
                    y=y_train,
                    features=features_used,
                    categorical=categorical,
                    builder=builder,
                    best_params=best_params,
                    method=method,
                    seed=args.seed,
                    oversample_pos_ratio=args.oversample_pos_ratio,
                )
                row = {
                    "k_star": k_star,
                    "model": model_name,
                    "stage": "calibration",
                    "params": json.dumps(best_params, ensure_ascii=False),
                    "calibrator": method,
                    **calib_metrics,
                }
                calib_results.append(row)
                leaderboard_rows.append(row)

            best_calib_row = (
                max(
                    calib_results,
                    key=lambda r: (
                        r["sens_mean"],
                        -r["brier_mean"],
                        r["roc_auc_mean"],
                    ),
                )
                if calib_results
                else None
            )
            best_calibrator = best_calib_row["calibrator"] if best_calib_row else None
            if best_calibrator:
                LOGGER.info(
                    "Лучший калибратор для %s: %s (sens=%.4f, brier=%.4f)",
                    model_name,
                    best_calibrator,
                    best_calib_row["sens_mean"],
                    best_calib_row["brier_mean"],
                )

            test_metrics = fit_and_eval_on_test(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                features=features_used,
                categorical=categorical,
                builder=builder,
                best_params=best_params,
                best_calibrator=best_calibrator,
                oversample_pos_ratio=args.oversample_pos_ratio,
                seed=args.seed,
            )

            leaderboard_rows.append(
                {
                    "k_star": k_star,
                    "model": model_name,
                    "stage": "test_final",
                    "params": json.dumps(best_params, ensure_ascii=False),
                    "calibrator": best_calibrator,
                    "roc_auc_mean": test_metrics["roc_auc"],
                    "pr_auc_mean": test_metrics["pr_auc"],
                    "brier_mean": test_metrics["brier"],
                    "sens_mean": test_metrics["sens"],
                    "spec_mean": test_metrics["spec"],
                    "roc_auc_std": 0.0,
                    "pr_auc_std": 0.0,
                    "brier_std": 0.0,
                    "sens_std": 0.0,
                    "spec_std": 0.0,
                }
            )
            kstar_comparison_rows.append(
                {
                    "k_star": k_star,
                    "model": model_name,
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_brier": test_metrics["brier"],
                    "test_sens": test_metrics["sens"],
                    "test_spec": test_metrics["spec"],
                }
            )
            best_test_sens_for_k = max(best_test_sens_for_k, test_metrics["sens"])

            per_k_models[model_name] = {
                "status": "ok",
                "best_params": best_params,
                "best_calibrator": best_calibrator,
                "inner_best_metrics": {
                    "roc_auc": best_row["roc_auc_mean"],
                    "pr_auc": best_row["pr_auc_mean"],
                    "brier": best_row["brier_mean"],
                    "sens": best_row["sens_mean"],
                    "spec": best_row["spec_mean"],
                },
                "inner_calibration_metrics": {
                    r["calibrator"]: {
                        "roc_auc": r["roc_auc_mean"],
                        "pr_auc": r["pr_auc_mean"],
                        "brier": r["brier_mean"],
                        "sens": r["sens_mean"],
                        "spec": r["spec_mean"],
                    }
                    for r in calib_results
                },
                "test_metrics": test_metrics,
            }

        if best_test_sens_for_k > best_k_score:
            best_k_score = best_test_sens_for_k
            best_k_star = k_star
            best_summary_for_models = per_k_models
            best_features_used = features_used

        kstar_results[str(k_star)] = {
            "k_star": k_star,
            "n_features": len(features_used),
            "features_used": features_used,
            "best_test_sens": best_test_sens_for_k,
            "models": per_k_models,
        }

    summary: Dict[str, Any] = {
        "seed": args.seed,
        "k_star": best_k_star,
        "k_star_final": best_k_star,
        "k_star_default": configured_k_star,
        "k_star_candidates": candidate_k_stars,
        "best_k_star": best_k_star,
        "target": target,
        "n_features": len(best_features_used),
        "features_used": best_features_used,
        "k_star_metrics": [
            {
                "k_star": int(k),
                "best_test_sens": v["best_test_sens"],
            }
            for k, v in kstar_results.items()
        ],
        "k_star_results": kstar_results,
        "models": best_summary_for_models,
    }

    out_dir = Path("artifacts/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(leaderboard_rows).to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8")
    pd.DataFrame(kstar_comparison_rows).to_csv(
        out_dir / "kstar_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    print("\nСводка результатов")
    print(f"Лучший k_star по test Sensitivity: {best_k_star}")
    for model_name in model_order:
        model_info = best_summary_for_models.get(model_name, {})
        status = model_info.get("status")
        if status != "ok":
            print(f"- {model_name}: пропущено ({status})")
            continue
        best_params = model_info["best_params"]
        best_calib = model_info["best_calibrator"]
        metrics = model_info["inner_best_metrics"]
        print(
            f"- {model_name}: top-1 параметры={best_params}, "
            f"inner_sens={metrics['sens']:.4f}, inner_brier={metrics['brier']:.4f}, "
            f"лучший_калибратор={best_calib}"
        )
    print(f"сохранено в: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
