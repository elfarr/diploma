from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

RANDOM_SEED = 42


def build_svm(params: Dict[str, Any]) -> SVC:
    class_weight = params.get("class_weight", "balanced")
    probability = bool(params.get("probability", True))
    if isinstance(class_weight, dict):
        normalized: Dict[Any, float] = {}
        for k, v in class_weight.items():
            key = k
            if isinstance(k, str):
                try:
                    key = int(k)
                except ValueError:
                    try:
                        key = float(k)
                    except ValueError:
                        key = k
            normalized[key] = float(v)
        class_weight = normalized

    return SVC(
        probability=probability,
        kernel="rbf",
        class_weight=class_weight,
        C=float(params.get("C", 1.0)),
        gamma=params.get("gamma", "scale"),
        random_state=RANDOM_SEED,
    )


def build_catboost(params: Dict[str, Any]) -> Optional[Any]:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None

    cat_params: Dict[str, Any] = {
        "loss_function": "Logloss",
        "verbose": False,
        "random_seed": RANDOM_SEED,
        "depth": int(params.get("depth", 6)),
        "n_estimators": int(params.get("n_estimators", 500)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
    }

    class_weights = params.get("class_weights")
    if class_weights == "balanced":
        cat_params["auto_class_weights"] = "Balanced"
    elif class_weights is not None:
        cat_params["class_weights"] = class_weights

    return CatBoostClassifier(**cat_params)


def build_xgboost(params: Dict[str, Any]) -> Optional[Any]:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    xgb_params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "n_estimators": int(params.get("n_estimators", 250)),
        "max_depth": int(params.get("max_depth", 2)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "min_child_weight": float(params.get("min_child_weight", 4.0)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.7)),
        "reg_lambda": float(params.get("reg_lambda", 6.0)),
        "gamma": float(params.get("gamma", 1.0)),
        "scale_pos_weight": float(params.get("scale_pos_weight", 4.0)),
    }
    return XGBClassifier(**xgb_params)


def build_mlp(params: Dict[str, Any]) -> MLPClassifier:
    layers = tuple(params.get("layers", [16]))
    early_stopping = bool(params.get("early_stopping", True))
    alpha = float(params.get("weight_decay", 1e-4))

    return MLPClassifier(
        hidden_layer_sizes=layers,
        alpha=alpha,
        early_stopping=early_stopping,
        n_iter_no_change=20,
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
