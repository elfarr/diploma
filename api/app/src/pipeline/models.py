from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

RANDOM_SEED = 42


def build_svm(params: Dict[str, Any]) -> SVC:
    return SVC(
        probability=True,
        kernel="rbf",
        class_weight="balanced",
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
