from __future__ import annotations

from typing import Iterable, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocess(
    features: Sequence[str],
    categorical: Iterable[str] | None = None,
) -> ColumnTransformer:
    categorical_set = set(categorical or [])
    numeric_features = [f for f in features if f not in categorical_set]
    categorical_features = [f for f in features if f in categorical_set]

    transformers = []
    if numeric_features:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric_features))

    if categorical_features:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _build_ohe()),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")
