from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_signature_features_ranges(signature_path: str | Path) -> tuple[List[str], Dict[str, Dict[str, float]]]:
    data = json.loads(Path(signature_path).read_text(encoding="utf-8"))
    features = data.get("input", {}).get("features", [])

    feature_names: List[str] = []
    ranges: Dict[str, Dict[str, float]] = {}
    for feat in features:
        name = feat.get("name")
        if not name:
            continue
        feature_names.append(name)

        mn = feat.get("min")
        mx = feat.get("max")
        if mn is not None or mx is not None:
            ranges[name] = {"low": mn, "high": mx}
    return feature_names, ranges


def build_meta_payload(
    model_version: str,
    schema_version: str,
    thresholds: Dict[str, float],
    models: List[Dict[str, Any]],
    features: List[str],
    ranges: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    return {
        "model_version": model_version,
        "schema_version": schema_version,
        "thresholds": thresholds,
        "models": models,
        "features": features,
        "ranges": ranges,
    }

