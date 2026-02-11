from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from ..core.config import settings
from ..models.onnx_runtime import OnnxModel
from ..utils.validators import RangeSpec, validate_ranges, convert_units_if_needed
from ..utils.calibration import calibrate
from ..utils.ood import mahalanobis, is_ood


def load_signature(path: str | Path) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, str]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    features = data.get("input", {}).get("features", [])
    feature_order: List[str] = []
    ranges: Dict[str, Dict[str, float]] = {}
    units: Dict[str, str] = {}

    for feat in features:
        name = feat.get("name")
        if not name:
            continue
        feature_order.append(name)

        mn = feat.get("min")
        mx = feat.get("max")
        if mn is not None and mx is not None:
            ranges[name] = {"low": float(mn), "high": float(mx)}

        unit = feat.get("unit")
        if unit:
            units[name] = str(unit)

    return feature_order, ranges, units


class MissingFeatureError(ValueError):
    pass


class BadInputError(ValueError):
    pass


def classify(p: float, t_low: float, t_high: float) -> str:
    if p < t_low:
        return "low"
    if p > t_high:
        return "high"
    return "undetermined"


def confidence(p_cal: float) -> float:
    return abs(p_cal - 0.5)


class PredictorService:
    def __init__(
        self,
        onnx_model: OnnxModel,
        feature_order: List[str],
        ranges: Dict[str, RangeSpec],
        t_low: float,
        t_high: float,
        mu: Optional[np.ndarray] = None,
        inv_cov: Optional[np.ndarray] = None,
        ood_threshold: Optional[float] = None,
        calib: Optional[Dict[str, Any]] = None,
    ):
        if not feature_order:
            raise ValueError()
        self.model = onnx_model
        self.feature_order = feature_order
        self.ranges = ranges
        self.t_low = t_low
        self.t_high = t_high
        self.mu = mu
        self.inv_cov = inv_cov
        self.ood_threshold = ood_threshold
        self.calib = calib

    def predict(self, features: Dict[str, float], unit_convert: bool) -> Tuple[Dict[str, Any], int]:
        t0 = time.perf_counter()

        feats = convert_units_if_needed(features, unit_convert)

        numeric_feats: Dict[str, float] = {}
        for f in self.feature_order:
            if f not in feats:
                continue
            try:
                numeric_feats[f] = float(feats[f])
            except (TypeError, ValueError):
                raise BadInputError(f"Некорректный тип для признака '{f}': {feats[f]}")

        missing = [f for f in self.feature_order if f not in feats]
        if missing:
            raise MissingFeatureError(f"Отсутствуют обязательные признаки: {missing}")

        try:
            validate_ranges(numeric_feats, self.ranges)
        except ValueError as e:
            raise BadInputError(str(e))

        x = np.array([[numeric_feats[f] for f in self.feature_order]], dtype=np.float32)

        out = self.model.predict_proba(x)
        if out.ndim == 2 and out.shape[1] >= 2:
            p_raw = float(out[0, 1])
        else:
            p_raw = float(out.ravel()[0])

        p_cal = calibrate(p_raw, self.calib)
        cls = classify(p_cal, self.t_low, self.t_high)
        conf = confidence(p_cal)

        ood = False
        if self.mu is not None and self.inv_cov is not None and self.ood_threshold is not None:
            score = mahalanobis(x.reshape(-1), self.mu, self.inv_cov)
            ood = is_ood(score, self.ood_threshold)

        timing_ms = int((time.perf_counter() - t0) * 1000)

        resp = {
            "class": cls,
            "p_raw": p_raw,
            "p_cal": p_cal,
            "confidence": conf,
            "thresholds": {"t_low": self.t_low, "t_high": self.t_high},
            "ood": ood,
            "model_version": settings.model_version,
            "schema_version": settings.schema_version,
            "timing_ms": timing_ms,
        }
        return resp, timing_ms
