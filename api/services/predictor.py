from __future__ import annotations
import time
from typing import Dict, List, Tuple

import numpy as np

from api.core.config import settings
from api.models.onnx_runtime import OnnxModel
from api.utils.validators import RangeSpec, validate_ranges, detect_ood, convert_units_if_needed

def classify(p: float, t_low: float, t_high: float) -> str:
    if p < t_low:
        return "low"
    if p > t_high:
        return "high"
    return "undetermined"

def confidence_bucket(p: float, t_low: float, t_high: float) -> str:
    if p < t_low:
        dist = (t_low - p)
    elif p > t_high:
        dist = (p - t_high)
    else:
        dist = min(p - t_low, t_high - p)

    if dist >= 0.20:
        return "high"
    if dist >= 0.08:
        return "med"
    return "low"

class PredictorService:
    def __init__(
        self,
        onnx_model: OnnxModel,
        feature_order: List[str],
        ranges: Dict[str, RangeSpec],
        t_low: float,
        t_high: float,
    ):
        self.model = onnx_model
        self.feature_order = feature_order
        self.ranges = ranges
        self.t_low = t_low
        self.t_high = t_high

    def predict(self, features: Dict[str, float], unit_convert: bool) -> Tuple[Dict, int]:
        t0 = time.perf_counter()

        feats = convert_units_if_needed(features, unit_convert)

        missing = [f for f in self.feature_order if f not in feats]
        if missing:
            raise ValueError(f"Отсутствуют обязательные признаки: {missing[:5]}... (всего {len(missing)})")
        
        validate_ranges(feats, self.ranges)
        x = np.array([[float(feats[f]) for f in self.feature_order]], dtype=np.float32)

        proba = self.model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            p_unf = float(proba[0, 1])
        else:
            p_unf = float(np.ravel(proba)[0])

        label = classify(p_unf, self.t_low, self.t_high)
        conf = confidence_bucket(p_unf, self.t_low, self.t_high)
        ood = detect_ood(feats, self.ranges)

        timing_ms = int((time.perf_counter() - t0) * 1000)

        resp = {
            "class": label,
            "prob_cal": p_unf,
            "confidence": conf,
            "thresholds": {"t_low": self.t_low, "t_high": self.t_high},
            "ood": ood,
            "model_version": settings.model_version,
            "schema_version": settings.schema_version,
            "timing_ms": timing_ms,
        }
        return resp, timing_ms
