from __future__ import annotations

import time
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from ..core.config import settings
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


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def select_model(
    p_svm: float,
    p_cat: float,
    p_mlp: float,
    competence: Dict[str, Any],
    fallback_model: str = "catboost",
) -> Dict[str, Any]:
    p_avg = float((float(p_svm) + float(p_cat) + float(p_mlp)) / 3.0)
    bin_id = int(math.floor(p_avg * 10.0))
    bin_id = max(0, min(9, bin_id))

    ece_map = competence.get("ece", {}) if isinstance(competence, dict) else {}
    brier_map = competence.get("brier", {}) if isinstance(competence, dict) else {}
    candidates: List[Tuple[str, float, float]] = []

    for model_name in ("svm_rbf", "catboost", "mlp"):
        ece_arr = ece_map.get(model_name)
        brier_arr = brier_map.get(model_name)
        if not isinstance(ece_arr, list) or bin_id >= len(ece_arr):
            continue
        if not isinstance(brier_arr, list) or bin_id >= len(brier_arr):
            continue

        ece_val = _safe_float(ece_arr[bin_id])
        if ece_val is None:
            continue
        brier_val = _safe_float(brier_arr[bin_id])
        if brier_val is None:
            brier_val = float("inf")
        candidates.append((model_name, ece_val, brier_val))

    if not candidates:
        return {"winner": fallback_model, "bin_id": bin_id, "p_avg": p_avg}

    winner = min(candidates, key=lambda x: (x[1], x[2]))[0]
    return {"winner": winner, "bin_id": bin_id, "p_avg": p_avg}


def load_artifacts(model_dir: str | Path, default_t_low: float, default_t_high: float) -> Dict[str, Any]:
    model_path = Path(model_dir)
    thresholds = {"t_low": float(default_t_low), "t_high": float(default_t_high)}
    competence: Dict[str, Any] = {"schema_version": "1.0", "bins": [], "ece": {}, "brier": {}}

    thr_path = model_path / "thresholds.json"
    if thr_path.exists():
        try:
            thr_data = json.loads(thr_path.read_text(encoding="utf-8"))
            thresholds = {
                "t_low": float(thr_data.get("t_low", default_t_low)),
                "t_high": float(thr_data.get("t_high", default_t_high)),
            }
        except Exception:
            pass

    competence_path = model_path / "competence_by_risk_bin.json"
    if competence_path.exists():
        try:
            competence = json.loads(competence_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError() from exc
    else:
        raise FileNotFoundError()

    calibrations: Dict[str, Dict[str, Any]] = {}
    model_names = ("svm_rbf", "catboost", "mlp")
    for model_name in model_names:
        cpath = model_path / f"calib_{model_name}.json"
        try:
            calibrations[model_name] = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError() from exc

    return {"thresholds": thresholds, "competence": competence, "calibrations": calibrations}


def classify(p: float, t_low: float, t_high: float) -> str:
    if t_low > t_high:
        t_low, t_high = t_high, t_low
    decision_threshold = (t_low + t_high) / 2.0
    if p < decision_threshold:
        return "low"
    return "high"


def confidence(p_cal: float) -> float:
    return abs(p_cal - 0.5)


def _is_binary_spec(spec: Optional[RangeSpec]) -> bool:
    if spec is None:
        return False
    return float(spec.low) == 0.0 and float(spec.high) == 1.0


def _group_prefix(name: str) -> Optional[str]:
    if "_" not in name:
        return None
    prefix = name.split("_", 1)[0].strip()
    return prefix or None


class PredictorService:
    def __init__(
        self,
        model_by_name: Dict[str, Any],
        feature_order: List[str],
        ranges: Dict[str, RangeSpec],
        model_dir: str | Path,
        default_t_low: float,
        default_t_high: float,
        mu: Optional[np.ndarray] = None,
        inv_cov: Optional[np.ndarray] = None,
        ood_threshold: Optional[float] = None,
        calib: Optional[Dict[str, Any]] = None,
    ):
        if not feature_order:
            raise ValueError()
        required_models = {"svm_rbf", "catboost", "mlp"}
        if set(model_by_name.keys()) != required_models:
            raise ValueError()
        if len({id(model_by_name[m]) for m in required_models}) != 3:
            raise ValueError()
        self.model_by_name = model_by_name
        self.feature_order = feature_order
        self.ranges = ranges
        self.mu = mu
        self.inv_cov = inv_cov
        self.ood_threshold = ood_threshold
        self.calib = calib
        self.artifacts = load_artifacts(
            model_dir=model_dir,
            default_t_low=default_t_low,
            default_t_high=default_t_high,
        )
        self.t_low = float(self.artifacts["thresholds"]["t_low"])
        self.t_high = float(self.artifacts["thresholds"]["t_high"])
        self.competence = self.artifacts["competence"]
        self.calib_by_model = self.artifacts.get("calibrations", {})
        self.categorical_group_prefixes = self._build_categorical_group_prefixes()
        self.calibration_method_by_model = {
            model_name: (
                self.calib_by_model.get(model_name, {}).get("type")
                if isinstance(self.calib_by_model.get(model_name), dict)
                else None
            )
            for model_name in ("svm_rbf", "catboost", "mlp")
        }

    def _build_categorical_group_prefixes(self) -> set[str]:
        by_prefix: Dict[str, List[str]] = {}
        for feature_name in self.feature_order:
            prefix = _group_prefix(feature_name)
            if prefix is None:
                continue
            if not _is_binary_spec(self.ranges.get(feature_name)):
                continue
            by_prefix.setdefault(prefix, []).append(feature_name)
        return {prefix for prefix, fields in by_prefix.items() if len(fields) >= 2}

    def _build_inputs(self, numeric_feats: Dict[str, float]) -> Tuple[Any, np.ndarray]:
        row = [float(numeric_feats[f]) for f in self.feature_order]
        x_np = np.array([row], dtype=np.float32)
        if pd is not None:
            x_sklearn: Any = pd.DataFrame([dict(zip(self.feature_order, row))], columns=self.feature_order)
        else:
            x_sklearn = x_np
        return x_sklearn, x_np

    def _neutral_value(self, feature_name: str, current_value: float) -> float:
        spec = self.ranges.get(feature_name)
        if _is_binary_spec(spec):
            return 0.0
        if spec is not None:
            return float((spec.low + spec.high) / 2.0)
        return 0.0

    def _explain_feature_candidates(self, numeric_feats: Dict[str, float]) -> List[str]:
        candidates: List[str] = []
        for feature_name in self.feature_order:
            if feature_name not in numeric_feats:
                continue
            prefix = _group_prefix(feature_name)
            current_value = float(numeric_feats[feature_name])
            if prefix is not None and prefix in self.categorical_group_prefixes:
                if current_value >= 0.5:
                    candidates.append(feature_name)
                continue
            candidates.append(feature_name)
        return candidates

    def _build_local_explain(
        self,
        numeric_feats: Dict[str, float],
        base_p_final: float,
        top_k: int = 5,
    ) -> List[Dict[str, float | str]]:
        scored: List[Tuple[str, float, float]] = []
        for feature_name in self._explain_feature_candidates(numeric_feats):
            current_value = float(numeric_feats[feature_name])
            neutral_value = self._neutral_value(feature_name, current_value)
            if math.isclose(current_value, neutral_value, rel_tol=0.0, abs_tol=1e-12):
                continue

            perturbed = dict(numeric_feats)
            perturbed[feature_name] = neutral_value
            try:
                x_sklearn_mut, x_np_mut = self._build_inputs(perturbed)
                p_mut = float(self.predict_one(x_sklearn=x_sklearn_mut, x_np=x_np_mut)["p_final"])
            except Exception:
                continue

            contribution = float(base_p_final - p_mut)
            if math.isclose(contribution, 0.0, rel_tol=0.0, abs_tol=1e-12):
                continue
            scored.append((feature_name, current_value, contribution))

        scored.sort(key=lambda item: abs(item[2]), reverse=True)
        top = scored[:top_k]
        return [
            {
                "name": feature_name,
                "value": value,
                "contribution": contribution,
            }
            for feature_name, value, contribution in top
        ]

    @staticmethod
    def _predict_positive_proba(model: Any, x: np.ndarray) -> float:
        if hasattr(model, "predict_proba"):
            out = np.asarray(model.predict_proba(x))
            if out.ndim == 2 and out.shape[1] >= 2:
                return float(out[0, 1])
            return float(out.ravel()[0])
        if hasattr(model, "decision_function"):
            score = float(np.asarray(model.decision_function(x)).reshape(-1)[0])
            return float(1.0 / (1.0 + np.exp(-score)))
        raise ValueError()

    def predict_one(self, x_sklearn: Any, x_np: np.ndarray) -> Dict[str, Any]:
        p_raw_svm = self._predict_positive_proba(self.model_by_name["svm_rbf"], x_sklearn)
        p_raw_cat = self._predict_positive_proba(self.model_by_name["catboost"], x_np)
        p_raw_mlp = self._predict_positive_proba(self.model_by_name["mlp"], x_sklearn)

        p_svm = calibrate(p_raw_svm, self.calib_by_model.get("svm_rbf"))
        p_cat = calibrate(p_raw_cat, self.calib_by_model.get("catboost"))
        p_mlp = calibrate(p_raw_mlp, self.calib_by_model.get("mlp"))
        sel = select_model(
            p_svm=p_svm,
            p_cat=p_cat,
            p_mlp=p_mlp,
            competence=self.competence,
            fallback_model="catboost",
        )

        winner = sel["winner"]
        if winner == "svm_rbf":
            p_final = float(p_svm)
        elif winner == "mlp":
            p_final = float(p_mlp)
        else:
            p_final = float(p_cat)

        risk_class = classify(p_final, self.t_low, self.t_high)
        conf = confidence(p_final)
        return {
            "risk_class": risk_class,
            "p_raw": float((p_raw_svm + p_raw_cat + p_raw_mlp) / 3.0),
            "p_raw_svm": float(p_raw_svm),
            "p_raw_cat": float(p_raw_cat),
            "p_raw_mlp": float(p_raw_mlp),
            "p_svm": float(p_svm),
            "p_cat": float(p_cat),
            "p_mlp": float(p_mlp),
            "p_final": p_final,
            "confidence": conf,
            "model_used": winner,
            "bin_id": int(sel["bin_id"]),
            "p_avg": float(sel["p_avg"]),
        }

    def predict(self, features: Dict[str, float], unit_convert: bool) -> Tuple[Dict[str, Any], int]:
        t0 = time.perf_counter()

        feats = dict(convert_units_if_needed(features, unit_convert))

        numeric_feats: Dict[str, float] = {}
        for f in self.feature_order:
            if f not in feats:
                continue
            try:
                numeric_feats[f] = float(feats[f])
            except (TypeError, ValueError):
                raise BadInputError()

        try:
            validate_ranges(numeric_feats, self.ranges)
        except ValueError as e:
            raise BadInputError(str(e))

        missing = [f for f in self.feature_order if f not in numeric_feats]
        if missing:
            sample = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise MissingFeatureError(f"missing_features: {sample}{suffix}")

        x_sklearn, x_np = self._build_inputs(numeric_feats)
        pred = self.predict_one(x_sklearn=x_sklearn, x_np=x_np)

        ood = False
        if self.mu is not None and self.inv_cov is not None and self.ood_threshold is not None:
            score = mahalanobis(x_np.reshape(-1), self.mu, self.inv_cov)
            ood = is_ood(score, self.ood_threshold)

        timing_ms = int((time.perf_counter() - t0) * 1000)

        resp = {
            "class": pred["risk_class"],
            "risk_class": pred["risk_class"],
            "p_raw": pred["p_raw"],
            "p_raw_svm": pred["p_raw_svm"],
            "p_raw_cat": pred["p_raw_cat"],
            "p_raw_mlp": pred["p_raw_mlp"],
            "p_svm": pred["p_svm"],
            "p_cat": pred["p_cat"],
            "p_mlp": pred["p_mlp"],
            "p_cal": pred["p_final"],
            "calibration_by_model": self.calibration_method_by_model,
            "confidence": pred["confidence"],
            "thresholds": {"t_low": self.t_low, "t_high": self.t_high},
            "ood": ood,
            "model_used": pred["model_used"],
            "bin_id": pred["bin_id"],
            "model_version": settings.model_version,
            "schema_version": settings.schema_version,
            "timing_ms": timing_ms,
            "explain": self._build_local_explain(
                numeric_feats=numeric_feats,
                base_p_final=float(pred["p_final"]),
                top_k=5,
            ),
        }
        return resp, timing_ms
