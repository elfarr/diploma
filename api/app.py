from __future__ import annotations

from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException

from api.core.config import settings
from api.models.onnx_runtime import OnnxModel
from api.schemas.request import PredictRequest
from api.schemas.response import PredictResponse
from api.services.explainer import explain_stub
from api.services.predictor import PredictorService, load_signature
from api.utils.validators import RangeSpec

app = FastAPI(title="Kidney Tx Risk API", version="0.1.0")

FEATURE_ORDER = []  
RANGES: Dict[str, RangeSpec] = {}

MODEL = None
PRED = None


@app.on_event("startup")
def startup():
    global MODEL, PRED, FEATURE_ORDER, RANGES

    sig_path = Path(f"models/{settings.model_version}/signature.json")
    feature_order, ranges_raw, _units = load_signature(sig_path)
    if not feature_order:
        raise RuntimeError(f"Сигнатура пуста или нет признаков: {sig_path}")
    if len(set(feature_order)) != len(feature_order):
        raise RuntimeError(f"В сигнатуре есть дубли имён признаков: {sig_path}")

    FEATURE_ORDER = feature_order
    RANGES = {
        name: RangeSpec(low=spec["low"], high=spec["high"])
        for name, spec in ranges_raw.items()
    }

    MODEL = OnnxModel.load(f"models/{settings.model_version}/model.onnx")

    PRED = PredictorService(
        onnx_model=MODEL,
        feature_order=FEATURE_ORDER,
        ranges=RANGES,
        t_low=settings.t_low,
        t_high=settings.t_high,
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    return {
        "model_version": settings.model_version,
        "schema_version": settings.schema_version,
        "thresholds": {"t_low": settings.t_low, "t_high": settings.t_high},
        "features_count": len(FEATURE_ORDER),
        "features": FEATURE_ORDER,
        "ranges": {k: {"low": v.low, "high": v.high} for k, v in RANGES.items()},
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if PRED is None:
        raise HTTPException(status_code=503)

    try:
        base_resp, _ = PRED.predict(req.features, req.unit_convert)
        explain = explain_stub(req.features)
        base_resp["explain"] = explain
        return base_resp
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500)
