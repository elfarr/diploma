import time
import json
import inspect
import functools
from pathlib import Path
import joblib
from fastapi import FastAPI, Depends, Response, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, Summary, Histogram, Counter
from starlette.responses import PlainTextResponse
from src.api.meta import build_meta_payload
from .core.config import settings
from .core.versioning import add_version_headers, enforce_model_version
from .core.middleware.request_id import RequestIDMiddleware
from .auth.deps import auth_bearer
from .services.predictor import (
    load_signature,
    PredictorService,
    MissingFeatureError,
    BadInputError,
)
from .models.onnx_runtime import OnnxModel
from .schemas.request import PredictRequest
from .utils.validators import RangeSpec

registry = CollectorRegistry()
REQ_COUNT = Counter("api_requests_total", "Total requests", ["path"], registry=registry)
REQ_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path", "method"], registry=registry)
PRED_COUNT = Counter("api_predict_total", "Total predictions", ["class"], registry=registry)

FEATURE_ORDER = []
RANGES = {}
PRED = None
THRESHOLDS = {"t_low": settings.T_LOW, "t_high": settings.T_HIGH}
MODEL_CATALOG = [
    {"model_name": "svm_rbf", "model_version": settings.MODEL_VERSION},
    {"model_name": "catboost", "model_version": settings.MODEL_VERSION},
    {"model_name": "mlp", "model_version": settings.MODEL_VERSION},
]


def _init_predictor() -> None:
    global FEATURE_ORDER, RANGES, PRED, THRESHOLDS
    model_dir = Path(settings.MODEL_DIR)
    FEATURE_ORDER, ranges_raw, _units = load_signature(model_dir / "signature.json")
    RANGES = {name: RangeSpec(low=spec["low"], high=spec["high"]) for name, spec in ranges_raw.items()}

    model_by_name = {}

    for model_name in ("svm_rbf", "mlp"):
        model_path = model_dir / f"model_{model_name}.pkl"
        if not model_path.exists():
            raise RuntimeError()
        model_by_name[model_name] = joblib.load(model_path)

    cat_model_path = model_dir / "model_catboost.pkl"
    cat_onnx_path = model_dir / "model_catboost.onnx"
    if cat_model_path.exists():
        try:
            model_by_name["catboost"] = joblib.load(cat_model_path)
        except ModuleNotFoundError:
            model_by_name["catboost"] = OnnxModel.load(str(cat_onnx_path))
    else:
        model_by_name["catboost"] = OnnxModel.load(str(cat_onnx_path))
    PRED = PredictorService(
        model_by_name=model_by_name,
        feature_order=FEATURE_ORDER,
        ranges=RANGES,
        model_dir=model_dir,
        default_t_low=settings.T_LOW,
        default_t_high=settings.T_HIGH,
        mu=None,
        inv_cov=None,
        ood_threshold=None,
        calib=None,
    )
    THRESHOLDS = {"t_low": PRED.t_low, "t_high": PRED.t_high}


try:
    _init_predictor()
except Exception:
    FEATURE_ORDER = []
    RANGES = {}
    PRED = None

def instrument(path_template: str, method: str):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                dur = time.perf_counter() - start
                REQ_LATENCY.labels(path_template, method).observe(dur)
        wrapper.__signature__ = inspect.signature(func)
        return wrapper
    return decorator

app = FastAPI(title=settings.APP_NAME)
app.add_middleware(RequestIDMiddleware)

@app.middleware("http")
async def prometheus_request_counter(request: Request, call_next):
    try:
        return await call_next(request)
    finally:
        REQ_COUNT.labels(request.url.path).inc()

@app.get("/healthz")
@instrument("/healthz", "GET")
async def healthz(response: Response):
    add_version_headers(response)
    return {"status": "ok"}

@app.get("/meta", dependencies=[Depends(auth_bearer), Depends(enforce_model_version)])
@instrument("/meta", "GET")
async def meta(response: Response):
    add_version_headers(response)
    ranges = {
        name: {"low": spec.low, "high": spec.high}
        for name, spec in RANGES.items()
    }
    return build_meta_payload(
        model_version=settings.MODEL_VERSION,
        schema_version=settings.SCHEMA_VERSION,
        thresholds=THRESHOLDS,
        models=MODEL_CATALOG,
        features=FEATURE_ORDER,
        ranges=ranges,
    )

@app.post("/predict", dependencies=[Depends(auth_bearer), Depends(enforce_model_version)])
@instrument("/predict", "POST")
async def predict(payload: PredictRequest, response: Response):
    if PRED is None:
        raise HTTPException(status_code=503)

    add_version_headers(response)
    feats = payload.features
    unit_convert = bool(payload.unit_convert)
    try:
        out, timing_ms = PRED.predict(feats, unit_convert)
    except MissingFeatureError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    except BadInputError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception:
        return JSONResponse(status_code=500, content={"error": str(e)})

    PRED_COUNT.labels(out.get("class", "unknown")).inc()
    undetermined = out.get("class") == "undetermined"

    model_proba_map = {
        "svm_rbf": out.get("p_svm"),
        "catboost": out.get("p_cat"),
        "mlp": out.get("p_mlp"),
    }
    calibration_method_map = out.get("calibration_by_model", {})
    models = []
    for model in MODEL_CATALOG:
        model_name = model["model_name"]
        model_proba = model_proba_map.get(model_name)
        calibration_method = calibration_method_map.get(model_name)
        models.append(
            {
                "model_name": model_name,
                "model_version": model["model_version"],
                "proba": model_proba,
                "calibrated": model_proba is not None,
                "calibration_method": calibration_method,
                "undetermined": undetermined,
            }
        )
    valid_probas = [m["proba"] for m in models if m["proba"] is not None]
    ensemble_proba = sum(valid_probas) / len(valid_probas) if valid_probas else None

    return {
        "models": models,
        "ensemble": {
            "proba": ensemble_proba,
            "strategy": "mean_calibrated",
        },
        "explain": out.get("explain", []),
        "class": out.get("class"),
        "p_raw": out.get("p_raw"),
        "p_raw_svm": out.get("p_raw_svm"),
        "p_raw_cat": out.get("p_raw_cat"),
        "p_raw_mlp": out.get("p_raw_mlp"),
        "p_svm": out.get("p_svm"),
        "p_cat": out.get("p_cat"),
        "p_mlp": out.get("p_mlp"),
        "p_cal": out.get("p_cal"),
        "model_used": out.get("model_used"),
        "bin_id": out.get("bin_id"),
        "undetermined": out.get("undetermined"),
        "confidence": out.get("confidence"),
        "thresholds": out.get("thresholds"),
        "ood": out.get("ood"),
        "model_version": out.get("model_version"),
        "schema_version": out.get("schema_version"),
        "timing_ms": out.get("timing_ms"),
    }

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.post("/admin/reload", dependencies=[Depends(auth_bearer)])
async def reload_model(response: Response):
    add_version_headers(response)
    _init_predictor()
    return {"reloaded": PRED is not None, "model_version": settings.MODEL_VERSION}
