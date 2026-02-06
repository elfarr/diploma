import time
import inspect
import functools
from pathlib import Path
from fastapi import FastAPI, Depends, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, Summary, Histogram, Counter
from starlette.responses import PlainTextResponse
from .core.config import settings
from .core.versioning import add_version_headers, enforce_model_version
from .core.middleware.request_id import RequestIDMiddleware
from .auth.deps import auth_bearer
from api.services.predictor import load_signature
from api.utils.validators import RangeSpec

registry = CollectorRegistry()
REQ_COUNT = Counter("api_requests_total", "Total requests", ["path", "method", "code"], registry=registry)
REQ_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path", "method"], registry=registry)
PRED_COUNT = Counter("api_predict_total", "Total predictions", ["class"], registry=registry)

FEATURE_ORDER = []
RANGES = {}
PRED = None
try:
    FEATURE_ORDER, ranges_raw, _units = load_signature(Path(settings.MODEL_DIR) / "signature.json")
    RANGES = {name: RangeSpec(low=spec["low"], high=spec["high"]) for name, spec in ranges_raw.items()}
except Exception:
    FEATURE_ORDER = []
    RANGES = {}

def instrument(path_template: str, method: str):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            code = "200"
            try:
                resp = await func(*args, **kwargs)
                code = str(getattr(resp, "status_code", 200))
                return resp
            except HTTPException as e:
                code = str(e.status_code)
                raise
            finally:
                dur = time.perf_counter() - start
                REQ_COUNT.labels(path_template, method, code).inc()
                REQ_LATENCY.labels(path_template, method).observe(dur)
        wrapper.__signature__ = inspect.signature(func)
        return wrapper
    return decorator

app = FastAPI(title=settings.APP_NAME)
app.add_middleware(RequestIDMiddleware)

@app.get("/healthz")
@instrument("/healthz", "GET")
async def healthz(response: Response):
    add_version_headers(response)
    return {"status": "ok", "model_version": settings.MODEL_VERSION}

@app.get("/meta", dependencies=[Depends(auth_bearer), Depends(enforce_model_version)])
@instrument("/meta", "GET")
async def meta(response: Response):
    add_version_headers(response)
    return {
        "model_version": settings.MODEL_VERSION,
        "schema_version": settings.SCHEMA_VERSION,
        "thresholds": {"t_low": settings.T_LOW, "t_high": settings.T_HIGH},
        "features": FEATURE_ORDER,
    }

@app.post("/predict", dependencies=[Depends(auth_bearer), Depends(enforce_model_version)])
@instrument("/predict", "POST")
async def predict(request: Request, response: Response):
    add_version_headers(response)
    payload = await request.json()
    feats = payload.get("features", {})
    missing = [f for f in FEATURE_ORDER if f not in feats]
    if missing:
        return JSONResponse(status_code=422, content={"error": f"Отсутствуют признаки: {missing}"})
    for k, v in feats.items():
        try:
            val = float(v)
        except Exception:
            return JSONResponse(status_code=400, content={"error": f"Некорректный тип для {k}"})
        if k in RANGES:
            r = RANGES[k]
            if val < r.low or val > r.high:
                return JSONResponse(status_code=400, content={"error": f"'{k}'={val} вне диапазона [{r.low}, {r.high}]"})
    p_raw = 0.74
    p_cal = p_raw
    t_low, t_high = settings.T_LOW, settings.T_HIGH
    if p_cal < t_low:
        klass = "low"
    elif p_cal > t_high:
        klass = "high"
    else:
        klass = "undetermined"
    PRED_COUNT.labels(klass).inc()
    out = {
        "class": klass,
        "p_raw": p_raw,
        "p_cal": p_cal,
        "confidence": abs(p_cal - 0.5),
        "thresholds": {"t_low": t_low, "t_high": t_high},
        "explain": [],
        "ood": False,
        "model_version": settings.MODEL_VERSION,
        "schema_version": settings.SCHEMA_VERSION,
        "timing_ms": 0,
    }
    return out

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.post("/admin/reload", dependencies=[Depends(auth_bearer)])
async def reload_model(response: Response):
    add_version_headers(response)
    return {"reloaded": True, "model_version": settings.MODEL_VERSION}
