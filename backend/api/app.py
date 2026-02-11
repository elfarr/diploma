import time
import json
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
from .services.predictor import (
    load_signature,
    PredictorService,
    MissingFeatureError,
    BadInputError,
)
from .models.onnx_runtime import OnnxModel
from .utils.validators import RangeSpec

registry = CollectorRegistry()
REQ_COUNT = Counter("api_requests_total", "Total requests", ["path", "method", "code"], registry=registry)
REQ_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path", "method"], registry=registry)
PRED_COUNT = Counter("api_predict_total", "Total predictions", ["class"], registry=registry)

FEATURE_ORDER = []
RANGES = {}
PRED = None
THRESHOLDS = {"t_low": settings.T_LOW, "t_high": settings.T_HIGH}


def _load_thresholds(model_dir: Path) -> dict:
    thr_path = model_dir / "thresholds.json"
    if thr_path.exists():
        try:
            data = json.loads(thr_path.read_text(encoding="utf-8"))
            return {
                "t_low": float(data.get("t_low", settings.T_LOW)),
                "t_high": float(data.get("t_high", settings.T_HIGH)),
            }
        except Exception:
            pass
    return {"t_low": settings.T_LOW, "t_high": settings.T_HIGH}


def _init_predictor() -> None:
    """Load model assets once on startup so /predict uses real inference."""
    global FEATURE_ORDER, RANGES, PRED, THRESHOLDS
    model_dir = Path(settings.MODEL_DIR)
    FEATURE_ORDER, ranges_raw, _units = load_signature(model_dir / "signature.json")
    RANGES = {name: RangeSpec(low=spec["low"], high=spec["high"]) for name, spec in ranges_raw.items()}
    THRESHOLDS = _load_thresholds(model_dir)

    onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        onnx_path = model_dir / "onnx" / "model.onnx"

    model = OnnxModel.load(str(onnx_path))
    PRED = PredictorService(
        onnx_model=model,
        feature_order=FEATURE_ORDER,
        ranges=RANGES,
        t_low=THRESHOLDS["t_low"],
        t_high=THRESHOLDS["t_high"],
        mu=None,
        inv_cov=None,
        ood_threshold=None,
        calib=None,
    )


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
        "thresholds": THRESHOLDS,
        "features": FEATURE_ORDER,
    }

@app.post("/predict", dependencies=[Depends(auth_bearer), Depends(enforce_model_version)])
@instrument("/predict", "POST")
async def predict(request: Request, response: Response):
    if PRED is None:
        raise HTTPException(status_code=503, detail="Предиктор недоступен")

    add_version_headers(response)
    payload = await request.json()
    feats = payload.get("features", {})
    if not isinstance(feats, dict):
        return JSONResponse(status_code=400, content={"error": "Некорректный формат features"})

    unit_convert = bool(payload.get("unit_convert", settings.UNIT_CONVERT_DEFAULT))
    try:
        out, timing_ms = PRED.predict(feats, unit_convert)
    except MissingFeatureError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    except BadInputError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Не удалось выполнить инференс"})

    PRED_COUNT.labels(out.get("class", "unknown")).inc()
    return out

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.post("/admin/reload", dependencies=[Depends(auth_bearer)])
async def reload_model(response: Response):
    add_version_headers(response)
    _init_predictor()
    return {"reloaded": PRED is not None, "model_version": settings.MODEL_VERSION}
