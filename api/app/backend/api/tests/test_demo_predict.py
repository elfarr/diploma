import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.api.app as appmod

app = appmod.app


class _FakePredictor:
    def predict(self, _features, _unit_convert):
        out = {
            "class": "low",
            "p_raw": 0.21,
            "p_raw_svm": 0.2,
            "p_raw_cat": 0.22,
            "p_raw_mlp": 0.21,
            "p_svm": 0.2,
            "p_cat": 0.22,
            "p_mlp": 0.21,
            "p_cal": 0.2,
            "model_used": "catboost",
            "bin_id": 2,
            "undetermined": False,
            "confidence": 0.3,
            "thresholds": {"t_low": 0.35, "t_high": 0.65},
            "ood": False,
            "model_version": "v2.0.0",
            "schema_version": "1",
            "timing_ms": 12,
            "calibration_by_model": {"svm_rbf": "platt", "catboost": "isotonic", "mlp": "platt"},
            "explain": [],
        }
        return out, 12


def _payload():
    return {"features": {"demo_feature": 1.0}, "unit_convert": False}


def test_demo_predict_disabled_returns_404():
    old_demo_enabled = appmod.settings.DEMO_ENABLED
    old_pred = appmod.PRED
    appmod.settings.DEMO_ENABLED = False
    appmod.PRED = _FakePredictor()
    appmod._demo_rate_hits.clear()
    try:
        with TestClient(app) as client:
            resp = client.post("/demo/predict", json=_payload())
            assert resp.status_code == 404
    finally:
        appmod.settings.DEMO_ENABLED = old_demo_enabled
        appmod.PRED = old_pred


def test_demo_predict_enabled_no_auth_and_without_technical_fields():
    old_demo_enabled = appmod.settings.DEMO_ENABLED
    old_pred = appmod.PRED
    appmod.settings.DEMO_ENABLED = True
    appmod.PRED = _FakePredictor()
    appmod._demo_rate_hits.clear()
    try:
        with TestClient(app) as client:
            resp = client.post("/demo/predict", json=_payload())
            assert resp.status_code == 200
            body = resp.json()
            for key in ["class", "p_cal", "confidence", "thresholds", "ood", "models", "ensemble"]:
                assert key in body
            for key in ["model_version", "schema_version", "timing_ms"]:
                assert key not in body
    finally:
        appmod.settings.DEMO_ENABLED = old_demo_enabled
        appmod.PRED = old_pred


def test_demo_predict_rate_limit_20_per_5m():
    old_demo_enabled = appmod.settings.DEMO_ENABLED
    old_pred = appmod.PRED
    appmod.settings.DEMO_ENABLED = True
    appmod.PRED = _FakePredictor()
    appmod._demo_rate_hits.clear()
    try:
        with TestClient(app) as client:
            payload = _payload()
            for _ in range(20):
                resp = client.post("/demo/predict", json=payload)
                assert resp.status_code == 200
            limited = client.post("/demo/predict", json=payload)
            assert limited.status_code == 429
    finally:
        appmod.settings.DEMO_ENABLED = old_demo_enabled
        appmod.PRED = old_pred


def test_predict_no_auth_when_demo_enabled_true():
    old_demo_enabled = appmod.settings.DEMO_ENABLED
    old_pred = appmod.PRED
    appmod.settings.DEMO_ENABLED = True
    appmod.PRED = _FakePredictor()
    try:
        with TestClient(app, headers={"X-Model-Version": appmod.settings.MODEL_VERSION}) as client:
            resp = client.post("/predict", json=_payload())
            assert resp.status_code == 200
    finally:
        appmod.settings.DEMO_ENABLED = old_demo_enabled
        appmod.PRED = old_pred


def test_predict_requires_auth_when_demo_enabled_false():
    old_demo_enabled = appmod.settings.DEMO_ENABLED
    old_pred = appmod.PRED
    appmod.settings.DEMO_ENABLED = False
    appmod.PRED = _FakePredictor()
    try:
        with TestClient(app, headers={"X-Model-Version": appmod.settings.MODEL_VERSION}) as client:
            resp = client.post("/predict", json=_payload())
            assert resp.status_code == 401
    finally:
        appmod.settings.DEMO_ENABLED = old_demo_enabled
        appmod.PRED = old_pred


def test_api_meta_no_auth_when_demo_enabled_true():
    old_demo_enabled = appmod.settings.DEMO_ENABLED
    appmod.settings.DEMO_ENABLED = True
    try:
        with TestClient(app) as client:
            resp = client.get("/api/meta")
            assert resp.status_code == 200
    finally:
        appmod.settings.DEMO_ENABLED = old_demo_enabled


def test_api_meta_requires_auth_when_demo_enabled_false():
    old_demo_enabled = appmod.settings.DEMO_ENABLED
    appmod.settings.DEMO_ENABLED = False
    try:
        with TestClient(app) as client:
            resp = client.get("/api/meta")
            assert resp.status_code == 401
    finally:
        appmod.settings.DEMO_ENABLED = old_demo_enabled
