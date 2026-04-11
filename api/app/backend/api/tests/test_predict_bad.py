import sys
import copy
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.api.app as appmod
app = appmod.app


def base_payload():
    feats = {name: 0.0 for name in appmod.FEATURE_ORDER}
    return {"features": feats, "unit_convert": False}


def test_missing_feature():
    with TestClient(app, headers={"Authorization": "Bearer secret", "X-Model-Version": "v2.0.0"}) as client:
        payload = base_payload()
        missing_key = appmod.FEATURE_ORDER[0]
        payload["features"].pop(missing_key)
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


def test_out_of_range():
    with TestClient(app, headers={"Authorization": "Bearer secret", "X-Model-Version": "v2.0.0"}) as client:
        payload = base_payload()
        key = next((name for name in appmod.FEATURE_ORDER if "ОХ" in name or "chol" in name.lower()), appmod.FEATURE_ORDER[0])
        payload["features"][key] = 1000.0
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400


def test_bad_type():
    with TestClient(app, headers={"Authorization": "Bearer secret", "X-Model-Version": "v2.0.0"}) as client:
        payload = base_payload()
        key = appmod.FEATURE_ORDER[1]
        payload["features"][key] = "not_a_number"
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400
