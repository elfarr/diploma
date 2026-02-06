import sys
import copy
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api.app as appmod
app = appmod.app
from api.utils.validators import RangeSpec


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
        key = appmod.FEATURE_ORDER[0]
        appmod.RANGES[key] = RangeSpec(low=-1.0, high=1.0)
        if appmod.PRED:
            appmod.PRED.ranges = appmod.RANGES
        payload["features"][key] = 10.0
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400


def test_bad_type():
    with TestClient(app, headers={"Authorization": "Bearer secret", "X-Model-Version": "v2.0.0"}) as client:
        payload = base_payload()
        key = appmod.FEATURE_ORDER[1]
        payload["features"][key] = "not_a_number"
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400
