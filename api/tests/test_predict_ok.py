import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api.app as appmod
app = appmod.app


def make_full_payload():
    feats = {name: 0.0 for name in appmod.FEATURE_ORDER}
    return {"features": feats, "unit_convert": False}


def test_predict_ok():
    with TestClient(app, headers={"Authorization": "Bearer secret", "X-Model-Version": "v2.0.0"}) as client:
        payload = make_full_payload()
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        for key in ["class", "p_cal", "confidence", "thresholds", "ood", "model_version", "timing_ms"]:
            assert key in body
