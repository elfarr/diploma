from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def build_calibrator(base_estimator, method: str) -> CalibratedClassifierCV:
    method_map = {
        "platt": "sigmoid",
        "isotonic": "isotonic",
    }

    calib_method = method_map[method]
    try:
        return CalibratedClassifierCV(estimator=base_estimator, method=calib_method, cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_estimator, method=calib_method, cv=3)
