from __future__ import annotations

import math
from typing import Dict, List, Optional, Union


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def calibrate(p_raw: float, calib: Optional[Union[Dict, None]]) -> float:
    if not calib or "type" not in calib:
        return float(p_raw)

    p = min(max(float(p_raw), 1e-6), 1.0 - 1e-6)
    ctype = calib.get("type")

    if ctype == "platt":
        a = float(calib.get("a", 0.0))
        b = float(calib.get("b", 0.0))
        z = a * _logit(p) + b
        return _sigmoid(z)

    if ctype == "isotonic":
        xs: List[float] = calib.get("x", [])
        ys: List[float] = calib.get("y", [])
        if not xs or not ys or len(xs) != len(ys):
            return p_raw
        if p <= xs[0]:
            return float(ys[0])
        if p >= xs[-1]:
            return float(ys[-1])
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if p >= xs[mid]:
                lo = mid
            else:
                hi = mid
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        if x1 == x0:
            return float(y0)
        t = (p - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    return float(p_raw)
