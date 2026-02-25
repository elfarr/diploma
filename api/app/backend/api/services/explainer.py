from __future__ import annotations

from typing import Dict, List


def explain(features: Dict[str, float], shap_values: Dict[str, float] | None = None, top_k: int = 5) -> List[dict]:
    if not shap_values:
        return []

    items = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    items = items[:top_k]
    total = sum(abs(v) for _, v in items) or 1.0

    result = []
    for name, val in items:
        direction = "neutral"
        if val > 0:
            direction = "up"
        elif val < 0:
            direction = "down"
        result.append(
            {
                "feature": name,
                "impact": abs(val) / total,
                "direction": direction,
            }
        )
    return result
