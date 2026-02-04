from __future__ import annotations

import json
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(logger, event: dict):
    meta = {
        "time": event.get("time", now_iso()),
        "version": event.get("version"),
        "p_cal": event.get("p_cal"),
        "class": event.get("class"),
        "ood": event.get("ood"),
        "timing_ms": event.get("timing_ms"),
    }
    logger.info(json.dumps(meta, ensure_ascii=False))
