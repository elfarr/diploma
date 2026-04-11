from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RangeSpec:
    low: float
    high: float


def _norm(value: str) -> str:
    return value.lower().replace(" ", "")


def _is_binary_spec(spec: RangeSpec) -> bool:
    return float(spec.low) == 0.0 and float(spec.high) == 1.0


def _impossible_value_message(name: str, value: float) -> str | None:
    key = _norm(name)

    if value < 0.0:
        return "Некорректное значение"
    if name == "relative risk" and value <= 0.0:
        return "Некорректное значение"
    if ("ох" in key or "chol" in key or "totalchol" in key) and value <= 0.0:
        return "Некорректное значение"
    if ("лпнп" in key or "ldl" in key) and value <= 0.0:
        return "Некорректное значение"
    if ("лпвп" in key or "hdl" in key) and value <= 0.0:
        return "Некорректное значение"
    if ("тг" in key or "triglycer" in key or key == "tg") and value <= 0.0:
        return "Некорректное значение"
    if ("мочева" in key or "uric" in key) and value <= 0.0:
        return "Некорректное значение"
    if ("сад" in key or "sbp" in key or "дад" in key or "dbp" in key) and value <= 0.0:
        return "Некорректное значение"
    if "фв" in key:
        if value <= 0.0:
            return "Некорректное значение"
        if value > 100.0:
            return "Некорректное значение"
    if (("эхо" in key)
            and value <= 0.0):
        return "Некорректное значение"
    return None


def _is_implausible_value(name: str, value: float) -> bool:
    key = _norm(name)

    if name == "relative risk":
        return value > 20.0
    if "ох" in key or "chol" in key or "totalchol" in key:
        return value < 1.0 or value > 15.0
    if "лпнп" in key or "ldl" in key:
        return value < 0.3 or value > 12.0
    if "лпвп" in key or "hdl" in key:
        return value < 0.1 or value > 5.0
    if "тг" in key or "triglycer" in key or key == "tg":
        return value < 0.1 or value > 20.0
    if "мочева" in key or "uric" in key:
        return value < 50.0 or value > 1500.0
    if "эхолп" in key:
        return value < 15.0 or value > 80.0
    if "эхокдр" in key:
        return value < 25.0 or value > 90.0
    if "эхомжп" in key or "эхозс" in key:
        return value < 4.0 or value > 25.0
    if "эхосдла" in key:
        return value < 5.0 or value > 130.0
    if "фв" in key:
        return value < 5.0 or value > 90.0
    if "эхоммлж" in key:
        return value < 20.0 or value > 700.0
    if "эхоиммлж" in key:
        return value < 10.0 or value > 350.0
    if "отт" in key:
        return value > 1.2
    if "сад" in key or "sbp" in key:
        return value < 60.0 or value > 260.0
    if "дад" in key or "dbp" in key:
        return value < 30.0 or value > 160.0
    return False


def validate_ranges(values: Dict[str, float], ranges: Dict[str, RangeSpec]) -> None:
    for k, v in values.items():
        rs = ranges.get(k)
        if rs is not None and _is_binary_spec(rs) and (v < 0.0 or v > 1.0):
            raise ValueError(
                f"Некорректное значение '{k}': для бинарного признака допустимы 0 или 1."
            )
        message = _impossible_value_message(k, v)
        if message is not None:
            raise ValueError(f"{message}. Поле: '{k}'.")


def detect_ood(values: Dict[str, float], ranges: Dict[str, RangeSpec]) -> bool:
    for k, v in values.items():
        rs = ranges.get(k)
        if rs is not None and _is_binary_spec(rs) and (v < 0.0 or v > 1.0):
            return True
        if _is_implausible_value(k, v):
            return True
        if _impossible_value_message(k, v) is not None:
            return True
    return False


def convert_units_if_needed(values: Dict[str, float], unit_convert: bool) -> Dict[str, float]:
    if not unit_convert:
        return values
    return values
