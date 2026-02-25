from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RangeSpec:
    low: float
    high: float


def validate_ranges(values: Dict[str, float], ranges: Dict[str, RangeSpec]) -> None:
    for k, v in values.items():
        if k not in ranges:
            continue
        rs = ranges[k]
        if v < rs.low or v > rs.high:
            raise ValueError(f"'{k}'={v} вне диапазона [{rs.low}, {rs.high}]")


def detect_ood(values: Dict[str, float], ranges: Dict[str, RangeSpec]) -> bool:
    for k, v in values.items():
        if k not in ranges:
            continue
        rs = ranges[k]
        if v < rs.low or v > rs.high:
            return True
    return False


def convert_units_if_needed(values: Dict[str, float], unit_convert: bool) -> Dict[str, float]:
    if not unit_convert:
        return values
    return values
