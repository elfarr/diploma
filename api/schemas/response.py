from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Literal

ClassLabel = Literal["low", "high", "undetermined"]
ConfidenceLabel = Literal["low", "med", "high"]

class ExplainItem(BaseModel):
    feature: str
    impact: float
    direction: Literal["up", "down"]

class PredictResponse(BaseModel):
    class_: ClassLabel = Field(..., alias="class")
    prob_cal: float
    confidence: ConfidenceLabel
    thresholds: Dict[str, float]
    explain: List[ExplainItem] = []
    ood: bool = False
    model_version: str
    schema_version: str
    timing_ms: int
