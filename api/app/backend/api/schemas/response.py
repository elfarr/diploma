from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional

ClassLabel = Literal["low", "high"]


class ExplainItem(BaseModel):
    feature: str
    impact: float
    direction: Literal["up", "down"]


class PredictResponse(BaseModel):
    class_: ClassLabel = Field(..., alias="class")
    p_raw: Optional[float] = Field(None, description="Вероятность положительного класса из модели")
    p_cal: float = Field(..., description="Калиброванная вероятность положительного класса")
    confidence: float = Field(..., description="Уверенность |p_cal-0.5|")
    thresholds: Dict[str, float]
    explain: List[ExplainItem] = []
    ood: bool = False
    model_version: str
    schema_version: str
    timing_ms: int
