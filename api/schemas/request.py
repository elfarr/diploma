from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Словарь признаков: имя - значение")
    unit_convert: bool = Field(False, description="Конвертировать единицы измерения")

    @field_validator("features")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("Поле features не может быть пустым")
        return v
