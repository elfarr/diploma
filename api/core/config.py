from pydantic import BaseModel

class Settings(BaseModel):
    model_version: str = "v2.0.0"
    schema_version: str = "kstar_XX"
    t_low: float = 0.35
    t_high: float = 0.55
    unit_convert_default: bool = False

settings = Settings()
