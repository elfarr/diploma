from typing import List
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "RiskAPI"
    ENV: str = "dev"
    CORS_ORIGINS: str = "http://localhost:5173"
    DEV_BEARER: str = "secret"
    API_TOKEN: str = ""
    MODEL_DIR: str = "backend/models/v2.0.0"
    MODEL_VERSION: str = "dev"
    SCHEMA_VERSION: str = "1"
    T_LOW: float = 0.35
    T_HIGH: float = 0.65
    UNIT_CONVERT_DEFAULT: bool = False

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @property
    def app_name(self):  
        return self.APP_NAME

    @property
    def env(self):  
        return self.ENV

    @property
    def cors_origins(self) -> List[str]:
        raw = (self.CORS_ORIGINS or "").strip()
        if not raw:
            return ["http://localhost:5173"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    @property
    def dev_bearer(self):  
        return self.DEV_BEARER

    @property
    def api_token(self):  
        return self.API_TOKEN

    @property
    def model_dir(self):  
        return self.MODEL_DIR

    @property
    def model_version(self): 
        return self.MODEL_VERSION

    @property
    def schema_version(self):  
        return self.SCHEMA_VERSION

    @property
    def t_low(self): 
        return self.T_LOW

    @property
    def t_high(self):  
        return self.T_HIGH

    @property
    def unit_convert_default(self):  
        return self.UNIT_CONVERT_DEFAULT


settings = Settings()
