from fastapi import Response, Header, HTTPException, status
from .config import settings

def add_version_headers(response: Response):
    response.headers["X-Model-Version"] = settings.MODEL_VERSION
    response.headers["X-Schema-Version"] = settings.SCHEMA_VERSION


async def enforce_model_version(x_model_version: str | None = Header(default=None)):
    if x_model_version and x_model_version != settings.MODEL_VERSION:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"requested {x_model_version}, available {settings.MODEL_VERSION}",
        )
    return True
