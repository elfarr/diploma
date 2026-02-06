from fastapi import Header, HTTPException, status
from ..core.config import settings

async def auth_bearer(authorization: str = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    allowed = {settings.API_TOKEN}
    if getattr(settings, "DEV_BEARER", None):
        allowed.add(settings.DEV_BEARER)
    if token not in allowed:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return True
