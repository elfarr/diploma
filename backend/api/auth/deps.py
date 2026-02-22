from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ..core.config import settings

bearer_scheme = HTTPBearer(auto_error=False)


async def auth_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    authorization: Optional[str] = Header(default=None),
):
    token: Optional[str] = None

    if credentials and credentials.scheme and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()

    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    allowed = {settings.API_TOKEN}
    if getattr(settings, "DEV_BEARER", None):
        allowed.add(settings.DEV_BEARER)
    if token not in allowed:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return True
