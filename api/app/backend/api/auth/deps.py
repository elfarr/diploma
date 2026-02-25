import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ..core.config import settings

bearer_scheme = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


def _configured_api_token() -> Optional[str]:
    token = (getattr(settings, "API_TOKEN", None) or "").strip()
    return token or None


async def auth_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    authorization: Optional[str] = Header(default=None),
):
    api_token = _configured_api_token()
    if api_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    token: Optional[str] = None

    if credentials and credentials.scheme and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()

    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    if token != api_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return True
