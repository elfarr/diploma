import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("backend.api.request")

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming_req_id = (request.headers.get("X-Request-ID") or "").strip()
        req_id = incoming_req_id or str(uuid.uuid4())
        request.state.request_id = req_id

        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            status_code = getattr(response, "status_code", "error")
            logger.info(
                "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
                req_id,
                request.method,
                request.url.path,
                status_code,
                duration_ms,
            )
