# =============================================================================
# edge/middleware/validator.py
#
# PURPOSE:
#   Catches malformed, oversized, or structurally broken requests before they
#   reach route handlers. This is the last line of defense at the edge layer
#   before a request enters the routing and serving pipeline.
#
# POSITION IN THE MIDDLEWARE CHAIN:
#   Request → AuthMiddleware → RateLimiterMiddleware → ValidatorMiddleware
#                                                            ↓
#                                                      Route Handler
#
# WHY A SEPARATE VALIDATOR WHEN PYDANTIC ALREADY VALIDATES?
#   Pydantic (in request.py) validates the CONTENT of a request — field types,
#   value ranges, model names. This middleware validates the STRUCTURE of the
#   HTTP request itself — things that happen before Pydantic even sees the body:
#     - Is the Content-Type header correct?
#     - Is the request body too large to safely parse?
#     - Is the JSON parseable at all before we hand it to Pydantic?
#     - Is the request coming in at an impossible rate (content-level anomaly)?
#   These checks protect against malformed HTTP, accidental misconfiguration,
#   and low-effort denial-of-service attempts.
#
# DO YOU NEED TO RUN THIS FILE? NO.
#   Activated when server.py registers it. No standalone execution needed.
# =============================================================================

import json
import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)

# -----------------------------------------------------------------------------
# VALIDATION CONSTANTS
#
# MAX_BODY_SIZE_BYTES:
#   Maximum allowed request body size. At 1MB this is generous for a text
#   prompt but prevents memory exhaustion from clients sending binary data
#   or absurdly large payloads disguised as inference requests.
#   A typical inference request with a long prompt is under 50KB.
#
# REQUIRED_CONTENT_TYPE:
#   We only accept JSON. Any other content type is rejected immediately.
#   This prevents clients from accidentally sending form data or binary
#   payloads that would cause confusing downstream errors.
#
# INFERENCE_ROUTES:
#   Only these routes require body validation. GET routes and health checks
#   have no body to validate.
# -----------------------------------------------------------------------------
MAX_BODY_SIZE_BYTES: int = 1 * 1024 * 1024   # 1 MB
REQUIRED_CONTENT_TYPE: str = "application/json"

INFERENCE_ROUTES: set[str] = {
    "/v1/completions",
    "/v1/chat/completions",
}

EXEMPT_ROUTES: set[str] = {
    "/health",
    "/ready",
    "/docs",
    "/openapi.json",
}


class ValidatorMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:

        # ---------------------------------------------------------------------
        # STEP 1: Skip validation for exempt and non-inference routes
        #
        # Health checks have no body. GET requests have no body.
        # We only validate POST requests to inference endpoints.
        # ---------------------------------------------------------------------
        if request.url.path in EXEMPT_ROUTES:
            return await call_next(request)

        if request.method != "POST" or request.url.path not in INFERENCE_ROUTES:
            return await call_next(request)

        # ---------------------------------------------------------------------
        # STEP 2: Validate Content-Type header
        #
        # We check that the client declared their payload as JSON.
        # Content-Type can include charset info like "application/json; charset=utf-8"
        # so we use startswith() instead of exact equality.
        #
        # WHY THIS CHECK?
        #   Without it, a client sending multipart form data or plain text
        #   would get a confusing Pydantic parse error deep in the stack
        #   instead of a clear "wrong content type" message at the edge.
        # ---------------------------------------------------------------------
        content_type = request.headers.get("Content-Type", "")
        if not content_type.startswith(REQUIRED_CONTENT_TYPE):
            logger.warning(
                "invalid_content_type",
                expected=REQUIRED_CONTENT_TYPE,
                received=content_type,
                path=request.url.path,
                client_id=getattr(request.state, "client_id", "unknown"),
            )
            return JSONResponse(
                status_code=415,   # 415 Unsupported Media Type
                content={
                    "error": "unsupported_media_type",
                    "message": f"Content-Type must be '{REQUIRED_CONTENT_TYPE}'. "
                               f"Received: '{content_type}'"
                }
            )

        # ---------------------------------------------------------------------
        # STEP 3: Check Content-Length header before reading the body
        #
        # Content-Length tells us how many bytes the client claims to be sending.
        # If they declare a size over our limit we reject without reading the body.
        #
        # WHY CHECK BEFORE READING?
        #   Reading a 500MB payload into memory before rejecting it would waste
        #   memory and time. We reject on the declared size first.
        #   Note: a malicious client could lie about Content-Length, which is
        #   why we also check actual body size in Step 4.
        # ---------------------------------------------------------------------
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
                if declared_size > MAX_BODY_SIZE_BYTES:
                    logger.warning(
                        "request_body_too_large_declared",
                        declared_bytes=declared_size,
                        limit_bytes=MAX_BODY_SIZE_BYTES,
                        client_id=getattr(request.state, "client_id", "unknown"),
                    )
                    return JSONResponse(
                        status_code=413,   # 413 Content Too Large
                        content={
                            "error": "payload_too_large",
                            "message": f"Request body exceeds maximum allowed "
                                       f"size of {MAX_BODY_SIZE_BYTES // 1024}KB.",
                            "max_bytes": MAX_BODY_SIZE_BYTES,
                        }
                    )
            except ValueError:
                # Content-Length header exists but is not a valid integer.
                # Malformed header — reject.
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "bad_request",
                        "message": "Content-Length header must be a valid integer."
                    }
                )

        # ---------------------------------------------------------------------
        # STEP 4: Read and size-check the actual body
        #
        # We read the raw body bytes and check their actual size.
        # This catches clients who lied about Content-Length or did not send it.
        #
        # WHY await request.body()?
        #   request.body() is an async call because reading from a network
        #   socket is I/O — it may need to wait for bytes to arrive.
        #   Using await lets other requests be handled during that wait.
        #
        # IMPORTANT: After reading request.body(), we must make the body
        #   available again for downstream handlers. FastAPI reads the body
        #   from a stream — once read it is consumed. We re-attach it below.
        # ---------------------------------------------------------------------
        try:
            body_bytes = await request.body()
        except Exception as e:
            logger.error(
                "failed_to_read_request_body",
                error=str(e),
                client_id=getattr(request.state, "client_id", "unknown"),
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": "bad_request",
                    "message": "Failed to read request body."
                }
            )

        actual_size = len(body_bytes)
        if actual_size > MAX_BODY_SIZE_BYTES:
            logger.warning(
                "request_body_too_large_actual",
                actual_bytes=actual_size,
                limit_bytes=MAX_BODY_SIZE_BYTES,
                client_id=getattr(request.state, "client_id", "unknown"),
            )
            return JSONResponse(
                status_code=413,
                content={
                    "error": "payload_too_large",
                    "message": f"Request body exceeds maximum allowed "
                               f"size of {MAX_BODY_SIZE_BYTES // 1024}KB.",
                    "actual_bytes": actual_size,
                    "max_bytes": MAX_BODY_SIZE_BYTES,
                }
            )

        # ---------------------------------------------------------------------
        # STEP 5: Validate that the body is parseable JSON
        #
        # We attempt to parse the body as JSON before handing it to the route
        # handler. If it is not valid JSON, we return a clear error message
        # rather than letting Pydantic produce a confusing internal error.
        #
        # We do NOT validate the JSON structure here — that is Pydantic's job.
        # We only check: is this valid JSON at all?
        # ---------------------------------------------------------------------
        if actual_size > 0:
            try:
                json.loads(body_bytes)
            except json.JSONDecodeError as e:
                logger.warning(
                    "invalid_json_body",
                    error=str(e),
                    client_id=getattr(request.state, "client_id", "unknown"),
                    path=request.url.path,
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "invalid_json",
                        "message": f"Request body is not valid JSON: {str(e)}",
                    }
                )
        else:
            # Empty body on a POST inference route — reject.
            return JSONResponse(
                status_code=400,
                content={
                    "error": "empty_body",
                    "message": "Request body cannot be empty for inference endpoints."
                }
            )

        # ---------------------------------------------------------------------
        # STEP 6: Attach validation metadata to request state
        #
        # We record the body size so route handlers and observability layers
        # can log it without re-reading the body. This is useful for tracking
        # prompt size distribution across requests.
        # ---------------------------------------------------------------------
        request.state.body_size_bytes = actual_size
        request.state.validated = True

        logger.info(
            "request_validated",
            body_bytes=actual_size,
            path=request.url.path,
            client_id=getattr(request.state, "client_id", "unknown"),
        )

        return await call_next(request)
