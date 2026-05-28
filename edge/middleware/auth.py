# =============================================================================
# edge/middleware/auth.py
#
# PURPOSE:
#   Checks every incoming request for a valid API key before it reaches any
#   route handler. Unauthenticated or unauthorized requests are rejected here
#   and never touch business logic, queues, or GPU resources.
#
# HOW IT WORKS:
#   The client sends their API key in the Authorization header like this:
#     Authorization: Bearer ng-abc123xyz
#   This middleware reads that header, validates the key, and either:
#     - Attaches the client identity to the request and passes it through, or
#     - Returns a 401 or 403 response immediately and stops the request.
#
# WHY MIDDLEWARE AND NOT A ROUTE HANDLER?
#   Middleware wraps every route automatically. One piece of code protects
#   the entire API. If auth lived inside each route handler, we would need
#   to write and maintain it in every single route — and one forgotten check
#   would leave that route wide open.
#
# DO YOU NEED TO RUN THIS FILE? NO.
#   This file defines the middleware class. It activates when server.py
#   registers it with the FastAPI app. No standalone execution needed.
# =============================================================================

import os
import time
import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# structlog gives us structured JSON logs instead of plain text.
# Every log line will be a JSON object with fields like timestamp,
# request_id, client_id, and event — making logs searchable and parseable
# by tools like CloudWatch, Datadog, or Grafana Loki.
logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
# SIMULATED API KEY STORE
#
# In production this would be a database lookup or a Redis cache check.
# We store keys here as a dictionary for now so the logic is clear and testable
# without needing a database running.
#
# Structure: { "api_key_string": "client_identifier" }
# The client identifier is what we attach to the request so downstream
# components (rate limiter, metrics, logs) know which client sent this request.
#
# WHY NOT HARDCODE IN SERVER.PY?
#   Keeping the key store here means auth logic is self-contained. When we
#   later replace this with a Redis or database lookup, only this file changes.
# -----------------------------------------------------------------------------
VALID_API_KEYS: dict[str, str] = {
    "ng-dev-key-001": "client_dev",
    "ng-test-key-002": "client_test",
    "ng-prod-key-003": "client_prod_a",
    "ng-prod-key-004": "client_prod_b",
}

# Routes that bypass authentication entirely.
# Health check endpoints must be publicly accessible so load balancers
# and monitoring systems can probe them without needing an API key.
PUBLIC_ROUTES: set[str] = {
    "/health",
    "/ready",
    "/docs",
    "/openapi.json",
}


# -----------------------------------------------------------------------------
# AuthMiddleware — the class FastAPI registers to intercept every request
#
# WHY BaseHTTPMiddleware?
#   FastAPI is built on Starlette. BaseHTTPMiddleware is Starlette's standard
#   way to write middleware. It gives us a dispatch() method that receives
#   every request and a call_next function to pass the request forward.
#
# THE PATTERN:
#   async def dispatch(request, call_next):
#       # do something before the route handler
#       response = await call_next(request)  # run the route handler
#       # do something after the route handler
#       return response
#
#   If we return early (like when auth fails) call_next is never called
#   and the route handler never runs.
# -----------------------------------------------------------------------------
class AuthMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:

        # ---------------------------------------------------------------------
        # STEP 1: Skip auth for public routes
        # Health checks and docs must work without an API key.
        # We check the path before doing anything else.
        # ---------------------------------------------------------------------
        if request.url.path in PUBLIC_ROUTES:
            return await call_next(request)

        # ---------------------------------------------------------------------
        # STEP 2: Extract the Authorization header
        #
        # request.headers is a dictionary of all HTTP headers.
        # We use .get() so a missing header returns None instead of raising
        # a KeyError — we handle the None case explicitly below.
        # ---------------------------------------------------------------------
        auth_header = request.headers.get("Authorization")

        # No Authorization header at all — reject with 401 Unauthorized.
        # 401 means "I don't know who you are — please identify yourself."
        if not auth_header:
            logger.warning(
                "missing_auth_header",
                path=request.url.path,
                client_ip=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Authorization header is required. "
                               "Send your API key as: Authorization: Bearer <key>"
                }
            )

        # ---------------------------------------------------------------------
        # STEP 3: Parse the Bearer token format
        #
        # Expected format: "Bearer ng-abc123xyz"
        # We split on the first space and take the second part.
        # If the format is wrong (no space, wrong prefix), reject with 401.
        # ---------------------------------------------------------------------
        parts = auth_header.split(" ", 1)

        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning(
                "malformed_auth_header",
                path=request.url.path,
                header_received=auth_header[:20],  # log only first 20 chars for safety
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Authorization header must use Bearer format: "
                               "Authorization: Bearer <key>"
                }
            )

        api_key = parts[1].strip()

        # ---------------------------------------------------------------------
        # STEP 4: Validate the API key against our store
        #
        # We look up the key in VALID_API_KEYS.
        # If found, we get back the client identifier (e.g. "client_prod_a").
        # If not found, we reject with 403 Forbidden.
        # 403 means "I know who you are but you are not allowed in."
        #
        # WHY 403 AND NOT 401 FOR A WRONG KEY?
        #   401 = we cannot identify you (no credentials provided)
        #   403 = we identified you but you are not authorized (bad credentials)
        #   This distinction matters for clients debugging auth failures.
        # ---------------------------------------------------------------------
        client_id = VALID_API_KEYS.get(api_key)

        if not client_id:
            logger.warning(
                "invalid_api_key",
                path=request.url.path,
                key_prefix=api_key[:8] + "...",  # never log the full key
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "forbidden",
                    "message": "Invalid API key. Check your credentials."
                }
            )

        # ---------------------------------------------------------------------
        # STEP 5: Attach client identity to the request state
        #
        # request.state is a simple object we can attach arbitrary data to.
        # By storing client_id here, every downstream component — the rate
        # limiter, route handlers, metrics, logs — can read request.state.client_id
        # without re-parsing the auth header.
        #
        # We also record the auth timestamp so we can measure auth overhead
        # separately from total request latency in our observability layer.
        # ---------------------------------------------------------------------
        request.state.client_id = client_id
        request.state.authenticated_at = time.time()

        logger.info(
            "request_authenticated",
            client_id=client_id,
            path=request.url.path,
        )

        # ---------------------------------------------------------------------
        # STEP 6: Pass the request to the next middleware or route handler
        #
        # call_next(request) hands the request forward in the middleware chain.
        # Everything above this line runs BEFORE the route handler.
        # Everything below this line runs AFTER the route handler returns.
        # ---------------------------------------------------------------------
        response = await call_next(request)

        # Attach client_id to the response header so clients can confirm
        # which identity was used for this request — useful for debugging.
        response.headers["X-Client-ID"] = client_id

        return response