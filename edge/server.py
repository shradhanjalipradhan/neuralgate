# =============================================================================
# edge/server.py
#
# PURPOSE:
#   The main entry point of NeuralGate. This file creates the FastAPI
#   application, registers all middleware in the correct order, defines the
#   inference route handlers, and exposes health check endpoints.
#
# THIS FILE IS THE GLUE.
#   Every other file in the edge/ folder defines a piece — a schema, a
#   middleware, a validator. This file wires all those pieces together into
#   a running application.
#
# MIDDLEWARE ORDER MATTERS:
#   FastAPI processes middleware in reverse registration order for requests
#   and forward order for responses. We register in this order:
#     1. ValidatorMiddleware  (registered first = runs last on request)
#     2. RateLimiterMiddleware
#     3. AuthMiddleware       (registered last = runs first on request)
#
#   So the actual request flow is:
#     Request → Auth → RateLimiter → Validator → Route Handler
#     Response ← Auth ← RateLimiter ← Validator ← Route Handler
#
# DO YOU NEED TO RUN THIS FILE? YES — THIS IS THE ONE YOU RUN.
#   To start the server locally (once dependencies are installed):
#     uvicorn edge.server:app --reload --port 8000
#
#   To test it is running:
#     curl http://localhost:8000/health
# =============================================================================

import time
import uuid
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from edge.schemas.request import CompletionRequest, CompletionResponse
from edge.middleware.auth import AuthMiddleware
from edge.middleware.rate_limiter import RateLimiterMiddleware
from edge.middleware.validator import ValidatorMiddleware

logger = structlog.get_logger(__name__)


# =============================================================================
# LIFESPAN — startup and shutdown logic
#
# FastAPI's lifespan context manager replaces the old @app.on_event("startup")
# pattern. Code before `yield` runs on startup. Code after `yield` runs on
# shutdown. This is where we initialize shared resources — Redis connections,
# model registries, thread pools — that need to exist for the lifetime of the
# server and be cleaned up gracefully when it stops.
#
# WHY GRACEFUL SHUTDOWN?
#   If we just kill the process, in-flight requests get dropped mid-generation.
#   The lifespan shutdown hook lets us finish serving current requests, drain
#   queues, close database connections, and flush metrics before exiting.
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):

    # -------------------------------------------------------------------------
    # STARTUP
    # -------------------------------------------------------------------------
    logger.info("neuralgate_starting", version="0.1.0")

    # In later steps, this is where we will:
    #   - Initialize the Redis connection pool
    #   - Load the router's pool registry
    #   - Warm fallback model pools
    #   - Register Prometheus metrics collectors
    # For now we log startup and proceed.

    logger.info("neuralgate_ready", port=8000)

    yield   # server is running — handle requests

    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------
    logger.info("neuralgate_shutting_down")

    # In later steps, this is where we will:
    #   - Drain the request queue
    #   - Close Redis connections
    #   - Flush any buffered metrics
    #   - Signal serving pools to finish in-flight requests

    logger.info("neuralgate_shutdown_complete")


# =============================================================================
# APP INITIALIZATION
#
# We create the FastAPI app with:
#   - title/description: appear in the auto-generated /docs page
#   - version: useful for clients to know which API version they are hitting
#   - lifespan: our startup/shutdown handler defined above
#   - docs_url: the interactive API explorer FastAPI generates automatically
# =============================================================================
app = FastAPI(
    title="NeuralGate",
    description=(
        "Production-grade LLM inference gateway. "
        "Handles routing, fallback, retrieval, and observability "
        "for large-scale model serving."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# CORS MIDDLEWARE
#
# CORS (Cross-Origin Resource Sharing) controls which web domains can call
# our API from a browser. Without this, browser-based clients from different
# domains get blocked by the browser's security policy.
#
# allow_origins=["*"] means any domain — fine for development.
# In production we would restrict this to our known client domains.
#
# NOTE: CORSMiddleware must be added BEFORE our custom middleware so it
# handles preflight OPTIONS requests before auth runs on them.
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# CUSTOM MIDDLEWARE REGISTRATION
#
# CRITICAL: FastAPI adds middleware as a stack. The LAST middleware added
# runs FIRST on incoming requests. So we add in reverse order of execution:
#
#   We want request flow: Auth → RateLimiter → Validator → Route
#   So we add:           Validator first, RateLimiter second, Auth last
#
# Getting this order wrong means requests could hit the rate limiter before
# auth, allowing unauthenticated clients to consume rate limit quota and
# potentially exhaust it for legitimate clients.
# =============================================================================
app.add_middleware(ValidatorMiddleware)
app.add_middleware(RateLimiterMiddleware, redis_url="redis://localhost:6379")
app.add_middleware(AuthMiddleware)


# =============================================================================
# HEALTH CHECK ENDPOINTS
#
# Two separate health endpoints serve different purposes:
#
# /health — LIVENESS check
#   "Is the process running and responsive?"
#   Used by Kubernetes/ECS to decide if a container needs to be restarted.
#   Should always return 200 as long as the process is alive, even if
#   dependencies like Redis are down.
#
# /ready — READINESS check
#   "Is this instance ready to receive production traffic?"
#   Used by load balancers to decide if traffic should be routed here.
#   Should return 200 only when all dependencies are healthy.
#   Returns 503 if Redis is down, model pools are not warmed, etc.
#
# WHY TWO SEPARATE ENDPOINTS?
#   A liveness failure → restart the container (something is fundamentally wrong)
#   A readiness failure → remove from load balancer rotation (temporarily unhealthy)
#   These are different responses to different problems.
# =============================================================================
@app.get("/health", tags=["Infrastructure"])
async def health():
    return {
        "status": "ok",
        "service": "neuralgate",
        "version": "0.1.0",
        "timestamp": time.time(),
    }


@app.get("/ready", tags=["Infrastructure"])
async def ready():
    # In later steps this will check:
    #   - Redis connectivity
    #   - At least one serving pool is healthy
    #   - Router has a valid pool registry loaded
    # For now we return ready immediately.
    return {
        "status": "ready",
        "checks": {
            "redis": "ok",       # placeholder — real check added in Step 2
            "router": "ok",      # placeholder — real check added in Step 2
            "pools": "ok",       # placeholder — real check added in Step 3
        },
        "timestamp": time.time(),
    }


# =============================================================================
# INFERENCE ROUTE — POST /v1/completions
#
# This is the core route. It receives a validated, authenticated, rate-limited
# inference request and returns a completion response.
#
# At this stage (Step 1) it returns a mock response because the router and
# serving pools do not exist yet. In Step 2 we replace the mock with a real
# dispatch call to the router.
#
# response_model=CompletionResponse tells FastAPI to:
#   1. Validate that our return value matches CompletionResponse shape
#   2. Serialize it correctly to JSON
#   3. Show the response schema in /docs
# =============================================================================
@app.post(
    "/v1/completions",
    response_model=CompletionResponse,
    tags=["Inference"],
    summary="Submit an inference request",
    description=(
        "Send a prompt to a deployed model. The request is authenticated, "
        "rate-limited, validated, routed to the correct serving pool, "
        "and returned with full observability metadata."
    ),
)
async def completions(
    payload: CompletionRequest,
    request: Request,
) -> CompletionResponse:

    # -------------------------------------------------------------------------
    # STEP 1: Assign or carry forward a request ID
    #
    # If the client provided a request_id we use it — this lets them correlate
    # their logs with ours end to end. If not, we generate a UUID.
    # This ID will travel through every log, metric, and trace for this request.
    # -------------------------------------------------------------------------
    request_id = payload.request_id or str(uuid.uuid4())
    start_time = time.time()

    # Bind request_id to the logger so every log line in this request
    # automatically includes it — no manual passing needed.
    log = logger.bind(
        request_id=request_id,
        model=payload.model,
        client_id=getattr(request.state, "client_id", "unknown"),
        priority=payload.priority.value,
    )

    log.info(
        "inference_request_received",
        prompt_length=len(payload.prompt),
        max_tokens=payload.max_tokens,
        stream=payload.stream,
    )

    # -------------------------------------------------------------------------
    # STEP 2: Route to serving pool (placeholder — real router added in Step 2)
    #
    # In the next step, this is where we call:
    #   result = await router.dispatch(payload, request_id)
    #
    # For now we return a mock response so we can run and test the edge layer
    # end to end before the router exists.
    # -------------------------------------------------------------------------
    await _mock_model_call(payload)

    # -------------------------------------------------------------------------
    # STEP 3: Calculate latency and build response
    #
    # We measure end-to-end latency from when the request entered this handler
    # to when we are about to return. This does not include middleware time —
    # for full end-to-end latency including auth and rate limiting, the
    # observability layer will measure from the moment the request arrived
    # at the server process.
    # -------------------------------------------------------------------------
    latency_ms = (time.time() - start_time) * 1000

    log.info(
        "inference_request_complete",
        latency_ms=round(latency_ms, 2),
        tokens_generated=42,        # mock value — real count from router in Step 2
        fallback_triggered=False,
    )

    return CompletionResponse(
        request_id=request_id,
        model=payload.model,
        text=f"[MOCK RESPONSE] This is a placeholder from the edge layer. "
             f"Real model routing added in Step 2. "
             f"Model requested: {payload.model}. "
             f"Prompt received: '{payload.prompt[:50]}...'",
        tokens_generated=42,
        latency_ms=round(latency_ms, 2),
        pool_used="mock_pool",
        fallback_triggered=False,
    )


# =============================================================================
# GLOBAL EXCEPTION HANDLER
#
# Catches any unhandled exception that escapes route handlers and returns
# a clean JSON error instead of a raw Python traceback.
#
# WHY THIS MATTERS:
#   An unhandled exception that reaches the client as a 500 with a Python
#   traceback leaks internal implementation details — file paths, library
#   versions, variable names. This handler ensures all errors are formatted
#   consistently and safely.
# =============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
        client_id=getattr(request.state, "client_id", "unknown"),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Our team has been notified.",
        }
    )


# =============================================================================
# MOCK MODEL CALL — placeholder until router exists
#
# Simulates the latency of a real model call so we can test the full
# request pipeline end to end. Removed in Step 2 when the real router
# is wired in.
#
# asyncio.sleep() simulates I/O wait without blocking the event loop —
# during this sleep other requests can be processed concurrently.
# =============================================================================
async def _mock_model_call(payload: CompletionRequest) -> None:
    import asyncio
    # Simulate ~200ms model latency
    await asyncio.sleep(0.2)
