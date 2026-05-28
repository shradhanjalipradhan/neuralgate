# =============================================================================
# edge/middleware/rate_limiter.py
#
# PURPOSE:
#   Enforces per-client request rate limits using a sliding window counter
#   stored in Redis. Even authenticated clients are rejected here if they
#   exceed their allowed request rate.
#
# WHY RATE LIMITING?
#   Without this, one client sending 10000 requests per second would:
#     - Monopolize all GPU capacity
#     - Starve every other client
#     - Drive infrastructure costs to unpredictable levels
#     - Potentially destabilize the serving pools through queue saturation
#
# HOW THE SLIDING WINDOW WORKS:
#   For each client we maintain a Redis sorted set where:
#     - Each member is a unique request ID
#     - Each score is the timestamp of that request in seconds
#   On every request we:
#     1. Remove entries older than (now - window_seconds) — the expired ones
#     2. Count remaining entries — this is requests in the last window_seconds
#     3. If count >= limit: reject with 429 and tell client when to retry
#     4. If count < limit: add this request and allow it through
#
# WHY SORTED SET AND NOT A SIMPLE COUNTER?
#   A simple counter (INCR in Redis) only supports fixed windows — it resets
#   at clock boundaries. A sorted set scored by timestamp lets us query
#   "how many requests in the last N seconds from right now" — a true
#   sliding window that prevents the boundary-gaming problem.
#
# WHY REDIS AND NOT IN-MEMORY?
#   If rate limit state lived in Python memory, each server instance would
#   have its own counter. A client hitting two different instances could
#   send 2x the limit. Redis is shared across all instances — one counter
#   per client regardless of which server handles each request.
#
# DO YOU NEED TO RUN THIS FILE? NO.
#   Activated when server.py registers it. Requires Redis to be running
#   when the full server starts. No standalone execution needed.
# =============================================================================

import time
import uuid
import structlog
import redis.asyncio as aioredis
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
# RATE LIMIT TIERS — different clients get different limits
#
# We define limits per priority tier because our business model charges
# differently for different service levels. A client paying for HIGH priority
# gets more requests per window than a LOW priority client.
#
# Structure: { "client_id_prefix_or_exact": (requests_allowed, window_seconds) }
#
# In production this would be looked up from a database per client.
# We use a tiered default system here for clarity.
# -----------------------------------------------------------------------------
RATE_LIMIT_TIERS: dict[str, tuple[int, int]] = {
    "client_dev":    (20,  60),   # 20 requests per 60 seconds — dev testing
    "client_test":   (50,  60),   # 50 requests per 60 seconds — test env
    "client_prod_a": (200, 60),   # 200 requests per 60 seconds — prod tier A
    "client_prod_b": (500, 60),   # 500 requests per 60 seconds — prod tier B
    "default":       (30,  60),   # fallback for any unknown client_id
}

# Routes that bypass rate limiting.
# Same public routes as auth — health checks must always respond.
RATE_LIMIT_EXEMPT: set[str] = {
    "/health",
    "/ready",
    "/docs",
    "/openapi.json",
}


# -----------------------------------------------------------------------------
# RateLimiterMiddleware
#
# Reads client_id from request.state (set by AuthMiddleware before this runs).
# Uses Redis sorted sets to implement a per-client sliding window counter.
# Rejects over-limit requests with 429 before they reach route handlers.
# -----------------------------------------------------------------------------
class RateLimiterMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: ASGIApp, redis_url: str = "redis://localhost:6379"):
        super().__init__(app)

        # ---------------------------------------------------------------------
        # Redis connection pool
        #
        # We use redis.asyncio (async Redis client) so Redis calls do not block
        # the event loop. A blocking Redis call would freeze request handling
        # for every other concurrent request during that call.
        #
        # decode_responses=True means Redis returns Python strings instead of
        # raw bytes — cleaner to work with throughout.
        # ---------------------------------------------------------------------
        self.redis = aioredis.from_url(
            redis_url,
            decode_responses=True,
            max_connections=20,     # pool of 20 connections shared across requests
        )

    async def dispatch(self, request: Request, call_next) -> Response:

        # ---------------------------------------------------------------------
        # STEP 1: Skip rate limiting for exempt routes
        # ---------------------------------------------------------------------
        if request.url.path in RATE_LIMIT_EXEMPT:
            return await call_next(request)

        # ---------------------------------------------------------------------
        # STEP 2: Get client_id from request state
        #
        # AuthMiddleware ran before this and set request.state.client_id.
        # If for some reason it is missing (auth misconfiguration), we use
        # the client IP as a fallback identifier rather than crashing.
        # ---------------------------------------------------------------------
        client_id = getattr(request.state, "client_id", None)
        if not client_id:
            client_id = request.client.host if request.client else "unknown"

        # ---------------------------------------------------------------------
        # STEP 3: Look up this client's rate limit tier
        #
        # We check RATE_LIMIT_TIERS for their exact client_id.
        # If not found we fall back to the "default" tier.
        # This means new clients are rate limited conservatively until
        # we explicitly configure their tier.
        # ---------------------------------------------------------------------
        limit, window_seconds = RATE_LIMIT_TIERS.get(
            client_id,
            RATE_LIMIT_TIERS["default"]
        )

        # ---------------------------------------------------------------------
        # STEP 4: Run the sliding window check in Redis
        #
        # Redis key: "ratelimit:{client_id}"
        # Each key holds a sorted set of request timestamps for that client.
        # ---------------------------------------------------------------------
        redis_key = f"ratelimit:{client_id}"
        now = time.time()
        window_start = now - window_seconds

        try:
            # -----------------------------------------------------------------
            # These four Redis operations must happen atomically — if two
            # requests from the same client arrive at exactly the same
            # millisecond, we cannot let both of them read "count=99" and
            # both decide they are under the limit and both add themselves.
            # That would allow 2 requests through when only 1 should pass.
            #
            # We use a Redis pipeline with transaction=True to group all four
            # operations into one atomic block. Redis executes the whole block
            # before processing any other command for this key.
            # -----------------------------------------------------------------
            async with self.redis.pipeline(transaction=True) as pipe:

                # Remove all entries older than the start of our window.
                # ZREMRANGEBYSCORE removes sorted set members with score
                # (timestamp) between -infinity and window_start.
                # After this, only requests from the last window_seconds remain.
                pipe.zremrangebyscore(redis_key, "-inf", window_start)

                # Count how many requests remain in the window.
                # ZCARD returns the cardinality (count) of the sorted set.
                pipe.zcard(redis_key)

                # Add this request to the sorted set.
                # Member = unique request ID, Score = current timestamp.
                # We add it before checking the count so the atomic block
                # includes the addition — we remove it conceptually if over limit.
                request_member = str(uuid.uuid4())
                pipe.zadd(redis_key, {request_member: now})

                # Set the key to expire after window_seconds of inactivity.
                # Without this, Redis would accumulate keys for every client
                # forever, growing unbounded. The TTL ensures cleanup.
                pipe.expire(redis_key, window_seconds * 2)

                # Execute all four commands atomically
                results = await pipe.execute()

            # results[1] is the ZCARD result — count before we added this request
            current_count = results[1]

        except Exception as e:
            # -----------------------------------------------------------------
            # REDIS FAILURE FALLBACK
            #
            # If Redis is down, we have a choice: block all requests (safe but
            # causes an outage) or allow all requests (risky but keeps serving).
            #
            # We choose to ALLOW requests when Redis is unavailable because
            # the inference engine itself is still healthy. We log the failure
            # loudly so on-call engineers know rate limiting is degraded.
            # This is a deliberate reliability tradeoff — availability over
            # perfect rate limit enforcement.
            # -----------------------------------------------------------------
            logger.error(
                "rate_limiter_redis_failure",
                client_id=client_id,
                error=str(e),
                action="allowing_request_in_degraded_mode",
            )
            return await call_next(request)

        # ---------------------------------------------------------------------
        # STEP 5: Decision — allow or reject
        #
        # current_count is the number of requests BEFORE this one in the window.
        # If it is already at or above the limit, this request pushes us over.
        # ---------------------------------------------------------------------
        remaining = max(0, limit - current_count - 1)
        retry_after = int(window_seconds - (now - window_start))

        if current_count >= limit:
            logger.warning(
                "rate_limit_exceeded",
                client_id=client_id,
                count=current_count,
                limit=limit,
                window_seconds=window_seconds,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=429,
                headers={
                    # Retry-After tells the client how many seconds to wait.
                    # This is part of the HTTP 429 contract — well-behaved
                    # clients will read this and back off instead of hammering.
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + retry_after)),
                },
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"You have exceeded {limit} requests per "
                               f"{window_seconds} seconds.",
                    "retry_after_seconds": retry_after,
                }
            )

        # ---------------------------------------------------------------------
        # STEP 6: Request is within limit — pass it through
        #
        # We attach rate limit headers to the response so clients can track
        # how close they are to their limit without waiting for a 429.
        # Good API design: tell clients their quota proactively.
        # ---------------------------------------------------------------------
        response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + window_seconds))

        logger.info(
            "rate_limit_check_passed",
            client_id=client_id,
            count=current_count + 1,
            limit=limit,
            remaining=remaining,
        )

        return response

    async def close(self):
        # Clean up Redis connection pool when the server shuts down.
        await self.redis.aclose()