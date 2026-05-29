# =============================================================================
# router/health_scorer.py
#
# PURPOSE:
#   Computes a real-time health score for each serving pool.
#   The router uses this score to rank candidate pools and decide when to
#   trigger early fallback before a pool fully fails.
#
# WHY A HEALTH SCORE INSTEAD OF BINARY HEALTHY/UNHEALTHY?
#   Binary health checks miss early degradation. A pool can be technically
#   "up" while suffering from rising queue depth, memory pressure, or
#   increasing tail latency — all signs that failure is approaching.
#   A weighted score captures these early signals and lets the router shift
#   traffic proactively instead of only reacting to outright failure.
#
# SCORE RANGE: 0.0 (completely unhealthy) to 1.0 (perfectly healthy)
# FALLBACK THRESHOLD: below 0.3 the router treats this pool as unavailable
# =============================================================================

import time
import structlog
import redis.asyncio as aioredis

logger = structlog.get_logger(__name__)

# Health score weights — must sum to 1.0
# These weights encode our priorities:
#   Error rate matters most — errors directly impact users
#   Queue depth matters second — it predicts future latency
#   Tail latency matters third — it measures current user experience
#   GPU utilization matters least — high utilization is fine if latency is ok
WEIGHTS = {
    "error_rate":    0.40,
    "queue_depth":   0.30,
    "p99_latency":   0.20,
    "gpu_util":      0.10,
}

# Thresholds for each signal — above these values the score starts degrading
THRESHOLDS = {
    "error_rate_critical":  0.10,   # 10% errors = score of 0
    "queue_depth_critical": 100,    # 100 queued requests = score of 0
    "p99_latency_critical": 5000,   # 5000ms p99 = score of 0
    "gpu_util_critical":    0.98,   # 98% GPU util = score of 0
}

FALLBACK_THRESHOLD = 0.3   # Below this score, stop routing to this pool


class PoolHealthScorer:

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)

    async def get_score(self, pool_id: str) -> float:
        # ---------------------------------------------------------------------
        # Read current metrics for this pool from Redis.
        # Each serving pool worker publishes its metrics to Redis keys
        # at regular intervals (every 5 seconds in production).
        # If a key is missing (pool just started) we use safe defaults.
        # ---------------------------------------------------------------------
        try:
            keys = [
                f"pool:metrics:{pool_id}:error_rate",
                f"pool:metrics:{pool_id}:queue_depth",
                f"pool:metrics:{pool_id}:p99_latency_ms",
                f"pool:metrics:{pool_id}:gpu_util",
                f"pool:metrics:{pool_id}:last_updated",
            ]
            values = await self.redis.mget(*keys)

            error_rate   = float(values[0]) if values[0] else 0.0
            queue_depth  = float(values[1]) if values[1] else 0.0
            p99_latency  = float(values[2]) if values[2] else 0.0
            gpu_util     = float(values[3]) if values[3] else 0.0
            last_updated = float(values[4]) if values[4] else 0.0

        except Exception as e:
            logger.error("health_score_redis_error", pool_id=pool_id, error=str(e))
            # Redis failure — return a mid-range score, not zero.
            # Zero would pull this pool out of rotation unnecessarily.
            return 0.5

        # ---------------------------------------------------------------------
        # STALENESS CHECK
        # If the pool has not published metrics in the last 30 seconds,
        # something is wrong — it may have crashed or lost connectivity.
        # Return a low score to reduce traffic to this pool.
        # ---------------------------------------------------------------------
        if last_updated > 0 and (time.time() - last_updated) > 30:
            logger.warning("pool_metrics_stale", pool_id=pool_id,
                          seconds_since_update=time.time() - last_updated)
            return 0.1

        # ---------------------------------------------------------------------
        # SCORE EACH SIGNAL INDEPENDENTLY — each returns 0.0 to 1.0
        # Then combine using weights.
        # ---------------------------------------------------------------------
        error_score   = self._score_error_rate(error_rate)
        queue_score   = self._score_queue_depth(queue_depth)
        latency_score = self._score_p99_latency(p99_latency)
        gpu_score     = self._score_gpu_util(gpu_util)

        weighted_score = (
            error_score   * WEIGHTS["error_rate"] +
            queue_score   * WEIGHTS["queue_depth"] +
            latency_score * WEIGHTS["p99_latency"] +
            gpu_score     * WEIGHTS["gpu_util"]
        )

        logger.info(
            "pool_health_score",
            pool_id=pool_id,
            score=round(weighted_score, 3),
            error_rate=error_rate,
            queue_depth=queue_depth,
            p99_latency_ms=p99_latency,
            gpu_util=gpu_util,
        )

        return round(weighted_score, 3)

    def _score_error_rate(self, rate: float) -> float:
        # 0% errors = 1.0, 10%+ errors = 0.0, linear between
        critical = THRESHOLDS["error_rate_critical"]
        if rate >= critical:
            return 0.0
        return 1.0 - (rate / critical)

    def _score_queue_depth(self, depth: float) -> float:
        # 0 queued = 1.0, 100+ queued = 0.0, linear between
        critical = THRESHOLDS["queue_depth_critical"]
        if depth >= critical:
            return 0.0
        return 1.0 - (depth / critical)

    def _score_p99_latency(self, latency_ms: float) -> float:
        # 0ms = 1.0 (unrealistic but math works), 5000ms+ = 0.0
        critical = THRESHOLDS["p99_latency_critical"]
        if latency_ms >= critical:
            return 0.0
        return 1.0 - (latency_ms / critical)

    def _score_gpu_util(self, util: float) -> float:
        # Under 85% GPU util = perfect score. 98%+ = 0.0
        # We want high GPU utilization (it means we are using hardware well)
        # but not so high that there is no headroom for traffic spikes.
        if util < 0.85:
            return 1.0
        critical = THRESHOLDS["gpu_util_critical"]
        if util >= critical:
            return 0.0
        return 1.0 - ((util - 0.85) / (critical - 0.85))

    def is_routable(self, score: float) -> bool:
        return score >= FALLBACK_THRESHOLD

    async def publish_mock_metrics(self, pool_id: str, healthy: bool = True):
        # -----------------------------------------------------------------------
        # DEVELOPMENT HELPER — publishes fake healthy metrics for a pool.
        # Used during local development and testing so the health scorer
        # returns realistic scores without real GPU workers running.
        # Remove or gate behind an env flag before production deployment.
        # -----------------------------------------------------------------------
        metrics = {
            f"pool:metrics:{pool_id}:error_rate":    "0.01" if healthy else "0.25",
            f"pool:metrics:{pool_id}:queue_depth":   "5"    if healthy else "90",
            f"pool:metrics:{pool_id}:p99_latency_ms": "450" if healthy else "4800",
            f"pool:metrics:{pool_id}:gpu_util":      "0.72" if healthy else "0.97",
            f"pool:metrics:{pool_id}:last_updated":  str(time.time()),
        }
        await self.redis.mset(metrics)
        for key in metrics:
            await self.redis.expire(key, 60)
