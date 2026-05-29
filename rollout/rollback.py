# =============================================================================
# rollout/rollback.py
#
# PURPOSE:
#   Manages the rollback process when a canary or full deployment regresses.
#   Coordinates traffic shift, circuit breaker state reset, and notification.
#
# WHAT A ROLLBACK INVOLVES:
#   1. Immediately set canary traffic percentage to 0%
#   2. Mark canary pool as UNHEALTHY in the pool registry
#   3. Open circuit breaker for canary pool (stop all calls)
#   4. Log rollback event with full context for post-incident review
#   5. Notify on-call via alert system
#   6. Optionally: drain in-flight canary requests gracefully
# =============================================================================

import time
import structlog
import redis.asyncio as aioredis

logger = structlog.get_logger(__name__)


class RollbackManager:

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self._rollback_history: list[dict] = []

    async def execute_rollback(
        self,
        canary_pool_id: str,
        stable_pool_id: str,
        reason: str,
        triggered_by: str = "auto",     # "auto" or "manual"
    ):
        rollback_id = f"rollback-{int(time.time())}"
        log = logger.bind(rollback_id=rollback_id, canary=canary_pool_id, stable=stable_pool_id)

        log.error("rollback_initiated", reason=reason, triggered_by=triggered_by)

        # Step 1: Zero out canary traffic weight in Redis
        await self.redis.set(f"canary:traffic_weight:{canary_pool_id}", "0")

        # Step 2: Mark canary pool as unhealthy
        await self.redis.set(f"pool:status:{canary_pool_id}", "unhealthy", ex=3600)

        # Step 3: Ensure stable pool is marked healthy
        await self.redis.set(f"pool:status:{stable_pool_id}", "healthy", ex=3600)

        # Step 4: Record rollback event for post-incident review
        event = {
            "rollback_id": rollback_id,
            "timestamp": time.time(),
            "canary_pool_id": canary_pool_id,
            "stable_pool_id": stable_pool_id,
            "reason": reason,
            "triggered_by": triggered_by,
        }
        self._rollback_history.append(event)
        await self.redis.lpush("rollback:history", str(event))
        await self.redis.ltrim("rollback:history", 0, 99)    # Keep last 100 rollbacks

        log.error("rollback_complete", **event)
        return rollback_id

    def get_rollback_history(self) -> list[dict]:
        return self._rollback_history
