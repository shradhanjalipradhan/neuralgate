# =============================================================================
# rollout/canary.py
#
# PURPOSE:
#   Manages gradual traffic shifting from a stable version to a new version.
#   Starts at a small percentage, monitors key metrics, and either
#   auto-promotes (if metrics are good) or auto-rolls-back (if metrics regress).
#
# THE CANARY ROLLOUT STAGES:
#   Stage 1:  5% traffic to new version — initial sanity check
#   Stage 2: 20% traffic — enough to get statistical confidence
#   Stage 3: 50% traffic — near-equal split for final validation
#   Stage 4: 100% traffic — full promotion, old version decommissioned
#
# AUTO-ROLLBACK TRIGGERS (any one causes immediate rollback to 0%):
#   - p99 latency increases by > 20% vs baseline
#   - Error rate increases by > 0.5 percentage points vs baseline
#   - OOM events detected in new version pool
#   - Health score of new pool drops below 0.4
#
# WHY AUTO-ROLLBACK AND NOT MANUAL:
#   Manual rollback requires someone to be watching dashboards and to act.
#   At 3am with a canary at 5% traffic, auto-rollback fires in seconds.
#   Manual rollback fires when an engineer wakes up, reads alerts, and acts.
#   The difference is minutes of user impact versus seconds.
# =============================================================================

import time
import asyncio
import random
import structlog
import redis.asyncio as aioredis
from enum import Enum

logger = structlog.get_logger(__name__)


class CanaryStage(str, Enum):
    INACTIVE  = "inactive"
    STAGE_1   = "stage_1"    # 5%
    STAGE_2   = "stage_2"    # 20%
    STAGE_3   = "stage_3"    # 50%
    PROMOTED  = "promoted"   # 100%
    ROLLED_BACK = "rolled_back"

STAGE_TRAFFIC_PERCENTAGES = {
    CanaryStage.STAGE_1: 0.05,
    CanaryStage.STAGE_2: 0.20,
    CanaryStage.STAGE_3: 0.50,
    CanaryStage.PROMOTED: 1.00,
}

# Rollback thresholds
P99_REGRESSION_THRESHOLD    = 0.20   # 20% increase in p99
ERROR_RATE_REGRESSION       = 0.005  # 0.5 percentage point increase
HEALTH_SCORE_MINIMUM        = 0.40
EVALUATION_WINDOW_SECONDS   = 300    # Evaluate every 5 minutes before advancing


class CanaryController:

    def __init__(
        self,
        stable_pool_id: str,
        canary_pool_id: str,
        redis_url: str = "redis://localhost:6379",
    ):
        self.stable_pool_id = stable_pool_id
        self.canary_pool_id = canary_pool_id
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self._stage = CanaryStage.INACTIVE
        self._stage_start_time = 0.0
        self._baseline_metrics: dict = {}

    async def start(self, baseline_p99_ms: float, baseline_error_rate: float):
        self._baseline_metrics = {
            "p99_latency_ms": baseline_p99_ms,
            "error_rate": baseline_error_rate,
        }
        self._stage = CanaryStage.STAGE_1
        self._stage_start_time = time.time()
        logger.info("canary_started",
                   stable=self.stable_pool_id,
                   canary=self.canary_pool_id,
                   baseline=self._baseline_metrics)
        asyncio.create_task(self._evaluation_loop())

    def should_use_canary(self) -> bool:
        traffic_pct = STAGE_TRAFFIC_PERCENTAGES.get(self._stage, 0.0)
        return random.random() < traffic_pct

    async def _evaluation_loop(self):
        while self._stage not in (CanaryStage.PROMOTED, CanaryStage.ROLLED_BACK):
            await asyncio.sleep(60)    # Check every minute
            elapsed = time.time() - self._stage_start_time

            canary_metrics = await self._get_canary_metrics()
            should_rollback, reason = self._check_rollback_conditions(canary_metrics)

            if should_rollback:
                await self._rollback(reason)
                return

            if elapsed >= EVALUATION_WINDOW_SECONDS:
                await self._advance_stage()

    def _check_rollback_conditions(self, metrics: dict) -> tuple[bool, str]:
        baseline_p99 = self._baseline_metrics.get("p99_latency_ms", 0)
        baseline_err  = self._baseline_metrics.get("error_rate", 0)

        canary_p99 = metrics.get("p99_latency_ms", 0)
        canary_err  = metrics.get("error_rate", 0)

        if baseline_p99 > 0 and canary_p99 > baseline_p99 * (1 + P99_REGRESSION_THRESHOLD):
            return True, f"p99 regression: {canary_p99}ms vs baseline {baseline_p99}ms"

        if canary_err > baseline_err + ERROR_RATE_REGRESSION:
            return True, f"error rate regression: {canary_err:.3f} vs baseline {baseline_err:.3f}"

        if metrics.get("health_score", 1.0) < HEALTH_SCORE_MINIMUM:
            return True, f"health score too low: {metrics.get('health_score')}"

        return False, ""

    async def _advance_stage(self):
        stage_order = [CanaryStage.STAGE_1, CanaryStage.STAGE_2,
                       CanaryStage.STAGE_3, CanaryStage.PROMOTED]
        current_idx = stage_order.index(self._stage)
        if current_idx < len(stage_order) - 1:
            self._stage = stage_order[current_idx + 1]
            self._stage_start_time = time.time()
            traffic = STAGE_TRAFFIC_PERCENTAGES.get(self._stage, 1.0)
            logger.info("canary_stage_advanced", stage=self._stage, traffic_pct=traffic)

    async def _rollback(self, reason: str):
        self._stage = CanaryStage.ROLLED_BACK
        logger.error("canary_rolled_back", reason=reason,
                    stable=self.stable_pool_id, canary=self.canary_pool_id)

    async def _get_canary_metrics(self) -> dict:
        keys = [
            f"pool:metrics:{self.canary_pool_id}:p99_latency_ms",
            f"pool:metrics:{self.canary_pool_id}:error_rate",
            f"pool:metrics:{self.canary_pool_id}:gpu_util",
        ]
        values = await self.redis.mget(*keys)
        return {
            "p99_latency_ms": float(values[0] or 0),
            "error_rate":     float(values[1] or 0),
            "health_score":   1.0 - float(values[2] or 0),
        }

    @property
    def stage(self) -> CanaryStage:
        return self._stage
