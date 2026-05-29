# =============================================================================
# rollout/shadow_mode.py
#
# PURPOSE:
#   Routes a copy of live traffic to a new model version without affecting
#   the response the user receives. The primary version serves the user.
#   The shadow version processes the same request silently in the background.
#   We compare outputs to validate the new version before promoting it.
#
# WHY SHADOW MODE BEFORE CANARY:
#   A canary exposes real users to the new version (even if a small %).
#   Shadow mode has zero user impact — the shadow response is never returned.
#   This lets us validate the new model on real production traffic distributions
#   before any user sees it. Only when shadow results look good do we canary.
#
# WHAT WE COMPARE IN SHADOW MODE:
#   - Latency difference (is the new version slower?)
#   - Token count difference (is it generating much more or less?)
#   - Error rate (does it fail on inputs the primary handles fine?)
#   - Output quality (requires human eval or automated scoring)
#
# WHAT WE DO NOT COMPARE:
#   - Exact output text (LLMs are non-deterministic — outputs will differ)
#   - We compare distributions and error rates, not individual outputs
# =============================================================================

import asyncio
import time
import structlog
from typing import Callable, Awaitable, Any

logger = structlog.get_logger(__name__)


class ShadowModeRouter:

    def __init__(self, shadow_pool_id: str, shadow_sample_rate: float = 1.0):
        # shadow_sample_rate: 1.0 = shadow all requests, 0.1 = shadow 10%
        # Start at 1.0 to collect maximum data, reduce if shadow adds overhead
        self.shadow_pool_id = shadow_pool_id
        self.shadow_sample_rate = shadow_sample_rate
        self._shadow_results: list[dict] = []    # stored for comparison analysis

    async def execute(
        self,
        request_data: dict,
        primary_fn: Callable[[], Awaitable[Any]],
        shadow_fn: Callable[[], Awaitable[Any]],
        request_id: str,
    ) -> Any:
        # ---------------------------------------------------------------------
        # Always run primary and return its result to the user.
        # Fire shadow as a background task — never await it in the critical path.
        # The user never waits for shadow to complete.
        # ---------------------------------------------------------------------
        import random
        log = logger.bind(request_id=request_id, shadow_pool=self.shadow_pool_id)

        # Run primary — this is what the user gets
        primary_start = time.time()
        primary_result = await primary_fn()
        primary_latency = (time.time() - primary_start) * 1000

        # Decide whether to shadow this request based on sample rate
        if random.random() < self.shadow_sample_rate:
            # Fire shadow as background task — does not block response
            asyncio.create_task(
                self._run_shadow(shadow_fn, primary_result, primary_latency, request_id, log)
            )

        return primary_result

    async def _run_shadow(self, shadow_fn, primary_result, primary_latency_ms, request_id, log):
        try:
            shadow_start = time.time()
            shadow_result = await shadow_fn()
            shadow_latency = (time.time() - shadow_start) * 1000

            comparison = {
                "request_id": request_id,
                "primary_latency_ms": round(primary_latency_ms, 1),
                "shadow_latency_ms": round(shadow_latency, 1),
                "latency_delta_ms": round(shadow_latency - primary_latency_ms, 1),
                "primary_tokens": getattr(primary_result, "tokens_generated", 0),
                "shadow_tokens": getattr(shadow_result, "tokens_generated", 0),
                "shadow_error": None,
            }
            self._shadow_results.append(comparison)

            log.info("shadow_request_complete", **comparison)

        except Exception as e:
            log.error("shadow_request_failed", error=str(e), request_id=request_id)
            self._shadow_results.append({
                "request_id": request_id,
                "shadow_error": str(e),
                "primary_latency_ms": round(primary_latency_ms, 1),
            })

    def get_shadow_stats(self) -> dict:
        if not self._shadow_results:
            return {"shadow_count": 0}

        successful = [r for r in self._shadow_results if r.get("shadow_error") is None]
        error_count = len(self._shadow_results) - len(successful)
        avg_latency_delta = (
            sum(r["latency_delta_ms"] for r in successful) / len(successful)
            if successful else 0
        )
        return {
            "shadow_count": len(self._shadow_results),
            "error_count": error_count,
            "error_rate": round(error_count / len(self._shadow_results), 3),
            "avg_latency_delta_ms": round(avg_latency_delta, 1),
        }
