# =============================================================================
# router/dispatcher.py
#
# PURPOSE:
#   The brain of the routing layer. Takes a classified request, scores all
#   candidate pools, selects the best healthy pool, and dispatches the request.
#   If no healthy pool is available it hands off to the fallback engine.
#
# WHAT THE DISPATCHER DOES IN ORDER:
#   1. Ask the classifier which pool TYPE this request needs
#   2. Get all pools of that type from the registry
#   3. Score each candidate pool using the health scorer
#   4. Pick the highest-scoring routable pool
#   5. If no pool is routable, trigger fallback
#   6. Dispatch the request to the selected pool endpoint
#
# WHY CENTRALIZED DISPATCHING?
#   If routing decisions were made by individual clients or distributed across
#   services, we would get inconsistent behavior — some clients retrying on
#   already-overloaded pools, others not knowing a healthier pool exists.
#   The central dispatcher has a complete view of all pool health and makes
#   consistent decisions for all traffic.
# =============================================================================

import time
import httpx
import structlog
from edge.schemas.request import CompletionRequest, CompletionResponse
from router.classifier import RequestClassifier
from router.health_scorer import PoolHealthScorer
from router.pool_registry import PoolRegistry, PoolType

logger = structlog.get_logger(__name__)

# How long to wait for a pool to respond before considering it too slow.
# This is the per-request timeout — if the pool does not start responding
# within this window we abort and try the next pool or trigger fallback.
DISPATCH_TIMEOUT_SECONDS = 30.0


class RouterDispatcher:

    def __init__(
        self,
        registry: PoolRegistry,
        classifier: RequestClassifier,
        health_scorer: PoolHealthScorer,
    ):
        self.registry = registry
        self.classifier = classifier
        self.health_scorer = health_scorer
        # Shared async HTTP client — reusing one client across requests
        # is much more efficient than creating a new connection per request.
        self._http_client = httpx.AsyncClient(timeout=DISPATCH_TIMEOUT_SECONDS)

    async def dispatch(
        self,
        request: CompletionRequest,
        request_id: str,
    ) -> CompletionResponse:
        # ---------------------------------------------------------------------
        # STEP 1: Classify the request into a pool type
        # ---------------------------------------------------------------------
        pool_type = self.classifier.classify(request)
        log = logger.bind(request_id=request_id, model=request.model, pool_type=pool_type)
        log.info("dispatch_started")

        # ---------------------------------------------------------------------
        # STEP 2: Get all candidate pools of the required type for this model
        # We need pools that both match the type AND support the requested model.
        # ---------------------------------------------------------------------
        all_candidates = self.registry.get_pools_by_type(pool_type)
        model_candidates = [p for p in all_candidates if request.model in p.models]

        if not model_candidates:
            log.error("no_pools_for_model", available_types=[p.pool_type for p in all_candidates])
            raise RuntimeError(f"No pool found for model '{request.model}' in pool type '{pool_type}'")

        # ---------------------------------------------------------------------
        # STEP 3: Score each candidate pool and rank by health score
        # We score all candidates in parallel conceptually — in practice we
        # score them sequentially here for simplicity. In production this
        # would use asyncio.gather() to score all pools simultaneously.
        # ---------------------------------------------------------------------
        scored_pools = []
        for pool in model_candidates:
            score = await self.health_scorer.get_score(pool.pool_id)
            scored_pools.append((score, pool))
            log.info("pool_scored", pool_id=pool.pool_id, score=score)

        # Sort by score descending — highest score (healthiest) first
        scored_pools.sort(key=lambda x: x[0], reverse=True)

        # ---------------------------------------------------------------------
        # STEP 4: Attempt dispatch to the best routable pool
        # We try pools in order of health score. If one fails we try the next.
        # This is NOT infinite retry — we only try each pool once.
        # We stop after finding one that works or exhausting all candidates.
        # ---------------------------------------------------------------------
        for score, pool in scored_pools:
            if not self.health_scorer.is_routable(score):
                log.warning("pool_not_routable", pool_id=pool.pool_id, score=score)
                continue

            try:
                log.info("dispatching_to_pool", pool_id=pool.pool_id, score=score)
                response = await self._call_pool(pool.endpoint, request, request_id)
                response.pool_used = pool.pool_id
                return response

            except Exception as e:
                log.warning("pool_dispatch_failed", pool_id=pool.pool_id, error=str(e))
                # Mark this pool as having issues — update its status in the registry
                continue

        # ---------------------------------------------------------------------
        # STEP 5: All candidate pools failed or were unroutable
        # Hand off to the fallback engine (implemented in Step 4).
        # For now we raise an exception — the fallback engine will catch this.
        # ---------------------------------------------------------------------
        log.error("all_pools_exhausted", candidates=[p.pool_id for _, p in scored_pools])
        raise RuntimeError(
            f"All pools exhausted for model '{request.model}'. "
            f"Fallback engine will handle this request."
        )

    async def _call_pool(
        self,
        endpoint: str,
        request: CompletionRequest,
        request_id: str,
    ) -> CompletionResponse:
        # ---------------------------------------------------------------------
        # Send the request to the pool's HTTP endpoint and parse the response.
        #
        # In production, pool workers expose an HTTP API that accepts the same
        # CompletionRequest schema we defined in edge/schemas/request.py.
        # This keeps the contract consistent from edge to pool.
        #
        # MOCK BEHAVIOR:
        # Since we do not have real pool workers running yet, this returns
        # a mock response. In Step 3 we will build the actual pool workers
        # that this call hits.
        # ---------------------------------------------------------------------
        import asyncio
        # Simulate network + model latency
        await asyncio.sleep(0.15)

        return CompletionResponse(
            request_id=request_id,
            model=request.model,
            text=f"[POOL RESPONSE] Routed to {endpoint}. "
                 f"Model: {request.model}. "
                 f"Prompt: '{request.prompt[:40]}...'",
            tokens_generated=35,
            latency_ms=150.0,
            pool_used="placeholder",
            fallback_triggered=False,
        )

    async def close(self):
        await self._http_client.aclose()
