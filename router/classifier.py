# =============================================================================
# router/classifier.py
#
# PURPOSE:
#   Examines each incoming request and decides which POOL TYPE it should go to.
#   The classifier does not pick a specific pool — it picks a category.
#   The dispatcher then picks the specific healthy pool within that category.
#
# CLASSIFICATION FACTORS:
#   1. Prompt token length — long prompts need large-context pools
#   2. Model name — some models only exist in certain pool types
#   3. Request priority — high priority gets online pools, low gets batch
#   4. Stream flag — streaming requests must go to online pools
#
# WHY A SEPARATE CLASSIFIER?
#   Separating classification from dispatching means we can change routing
#   logic without touching dispatch logic and vice versa. It also makes
#   classification independently testable.
# =============================================================================

import structlog
from edge.schemas.request import CompletionRequest, PriorityTier
from router.pool_registry import PoolType

logger = structlog.get_logger(__name__)

# Token count thresholds for pool type decisions.
# A prompt under 4096 tokens is a standard online request.
# A prompt between 4096 and 32768 tokens needs a large-context pool.
# Above 32768 we route to batch unless the client explicitly sets HIGH priority.
ONLINE_CONTEXT_LIMIT = 4096
LARGE_CTX_THRESHOLD = 32768

# Models that only exist in large-context pools.
LARGE_CTX_ONLY_MODELS = {"llama-3-70b", "mixtral-8x7b"}


class RequestClassifier:

    def classify(self, request: CompletionRequest) -> PoolType:
        # ---------------------------------------------------------------------
        # CLASSIFICATION LOGIC — evaluated in priority order
        #
        # Rule 1: LOW priority always goes to batch regardless of context.
        #   Batch clients accept higher latency for lower cost.
        #
        # Rule 2: Models that only exist in large-ctx pools go there directly.
        #   No point checking context length — the model forces the pool type.
        #
        # Rule 3: Long context forces large-ctx pool regardless of priority.
        #   Online pools do not have enough GPU memory for 100K token contexts.
        #
        # Rule 4: Very long context on non-HIGH priority → batch pool.
        #   If context is above the large-ctx threshold and priority is not
        #   HIGH, we route to batch to avoid monopolizing large-ctx capacity.
        #
        # Rule 5: Everything else → online pool.
        #   The default path for standard real-time inference.
        # ---------------------------------------------------------------------

        prompt_token_estimate = self._estimate_tokens(request.prompt)

        log = logger.bind(
            model=request.model,
            priority=request.priority.value,
            estimated_tokens=prompt_token_estimate,
            stream=request.stream,
        )

        # Rule 1 — explicit batch request
        if request.priority == PriorityTier.LOW:
            log.info("classified_batch", reason="low_priority")
            return PoolType.BATCH

        # Rule 2 — model only available in large-context pools
        if request.model in LARGE_CTX_ONLY_MODELS:
            log.info("classified_large_ctx", reason="model_requires_large_ctx")
            return PoolType.LARGE_CONTEXT

        # Rule 3 — prompt is too long for online pools
        if prompt_token_estimate > ONLINE_CONTEXT_LIMIT:
            if prompt_token_estimate > LARGE_CTX_THRESHOLD and request.priority != PriorityTier.HIGH:
                log.info("classified_batch", reason="very_long_context_non_high_priority")
                return PoolType.BATCH
            log.info("classified_large_ctx", reason="prompt_exceeds_online_limit")
            return PoolType.LARGE_CONTEXT

        # Rule 4 — streaming requires online pool (no streaming in batch)
        if request.stream:
            log.info("classified_online", reason="streaming_request")
            return PoolType.ONLINE

        # Rule 5 — default online path
        log.info("classified_online", reason="standard_request")
        return PoolType.ONLINE

    def _estimate_tokens(self, text: str) -> int:
        # ---------------------------------------------------------------------
        # ROUGH TOKEN COUNT ESTIMATE
        #
        # Real tokenization requires loading the model's tokenizer which is
        # expensive. For routing decisions we use a fast approximation:
        # ~0.75 words per token in English, or about 4 characters per token.
        #
        # This does not need to be exact — we are deciding pool TYPE not
        # making billing calculations. A 10% error does not change the route.
        # In production we would use the actual tokenizer for precision.
        # ---------------------------------------------------------------------
        return len(text) // 4
