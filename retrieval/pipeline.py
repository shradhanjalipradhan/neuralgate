# =============================================================================
# retrieval/pipeline.py
#
# PURPOSE:
#   Orchestrates the full RAG (Retrieval-Augmented Generation) pipeline.
#   Takes an inference request, retrieves relevant context within a time budget,
#   assembles the augmented prompt, and returns it for model serving.
#
# THE FULL PIPELINE:
#   1. Embed the query (convert text to vector)
#   2. Search vector store for similar document chunks (within time budget)
#   3. If budget exceeded: use cached fallback chunks
#   4. Truncate chunks to fit within context window
#   5. Assemble augmented prompt: [retrieved context] + [original prompt]
#   6. Return augmented prompt to router for serving
#
# THE INCIDENT THAT SHAPED THIS DESIGN:
#   When we first deployed RAG for an enterprise client, retrieval and inference
#   ran sequentially in the same request path. As their document corpus grew,
#   retrieval times became inconsistent — p99 latency hit 4200ms because some
#   queries triggered slow index scans. The time budget system cut p99 to 780ms
#   by degrading gracefully instead of blocking on slow retrievals.
# =============================================================================

import time
import structlog
from edge.schemas.request import CompletionRequest, PriorityTier
from retrieval.vector_store import VectorStore, DocumentChunk
from retrieval.time_budget import TimeBudget, run_with_budget
from retrieval.context_truncator import ContextTruncator

logger = structlog.get_logger(__name__)

MAX_CONTEXT_TOKENS_FULL     = 2048    # Full retrieval context budget
MAX_CONTEXT_TOKENS_FALLBACK = 512     # Reduced context when budget exceeded


class RetrievalPipeline:

    def __init__(self):
        self.vector_store = VectorStore()
        self.truncator = ContextTruncator()

    async def augment(
        self,
        request: CompletionRequest,
        request_id: str,
    ) -> tuple[str, dict]:
        # -----------------------------------------------------------------------
        # Returns (augmented_prompt, retrieval_metadata).
        # retrieval_metadata contains info for observability:
        #   used_fallback, chunks_retrieved, context_tokens, retrieval_latency_ms
        # -----------------------------------------------------------------------
        log = logger.bind(request_id=request_id, model=request.model)
        budget = TimeBudget.for_priority(request.priority.value)
        start = time.time()

        # Step 1: Embed query and search within time budget
        async def full_retrieval():
            embedding = await self.vector_store.embed_query(request.prompt)
            chunks = await self.vector_store.search(embedding, top_k=5)
            return chunks

        async def cached_fallback():
            # Fast path: return pre-cached top results without embedding
            log.warning("retrieval_using_cached_fallback", request_id=request_id)
            return await self.vector_store.search_cached(request.prompt, top_k=3)

        chunks, used_fallback = await run_with_budget(
            operation=full_retrieval,
            budget=budget,
            fallback_fn=cached_fallback,
            operation_name="vector_search",
        )

        retrieval_latency_ms = (time.time() - start) * 1000

        # Step 2: Truncate to fit context window
        max_tokens = MAX_CONTEXT_TOKENS_FALLBACK if used_fallback else MAX_CONTEXT_TOKENS_FULL
        context_text, context_tokens = self.truncator.truncate(
            chunks=chunks,
            max_context_tokens=max_tokens,
            query=request.prompt,
        )

        # Step 3: Assemble augmented prompt
        if context_text:
            augmented_prompt = (
                f"Use the following context to answer the question.\n\n"
                f"{context_text}\n\n"
                f"Question: {request.prompt}"
            )
        else:
            # No context retrieved — pass original prompt unchanged
            augmented_prompt = request.prompt

        metadata = {
            "used_fallback": used_fallback,
            "chunks_retrieved": len(chunks),
            "context_tokens": context_tokens,
            "retrieval_latency_ms": round(retrieval_latency_ms, 1),
        }

        log.info("retrieval_complete", **metadata)
        return augmented_prompt, metadata
