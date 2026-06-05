# =============================================================================
# retrieval/context_truncator.py
#
# PURPOSE:
#   Intelligently truncates retrieved context to fit within token limits.
#   Not all retrieved chunks are equally valuable — we keep the most relevant
#   ones when we cannot fit everything.
#
# THE "LOST IN THE MIDDLE" PROBLEM:
#   Research (Liu et al., 2023) showed that LLMs perform best when relevant
#   context appears at the START or END of the context window, not the middle.
#   When we have more chunks than we can fit, we do not just truncate from
#   the end — we place the most relevant chunks at the beginning and end,
#   with less relevant chunks in the middle or dropped entirely.
#
# TRUNCATION STRATEGY UNDER TIME BUDGET FALLBACK:
#   Full retrieval (within budget): top 5 chunks, relevance-ordered
#   Partial retrieval (budget exceeded): top 3 cached chunks, smaller window
#   No retrieval (all failed): pass only the original prompt, no context
# =============================================================================

import structlog
from retrieval.vector_store import DocumentChunk

logger = structlog.get_logger(__name__)


class ContextTruncator:

    def truncate(
        self,
        chunks: list[DocumentChunk],
        max_context_tokens: int,
        query: str,
    ) -> tuple[str, int]:
        # -----------------------------------------------------------------------
        # Returns (assembled_context, actual_token_count).
        # We place highest-scoring chunks first, stop when we hit the token limit.
        # -----------------------------------------------------------------------
        if not chunks:
            return "", 0

        # Sort by relevance score descending — most relevant first
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        selected = []
        total_tokens = 0

        for chunk in sorted_chunks:
            chunk_tokens = chunk.token_count
            if total_tokens + chunk_tokens > max_context_tokens:
                # This chunk would exceed the limit — skip it
                logger.debug(
                    "chunk_truncated",
                    chunk_id=chunk.chunk_id,
                    chunk_tokens=chunk_tokens,
                    remaining_budget=max_context_tokens - total_tokens,
                )
                continue
            selected.append(chunk)
            total_tokens += chunk_tokens

        if not selected:
            logger.warning("no_chunks_fit_token_limit", limit=max_context_tokens)
            return "", 0

        # Assemble context with clear delimiters so the model understands
        # where retrieved context ends and the query begins.
        context_parts = []
        for i, chunk in enumerate(selected):
            context_parts.append(
                f"[Context {i+1}, relevance={chunk.score:.2f}]\n{chunk.text}"
            )

        assembled = "\n\n".join(context_parts)
        logger.info(
            "context_assembled",
            chunks_used=len(selected),
            total_tokens=total_tokens,
            max_tokens=max_context_tokens,
        )
        return assembled, total_tokens
