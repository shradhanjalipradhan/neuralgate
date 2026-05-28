# =============================================================================
# edge/schemas/request.py
#
# PURPOSE:
#   Defines the exact shape of every inference request that enters NeuralGate.
#   Think of this as the contract between the outside world and our system.
#   If a request does not match this contract, it is rejected here before it
#   touches any business logic, any queue, or any GPU resource.
#
# WHY THIS FILE EXISTS FIRST:
#   Every other file in this project either produces or consumes a
#   CompletionRequest or CompletionResponse. We define the shape before
#   writing any logic so everything else has a clear consistent foundation.
#
# DO YOU NEED TO RUN THIS FILE? NO.
#   This file only defines data shapes. Nothing executes until server.py
#   imports and uses these classes. Just save it and move on.
# =============================================================================

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# -----------------------------------------------------------------------------
# PriorityTier — the three levels of request priority in NeuralGate
#
# WHY AN ENUM?
#   Priority can only ever be one of three values: high, normal, or low.
#   An enum enforces this at the type level. If a client sends priority="urgent"
#   Pydantic rejects it automatically without us writing any extra code.
#   Downstream in the router we can compare priority == PriorityTier.HIGH
#   with full confidence the value is always one of these three.
#
# WHY (str, Enum)?
#   Inheriting from both str and Enum means the value serializes to a plain
#   string in JSON ("high") instead of <PriorityTier.HIGH: 'high'> which is
#   not JSON-serializable.
# -----------------------------------------------------------------------------
class PriorityTier(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# -----------------------------------------------------------------------------
# CompletionRequest — the validated shape of every incoming inference request
#
# WHY PYDANTIC BaseModel?
#   When a request arrives over HTTP it is raw JSON with zero guarantees.
#   Pydantic transforms that into a typed Python object where required fields
#   are enforced, types are validated, value ranges are checked, and custom
#   business rules are applied — all before our business logic runs.
# -----------------------------------------------------------------------------
class CompletionRequest(BaseModel):

    # -------------------------------------------------------------------------
    # model — which LLM to route this request to
    # Field(...) means REQUIRED. Missing this field = immediate rejection.
    # -------------------------------------------------------------------------
    model: str = Field(
        ...,
        description="Model identifier to route this request to",
        examples=["llama-3-8b", "llama-3-70b", "mistral-7b"]
    )

    # -------------------------------------------------------------------------
    # prompt — the text input the model responds to
    # min_length=1 rejects empty strings.
    # max_length=128000 is roughly 96000 words — our maximum context support.
    # The custom validator below also rejects whitespace-only strings.
    # -------------------------------------------------------------------------
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=128000,
        description="Input prompt for the model"
    )

    # -------------------------------------------------------------------------
    # max_tokens — how many tokens the model should generate
    # ge=1 means at least 1 token must be requested.
    # le=4096 caps runaway requests that would monopolize GPU memory.
    # A client asking for 100000 tokens is either a bug or an attack.
    # -------------------------------------------------------------------------
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate"
    )

    # -------------------------------------------------------------------------
    # temperature — controls randomness in token sampling
    # 0.0 = fully deterministic (always picks the most likely next token)
    # 2.0 = very random (samples from a much wider token distribution)
    # 0.7 = the practical default — creative but coherent
    # This directly affects the softmax output of the model's final layer.
    # -------------------------------------------------------------------------
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = deterministic, 2 = very random"
    )

    # -------------------------------------------------------------------------
    # top_p — nucleus sampling threshold
    # At each step take the smallest set of tokens whose combined probability
    # reaches top_p, then sample only from those. Cuts off the long tail of
    # very unlikely tokens that tend to produce incoherent output.
    # -------------------------------------------------------------------------
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold"
    )

    # -------------------------------------------------------------------------
    # priority — which serving tier this request targets
    # HIGH   -> online pool, lowest latency, warm fallback always ready
    # NORMAL -> standard online pool
    # LOW    -> batch pool, high throughput, latency is less critical
    # The router and fallback engine both read this field.
    # -------------------------------------------------------------------------
    priority: PriorityTier = Field(
        default=PriorityTier.NORMAL,
        description="Request priority — affects routing and fallback behavior"
    )

    # -------------------------------------------------------------------------
    # stream — whether to return tokens one by one as they are generated
    # False -> wait for full response, return everything at once
    # True  -> return each token as the model generates it (for chat UIs)
    # Streaming changes the HTTP response from JSON to Server-Sent Events.
    # -------------------------------------------------------------------------
    stream: bool = Field(
        default=False,
        description="Whether to stream tokens back as they are generated"
    )

    # -------------------------------------------------------------------------
    # request_id — optional client-provided ID for end-to-end tracing
    # Optional[str] means this can be a string OR None.
    # If provided we carry it through every log metric and trace.
    # If absent server.py generates a UUID automatically.
    # -------------------------------------------------------------------------
    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-provided request ID for tracing"
    )

    # -------------------------------------------------------------------------
    # VALIDATOR: reject whitespace-only prompts
    # Pydantic's min_length=1 passes for "   " because it has length 3.
    # This catches that. It also strips all leading/trailing whitespace
    # so the model never wastes context window space on padding.
    # -------------------------------------------------------------------------
    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()

    # -------------------------------------------------------------------------
    # VALIDATOR: reject unknown model names
    # We only serve models we have deployed in our serving pools.
    # Catching this here means the router never sees an unknown model name.
    # Cheaper to reject at the edge than after acquiring a queue slot.
    # -------------------------------------------------------------------------
    @field_validator("model")
    @classmethod
    def model_must_be_known(cls, v: str) -> str:
        allowed = {
            "llama-3-8b",
            "llama-3-70b",
            "mistral-7b",
            "mixtral-8x7b",
        }
        if v not in allowed:
            raise ValueError(
                f"Unknown model '{v}'. Allowed models: {sorted(allowed)}"
            )
        return v


# -----------------------------------------------------------------------------
# CompletionResponse — the shape of every response NeuralGate sends back
#
# Defining the response shape as a Pydantic model means FastAPI validates our
# own output before sending it to clients and auto-generates API documentation.
#
# pool_used       — which serving pool handled this request. Critical for
#                   debugging latency issues in post-incident reviews.
# fallback_triggered — True means a fallback model handled this request not
#                   the originally requested model. Clients can use this to
#                   decide whether to retry later for full quality.
# latency_ms      — end-to-end latency we log for every request to track
#                   p95 and p99 distributions over time.
# -----------------------------------------------------------------------------
class CompletionResponse(BaseModel):
    request_id: str
    model: str
    text: str
    tokens_generated: int
    latency_ms: float
    pool_used: str
    fallback_triggered: bool = False
