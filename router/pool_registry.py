# =============================================================================
# router/pool_registry.py
#
# PURPOSE:
#   Maintains the registry of all serving pools known to NeuralGate.
#   The router reads this registry to know which pools exist, what models
#   they serve, what hardware they run on, and what their current status is.
#
# THINK OF IT AS:
#   A phonebook for serving pools. Before the router can send a request
#   anywhere it needs to know what destinations exist and whether they are
#   reachable.
#
# IN PRODUCTION:
#   This registry would be backed by a service discovery system like
#   Consul or a database. Here we initialize it from a YAML config file
#   and keep live status in Redis.
# =============================================================================

import yaml
import structlog
import redis.asyncio as aioredis
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = structlog.get_logger(__name__)


class PoolType(str, Enum):
    ONLINE = "online"           # Low latency, real-time requests
    LARGE_CONTEXT = "large_ctx" # Long context window requests
    BATCH = "batch"             # Async high-throughput requests


class PoolStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"       # Serving but with reduced capacity
    UNHEALTHY = "unhealthy"     # Not serving — removed from rotation
    WARMING = "warming"         # Starting up — not yet ready


@dataclass
class PoolConfig:
    pool_id: str
    pool_type: PoolType
    models: list[str]           # Models this pool can serve
    max_context_length: int     # Maximum token context this pool supports
    worker_count: int           # Number of GPU workers in this pool
    gpu_type: str               # e.g. "A100", "T4", "mock"
    endpoint: str               # HTTP endpoint to send requests to
    status: PoolStatus = PoolStatus.WARMING
    priority_tier: str = "normal"
    fallback_order: list[str] = field(default_factory=list)


class PoolRegistry:
    # -------------------------------------------------------------------------
    # WHAT THIS CLASS DOES:
    #   Loads pool configurations from YAML at startup.
    #   Keeps live pool status in Redis so all server instances share the same
    #   view of which pools are healthy.
    #   Provides lookup methods the router uses to find candidate pools.
    # -------------------------------------------------------------------------

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self._pools: dict[str, PoolConfig] = {}

    async def load_from_config(self, config_path: str = "configs/pool_config.yaml"):
        # ---------------------------------------------------------------------
        # Load pool definitions from YAML config.
        # If the config file does not exist yet we load default pools so
        # the system can start without a config file during development.
        # ---------------------------------------------------------------------
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            pools_data = config.get("pools", [])
            logger.info("pool_config_loaded", path=config_path, count=len(pools_data))
        except FileNotFoundError:
            logger.warning("pool_config_not_found", path=config_path, action="using_defaults")
            pools_data = self._default_pools()

        for pool_data in pools_data:
            pool = PoolConfig(**pool_data)
            self._pools[pool.pool_id] = pool
            logger.info("pool_registered", pool_id=pool.pool_id, type=pool.pool_type)

    def _default_pools(self) -> list[dict]:
        # Default pools used in development when no config file exists.
        return [
            {
                "pool_id": "online-pool-1",
                "pool_type": "online",
                "models": ["llama-3-8b", "mistral-7b"],
                "max_context_length": 8192,
                "worker_count": 4,
                "gpu_type": "mock",
                "endpoint": "http://localhost:9001",
                "fallback_order": ["online-pool-2", "large-ctx-pool-1"],
            },
            {
                "pool_id": "online-pool-2",
                "pool_type": "online",
                "models": ["llama-3-8b", "mistral-7b"],
                "max_context_length": 8192,
                "worker_count": 4,
                "gpu_type": "mock",
                "endpoint": "http://localhost:9002",
                "fallback_order": ["large-ctx-pool-1"],
            },
            {
                "pool_id": "large-ctx-pool-1",
                "pool_type": "large_ctx",
                "models": ["llama-3-70b", "mixtral-8x7b"],
                "max_context_length": 128000,
                "worker_count": 2,
                "gpu_type": "mock",
                "endpoint": "http://localhost:9003",
                "fallback_order": ["online-pool-1"],
            },
            {
                "pool_id": "batch-pool-1",
                "pool_type": "batch",
                "models": ["llama-3-8b", "llama-3-70b", "mistral-7b", "mixtral-8x7b"],
                "max_context_length": 128000,
                "worker_count": 8,
                "gpu_type": "mock",
                "endpoint": "http://localhost:9004",
                "fallback_order": [],
            },
        ]

    def get_pool(self, pool_id: str) -> Optional[PoolConfig]:
        return self._pools.get(pool_id)

    def get_pools_for_model(self, model: str) -> list[PoolConfig]:
        # Return all pools that can serve the requested model.
        return [p for p in self._pools.values() if model in p.models]

    def get_pools_by_type(self, pool_type: PoolType) -> list[PoolConfig]:
        return [p for p in self._pools.values() if p.pool_type == pool_type]

    def all_pools(self) -> list[PoolConfig]:
        return list(self._pools.values())

    async def update_pool_status(self, pool_id: str, status: PoolStatus):
        # Write status to Redis so all server instances see the update.
        await self.redis.set(f"pool:status:{pool_id}", status.value, ex=60)
        if pool_id in self._pools:
            self._pools[pool_id].status = status
        logger.info("pool_status_updated", pool_id=pool_id, status=status)

    async def get_pool_status(self, pool_id: str) -> PoolStatus:
        # Read from Redis for the most current status.
        status_str = await self.redis.get(f"pool:status:{pool_id}")
        if status_str:
            return PoolStatus(status_str)
        return PoolStatus.WARMING
