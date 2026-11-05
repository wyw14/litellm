from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from litellm._internal_context import with_service_target
from litellm.redis.stores.connection import RedisConnection, client_of
from litellm.redis.stores.router_usage.handle import HandleFactory
from litellm.redis.stores.router_usage.types import (
    ROUTER_USAGE_TARGET,
    CounterKind,
    UsageCounter,
    native_counter,
)
from litellm.rust_bridge._native import NativeRouterUsage

__all__ = [
    "ROUTER_USAGE_TARGET",
    "CounterKind",
    "RouterUsage",
    "RouterUsageStore",
    "UsageCounter",
]


class RouterUsage(Protocol):
    def attach_redis(self, connection: RedisConnection | None) -> None: ...
    def flush_local_counters(self) -> None: ...
    async def async_increment(self, counter: UsageCounter, value: float, ttl: int) -> float | None: ...
    async def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None: ...
    async def async_batch_get(self, counters: Sequence[UsageCounter]) -> Sequence[object] | None: ...
    def snapshot(self) -> Mapping[str, float]: ...


class RouterUsageStore:
    def __init__(self, connection: RedisConnection | None, *, handle: HandleFactory = NativeRouterUsage) -> None:
        self._handle: Final = handle(client_of(connection))

    def attach_redis(self, connection: RedisConnection | None) -> None:
        """Adds the Redis tier once, keeping this router's memory counts, as ``DualCache.attach_redis_cache`` does."""
        client: Final = client_of(connection)
        if client is not None:
            self._handle.attach(client)

    def flush_local_counters(self) -> None:
        self._handle.flush_local_counters()

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_increment(self, counter: UsageCounter, value: float, ttl: int) -> float | None:
        return await self._handle.increment(native_counter(counter), value, ttl)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None:
        await self._handle.increment_deployment(deployment_id, model, minute, tokens, requests)

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_batch_get(self, counters: Sequence[UsageCounter]) -> Sequence[object] | None:
        return await self._handle.get_many(tuple(map(native_counter, counters)))

    def snapshot(self) -> Mapping[str, float]:
        return self._handle.snapshot()
