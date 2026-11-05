from __future__ import annotations

from typing import Final

from litellm._internal_context import with_service_target
from litellm.redis.stores.connection import RedisConnection, client_of
from litellm.redis.stores.sensitive_route_pins.handle import HandleFactory
from litellm.rust_bridge._native import NativeSensitiveRoutePins

__all__ = [
    "SENSITIVE_ROUTE_PINS_TARGET",
    "SensitiveRoutePinStore",
]

SENSITIVE_ROUTE_PINS_TARGET: Final = "sensitive_route_pins"


class SensitiveRoutePinStore:
    def __init__(self, connection: RedisConnection | None, *, handle: HandleFactory = NativeSensitiveRoutePins) -> None:
        self._handle: Final = handle(client_of(connection))

    @with_service_target(SENSITIVE_ROUTE_PINS_TARGET)
    async def get(self, tenant: str, session_id: str, default_ttl: float) -> str | None:
        return await self._handle.get(tenant, session_id, default_ttl)

    @with_service_target(SENSITIVE_ROUTE_PINS_TARGET)
    async def set(self, tenant: str, session_id: str, model: str, ttl: float) -> None:
        await self._handle.set(tenant, session_id, model, ttl)
