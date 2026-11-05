from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, TypeAlias

from litellm.rust_bridge._native import NativeRedisClient


class SensitiveRoutePinsHandle(Protocol):
    def get(self, tenant: str, session_id: str, default_ttl_seconds: float) -> Awaitable[str | None]: ...
    def set(self, tenant: str, session_id: str, model: str, ttl_seconds: float) -> Awaitable[None]: ...


HandleFactory: TypeAlias = Callable[[NativeRedisClient | None], SensitiveRoutePinsHandle]
