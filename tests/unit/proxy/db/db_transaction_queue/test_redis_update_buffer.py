import json
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
    WindowSpendTransaction,
    WindowSpendUpdateQueue,
    build_window_spend_transaction,
)
from litellm.proxy.proxy_server import ProxyStartupEvent


@pytest.fixture
def mock_redis_cache():
    """Create a mock RedisCache instance"""
    mock = AsyncMock()
    return mock


@pytest.fixture
def redis_update_buffer(mock_redis_cache):
    """Create a RedisUpdateBuffer with a mock RedisCache"""
    return RedisUpdateBuffer(redis_cache=mock_redis_cache)


def test_validate_redis_transaction_buffer_raises_without_redis():
    """
    When use_redis_transaction_buffer=true but no Redis cache is configured,
    the proxy should refuse to start with a clear error message.
    """
    with pytest.raises(ValueError, match="use_redis_transaction_buffer"):
        ProxyStartupEvent._validate_redis_transaction_buffer_config(
            general_settings={"use_redis_transaction_buffer": True},
            redis_usage_cache=None,
        )


def test_validate_redis_transaction_buffer_passes_with_redis():
    """
    When use_redis_transaction_buffer=true and Redis cache is configured,
    validation should pass without error.
    """
    # Should not raise
    ProxyStartupEvent._validate_redis_transaction_buffer_config(
        general_settings={"use_redis_transaction_buffer": True},
        redis_usage_cache=MagicMock(),
    )


def test_validate_redis_transaction_buffer_passes_when_disabled():
    """
    When use_redis_transaction_buffer is not set or false,
    validation should pass regardless of Redis configuration.
    """
    # Should not raise even without Redis
    ProxyStartupEvent._validate_redis_transaction_buffer_config(
        general_settings={},
        redis_usage_cache=None,
    )


def test_get_transaction_buffer_redis_cache_builds_from_env(monkeypatch):
    """
    When use_redis_transaction_buffer=true, a standalone RedisCache is built from
    REDIS_* environment variables so the buffer works without a Redis cache backend.
    """
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")

    with patch("litellm.proxy.proxy_server.RedisCache") as mock_redis_cache:
        result = ProxyStartupEvent._get_transaction_buffer_redis_cache(
            general_settings={"use_redis_transaction_buffer": True},
        )

    mock_redis_cache.assert_called_once()
    assert mock_redis_cache.call_args.kwargs["host"] == "localhost"
    assert result is mock_redis_cache.return_value


def test_get_transaction_buffer_redis_cache_none_when_disabled():
    """When use_redis_transaction_buffer is not enabled, no standalone cache is built."""
    result = ProxyStartupEvent._get_transaction_buffer_redis_cache(
        general_settings={},
    )
    assert result is None


def test_get_transaction_buffer_redis_cache_none_without_redis_env():
    """
    When use_redis_transaction_buffer=true but no REDIS_* env vars are set,
    no standalone cache is built (startup validation then raises the config error).
    """
    with patch("litellm._redis._redis_kwargs_from_environment", return_value={}):
        result = ProxyStartupEvent._get_transaction_buffer_redis_cache(
            general_settings={"use_redis_transaction_buffer": True},
        )
    assert result is None


def test_get_transaction_buffer_redis_cache_none_without_host_or_url():
    """
    A REDIS_* var that is not a connection target (e.g. REDIS_SOCKET_TIMEOUT) must not
    trigger a build. Without a host or url, get_redis_client raises, so return None and
    let startup validation surface the config error instead of crashing.
    """
    with patch(
        "litellm._redis._redis_kwargs_from_environment",
        return_value={"socket_timeout": 5.0},
    ):
        result = ProxyStartupEvent._get_transaction_buffer_redis_cache(
            general_settings={"use_redis_transaction_buffer": True},
        )
    assert result is None


def test_get_transaction_buffer_redis_cache_parses_string_flag(monkeypatch):
    """
    use_redis_transaction_buffer accepts a string value (e.g. from env/YAML); "true"
    is parsed to a bool before the standalone cache is built.
    """
    monkeypatch.setenv("REDIS_HOST", "localhost")

    with patch("litellm.proxy.proxy_server.RedisCache") as mock_redis_cache:
        result = ProxyStartupEvent._get_transaction_buffer_redis_cache(
            general_settings={"use_redis_transaction_buffer": "true"},
        )

    mock_redis_cache.assert_called_once()
    assert result is mock_redis_cache.return_value


class _ListRedis:
    def __init__(self) -> None:
        self.rows: list[str] = []

    async def async_rpush(self, key: str, values: list[str]) -> int:
        self.rows.extend(values)
        return len(self.rows)

    async def async_rpush_and_trim(self, key: str, values: list[str], max_len: int) -> int:
        self.rows.extend(values)
        pushed_len = len(self.rows)
        del self.rows[:-max_len]
        return pushed_len

    async def async_lpop(self, key: str, count: int | None = None, **kwargs: object) -> list[str] | None:
        if not self.rows:
            return None
        popped = self.rows[:count]
        del self.rows[:count]
        return popped


@pytest.mark.asyncio
async def test_store_spend_logs_in_redis_drops_oldest_rows_past_the_cap():
    redis = _ListRedis()
    buffer = RedisUpdateBuffer(redis_cache=redis)
    buffer.should_commit_spend_updates_to_redis = MagicMock(return_value=True)

    assert await buffer.store_spend_logs_in_redis([{"request_id": "old"}, {"request_id": "mid"}], max_rows=2) is True
    assert await buffer.store_spend_logs_in_redis([{"request_id": "new"}], max_rows=2) is True

    parked = await buffer.get_spend_logs_from_redis_buffer(limit=10)
    assert [row["request_id"] for row in parked] == ["mid", "new"]
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == ()


@pytest.mark.asyncio
async def test_a_spend_log_row_parked_with_a_nan_spend_is_read_back_with_zero_spend():
    buffer: Final = RedisUpdateBuffer(redis_cache=_ListRedis())
    buffer.should_commit_spend_updates_to_redis = MagicMock(return_value=True)

    assert await buffer.store_spend_logs_in_redis([{"request_id": "nan-row", "spend": float("nan")}]) is True

    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == ({"request_id": "nan-row", "spend": 0.0},)


@pytest.mark.asyncio
async def test_store_spend_logs_in_redis_reports_failure_without_redis():
    buffer = RedisUpdateBuffer(redis_cache=None)

    assert await buffer.store_spend_logs_in_redis([{"request_id": "a"}]) is False
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == ()


@pytest.mark.asyncio
async def test_store_spend_logs_in_redis_is_off_unless_transaction_buffering_is_enabled():
    redis = _ListRedis()
    buffer = RedisUpdateBuffer(redis_cache=redis)
    buffer.should_commit_spend_updates_to_redis = MagicMock(return_value=False)

    assert await buffer.store_spend_logs_in_redis([{"request_id": "a"}]) is False
    assert redis.rows == []


@pytest.mark.asyncio
async def test_tool_discoveries_go_back_on_the_queue_when_the_redis_push_fails():
    from litellm.proxy.db.db_transaction_queue.tool_discovery_queue import ToolDiscoveryQueue

    redis_cache = MagicMock()
    redis_cache.async_rpush_pipeline = AsyncMock(side_effect=ConnectionError("redis down"))
    queue = ToolDiscoveryQueue()
    queue.add_update({"tool_name": "get_weather", "origin": "user_defined"})
    queue.add_update({"tool_name": "get_weather", "origin": "user_defined"})

    await RedisUpdateBuffer(redis_cache=redis_cache).store_tool_discoveries_in_redis(queue)

    assert queue.flush() == [{"tool_name": "get_weather", "origin": "user_defined", "call_count": 2}]


def _window_increment(entity_id: str, spend: float) -> WindowSpendTransaction:
    return build_window_spend_transaction(
        entity_type="model_max_budget",
        entity_id=entity_id,
        window_duration="30d",
        window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        spend=spend,
        started_at=datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_window_increments_from_every_pod_drain_from_redis_summed_once():
    redis: Final = _ListRedis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis)
    for spend in (1.25, 0.75):
        queue = WindowSpendUpdateQueue()
        await queue.add_update(_window_increment("key-1", spend))
        await buffer.store_window_spend_updates_in_redis(queue)

    drained: Final = await buffer.get_window_spend_updates_from_redis()

    assert [(item["entity_id"], item["spend"]) for item in drained] == [("key-1", 2.0)]
    assert await buffer.get_window_spend_updates_from_redis() == ()


@pytest.mark.asyncio
async def test_window_payloads_keep_request_ids_for_older_leaders():
    """An older leader indexes request_ids while merging what it popped, and the pop is destructive,
    so a payload without the key would cost a rolling deploy those increments."""
    redis: Final = _ListRedis()
    queue: Final = WindowSpendUpdateQueue()
    await queue.add_update(_window_increment("key-1", 1.25))

    await RedisUpdateBuffer(redis_cache=redis).store_window_spend_updates_in_redis(queue)

    assert [payload["request_ids"] for payload in json.loads(redis.rows[0])] == [[]]


@pytest.mark.asyncio
async def test_window_increments_stay_in_memory_when_the_redis_push_fails():
    redis_cache: Final = MagicMock()
    redis_cache.async_rpush = AsyncMock(side_effect=ConnectionError("redis went away"))
    queue: Final = WindowSpendUpdateQueue()
    await queue.add_update(_window_increment("team-1", 4.0))

    await RedisUpdateBuffer(redis_cache=redis_cache).store_window_spend_updates_in_redis(queue)

    restored: Final = await queue.flush_and_get_aggregated_window_spend_transactions()
    assert [(item["entity_id"], item["spend"]) for item in restored] == [("team-1", 4.0)]


@pytest.mark.asyncio
async def test_restored_window_increments_drain_back_unchanged():
    buffer: Final = RedisUpdateBuffer(redis_cache=_ListRedis())
    increments: Final = (_window_increment("key-1", 3.0),)

    await buffer.restore_window_spend_updates_to_redis(increments)

    assert await buffer.get_window_spend_updates_from_redis() == increments


@pytest.mark.asyncio
async def test_restoring_window_increments_survives_a_redis_error():
    from redis.exceptions import RedisError

    redis_cache: Final = MagicMock()
    redis_cache.async_rpush = AsyncMock(side_effect=RedisError("redis down"))

    await RedisUpdateBuffer(redis_cache=redis_cache).restore_window_spend_updates_to_redis(
        (_window_increment("key-1", 3.0),)
    )

    redis_cache.async_rpush.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_nan_window_spend_parked_by_an_older_version_drains_as_zero():
    redis: Final = _ListRedis()
    redis.rows.extend(
        json.dumps(
            [
                {
                    "entity_type": "model_max_budget",
                    "entity_id": "key-1",
                    "window_duration": "30d",
                    "window_start": "2026-08-01T00:00:00.000000",
                    "spend": spend,
                    "started_at": None,
                }
            ],
            allow_nan=True,
        )
        for spend in (float("nan"), 3.0)
    )

    drained: Final = await RedisUpdateBuffer(redis_cache=redis).get_window_spend_updates_from_redis()

    assert [item["spend"] for item in drained] == [3.0]
