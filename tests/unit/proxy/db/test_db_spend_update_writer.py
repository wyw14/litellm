import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from prisma.errors import DataError as PrismaDataError
from prisma.errors import RawQueryError
from redis.exceptions import DataError

import litellm
from litellm.proxy.db.db_spend_update_writer import (
    DBSpendUpdateWriter,
)
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
    build_window_spend_transaction,
)
from litellm.proxy.db.rollup_lock_timeout import ROLLUP_LOCK_TIMEOUT_SQL
from litellm.responses.streaming_iterator import drain_logging_tasks


@pytest.mark.asyncio
async def test_update_database_attributes_router_rejected_failure_to_model_group_provider():
    db_writer = DBSpendUpdateWriter()
    db_writer._insert_spend_log_to_db = AsyncMock()
    llm_router: Final = litellm.Router(
        model_list=[
            {"model_name": "openai-outage", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-a"}},
            {"model_name": "openai-outage", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-b"}},
        ]
    )

    with (
        patch(
            "litellm.proxy.proxy_server.disable_spend_logs", True
        ),  # test-quality-ok: update_database reads this proxy_server module global at call time; no injection seam
        patch(
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ),  # test-quality-ok: update_database reads this proxy_server module global at call time; no injection seam
        patch(
            "litellm.proxy.proxy_server.user_api_key_cache", MagicMock()
        ),  # test-quality-ok: update_database reads this proxy_server module global at call time; no injection seam
        patch(
            "litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"
        ),  # test-quality-ok: update_database reads this proxy_server module global at call time; no injection seam
        patch(
            "litellm.proxy.proxy_server.llm_router", llm_router
        ),  # test-quality-ok: get_llm_router reads this proxy_server module global at call time; no injection seam
    ):
        await db_writer.update_database(
            token="test-token",
            user_id="test-user",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={
                "model": "openai-outage",
                "litellm_params": {
                    "metadata": {"user_api_key": "test-token", "model_group": "openai-outage", "status": "failure"}
                },
            },
            completion_response={},
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=0.0,
        )
        await asyncio.sleep(0)

    accounting: Final = db_writer._insert_spend_log_to_db.call_args[1]["payload"]["accounting_payload"]
    assert accounting["model_group"] == "openai-outage"
    assert accounting["custom_llm_provider"] == "openai"


def _tool_call_response(*names: str) -> object:

    tool_calls = [SimpleNamespace(function=SimpleNamespace(name=name)) for name in names]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls))])


def _tool_usage_prisma() -> MagicMock:
    prisma = MagicMock()
    prisma.tool_usage_transactions = []
    prisma._tool_usage_transactions_lock = asyncio.Lock()
    prisma.spend_log_transactions = []
    prisma._spend_log_transactions_lock = asyncio.Lock()
    return prisma


def _projected(row: dict) -> dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, int | float]]:
    from litellm.accounting.payload import AccountingPayload
    from litellm.accounting.projection import project

    return {
        (item.table, item.key): dict(item.increments)
        for item in project(AccountingPayload.model_validate(row["accounting_payload"]))
    }


def _rows_of(projected: dict, table: str) -> list[dict[str, int | float]]:
    return [increments for (name, _), increments in projected.items() if name == table]


_HASHED_KEY: Final = "ab" * 32


def _minimal_spend_payload() -> dict:
    return {
        "request_id": "req-tool-1",
        "startTime": datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc),
        "endTime": datetime(2026, 7, 25, 10, 0, 1, tzinfo=timezone.utc),
        "spend": 0.0,
        "total_tokens": 42,
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "mcp_namespaced_tool_name": None,
        "call_type": "acompletion",
        "status": "success",
        "api_key": "",
        "user": "",
        "team_id": "",
        "organization_id": None,
        "end_user": "",
        "agent_id": None,
        "model": "gpt-5.5",
        "model_group": "",
        "model_id": "",
        "custom_llm_provider": "openai",
        "request_tags": "[]",
        "metadata": "{}",
    }


@pytest.mark.asyncio
async def test_update_database_enqueues_tool_usage_for_invoked_tools():
    db_writer = DBSpendUpdateWriter()
    db_writer._insert_spend_log_to_db = AsyncMock()
    prisma = _tool_usage_prisma()

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", False),
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
        patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            return_value=_minimal_spend_payload(),
        ),
    ):
        await db_writer.update_database(
            token="test-token",
            user_id="test-user",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={"model": "gpt-4"},
            completion_response=_tool_call_response("get_weather"),
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=0.1,
        )
        await asyncio.sleep(0)

    assert len(prisma.tool_usage_transactions) == 1
    transaction = prisma.tool_usage_transactions[0]
    assert transaction.request_id == "req-tool-1"
    assert transaction.tool_names == ("get_weather",)
    assert transaction.spend == 0.1
    assert transaction.total_tokens == 42
    assert transaction.date == "2026-07-25"


@pytest.mark.asyncio
async def test_update_database_enqueues_realtime_tool_usage():
    db_writer = DBSpendUpdateWriter()
    db_writer._insert_spend_log_to_db = AsyncMock()
    prisma = _tool_usage_prisma()

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", False),
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
        patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            return_value=_minimal_spend_payload(),
        ),
    ):
        await db_writer.update_database(
            token="test-token",
            user_id="test-user",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={
                "model": "gpt-realtime",
                "realtime_tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "rt_tool", "arguments": "{}"}}
                ],
            },
            completion_response=None,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=0.2,
        )
        await asyncio.sleep(0)

    assert len(prisma.tool_usage_transactions) == 1
    assert prisma.tool_usage_transactions[0].tool_names == ("rt_tool",)


def test_enqueue_tool_registry_upsert_reads_every_choice():
    from types import SimpleNamespace as NS

    db_writer = DBSpendUpdateWriter()
    db_writer.tool_discovery_queue = MagicMock()
    response = NS(
        choices=[
            NS(message=NS(tool_calls=[NS(function=NS(name="tool_alpha"))])),
            NS(message=NS(tool_calls=[NS(function=NS(name="tool_beta"))])),
        ]
    )

    db_writer._enqueue_tool_registry_upsert(kwargs={}, completion_response=response)

    enqueued = [c.args[0]["tool_name"] for c in db_writer.tool_discovery_queue.add_update.call_args_list]
    assert enqueued == ["tool_alpha", "tool_beta"]


@pytest.mark.asyncio
async def test_update_database_skips_tool_usage_when_spend_logs_disabled():
    db_writer = DBSpendUpdateWriter()
    db_writer._insert_spend_log_to_db = AsyncMock()
    prisma = _tool_usage_prisma()

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", True),
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
        patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            return_value=_minimal_spend_payload(),
        ),
    ):
        await db_writer.update_database(
            token="test-token",
            user_id="test-user",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={"model": "gpt-4"},
            completion_response=_tool_call_response("get_weather"),
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=0.1,
        )
        await asyncio.sleep(0)

    assert prisma.tool_usage_transactions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_spend_logs", [True, False])
@pytest.mark.parametrize("session_id", ["session-1", None])
async def test_a_routed_request_reaches_the_auto_router_rollup_whether_or_not_spend_logs_are_kept(
    disable_spend_logs: bool, session_id: str | None
) -> None:
    db_writer = DBSpendUpdateWriter()
    db_writer._insert_spend_log_to_db = AsyncMock()
    prisma = _tool_usage_prisma()
    prisma.autorouter_turn_transactions = []
    prisma._autorouter_turn_transactions_lock = asyncio.Lock()
    routed_payload: Final = {
        **_minimal_spend_payload(),
        "status": "success",
        "api_key": "hashed-key",
        "user": "u1",
        "session_id": session_id,
        "model": "claude-haiku-4-5",
        "model_group": "smart-router",
        "spend": 0.25,
        "startTime": "2026-07-25T10:00:00+00:00",
        "metadata": json.dumps(
            {
                "routing_decision": {"router_model_name": "smart-router", "router_type": "complexity"},
                "autorouter_savings": 1.5,
            }
        ),
    }

    with (
        patch(
            "litellm.proxy.proxy_server.disable_spend_logs", disable_spend_logs
        ),  # test-quality-ok: update_database reads this proxy_server module global at call time; no injection seam
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
        patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            return_value=routed_payload,
        ),
    ):
        await db_writer.update_database(
            token="test-token",
            user_id="u1",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={"model": "smart-router"},
            completion_response=_tool_call_response("get_weather"),
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=0.25,
        )

    (turn,) = prisma.autorouter_turn_transactions
    stored_session: Final = session_id if session_id and not disable_spend_logs else ""
    assert (turn.router_name, turn.router_type, turn.session_id) == ("smart-router", "complexity", stored_session)
    assert (turn.spend, turn.saved_spend) == (0.25, 1.5)
    assert (prisma.tool_usage_transactions == []) is disable_spend_logs


Statement = tuple[str, tuple[object, ...]]


class _RecordingDb:
    """Records the statements the writer sends, in place of a real query engine."""

    def __init__(self, execute_raw: Callable[[], int] | None = None) -> None:
        self.statements: list[Statement] = []
        self.session_settings: list[Statement] = []
        self._execute_raw = execute_raw

    async def execute_raw(self, query: str, *args: object) -> int:
        if query == ROLLUP_LOCK_TIMEOUT_SQL:
            self.session_settings.append((query, args))
            return 0
        self.statements.append((query, args))
        if self._execute_raw is not None:
            return self._execute_raw()
        return len(args)

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator["_RecordingDb"]:
        yield self

    def tx(self, timeout: timedelta | None = None) -> AbstractAsyncContextManager["_RecordingDb"]:
        return self._tx()


# Tag Spend Tracking Tests


@pytest.mark.asyncio
@pytest.mark.parametrize("response_cost", [float("nan"), float("inf"), float("-inf")])
async def test_update_database_records_a_non_finite_response_cost_as_zero(response_cost: float) -> None:
    db_writer: Final = DBSpendUpdateWriter()
    prisma: Final = _tool_usage_prisma()
    prisma.get_request_status = MagicMock(return_value="success")
    response: Final = litellm.ModelResponse(
        model="gpt-5.6-luna",
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", False),
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
    ):
        await db_writer.update_database(
            token=_HASHED_KEY,
            user_id="test-user",
            end_user_id=None,
            team_id="test-team",
            org_id=None,
            kwargs={
                "model": "gpt-5.6-luna",
                "custom_llm_provider": "openai",
                "call_type": "acompletion",
                "litellm_call_id": "req-non-finite-cost",
                "response_cost": response_cost,
                "litellm_params": {"metadata": {"user_api_key": "hashed-key", "user_api_key_team_id": "test-team"}},
            },
            completion_response=response,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=response_cost,
        )
        await drain_logging_tasks()

    (queued,) = prisma.spend_log_transactions
    assert queued["spend"] == 0
    projected: Final = _projected(queued)
    assert projected[("LiteLLM_VerificationToken", (("token", _HASHED_KEY),))] == {"spend": 0}
    assert projected[("LiteLLM_TeamTable", (("team_id", "test-team"),))] == {"spend": 0}
    (user_row,) = _rows_of(projected, "LiteLLM_DailyUserSpend")
    (team_row,) = _rows_of(projected, "LiteLLM_DailyTeamSpend")
    assert (user_row["api_requests"], user_row["spend"]) == (1, 0)
    assert (team_row["api_requests"], team_row["spend"]) == (1, 0)


@pytest.mark.asyncio
async def test_spend_update_path_never_queries_user_cache_with_none_user_id():
    """
    When user_id is None, the spend-update path must not perform a user-cache
    lookup at all. With a Redis-backed auth cache (enable_redis_auth_cache),
    a lookup with key=None raises redis.exceptions.DataError, which would abort
    the request's charge, including the end-user spend that must survive.
    """
    db_writer = DBSpendUpdateWriter()
    prisma = _tool_usage_prisma()

    strict_redis_backed_cache = MagicMock()
    strict_redis_backed_cache.async_get_cache = AsyncMock(side_effect=DataError("Invalid input of type: 'NoneType'"))

    with (
        patch.object(litellm, "max_budget", 0),
        patch("litellm.proxy.proxy_server.disable_spend_logs", True),
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", strict_redis_backed_cache),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "litellm-proxy-budget"),
        patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            return_value=_minimal_spend_payload(),
        ),
    ):
        await db_writer.update_database(
            token=None,
            user_id=None,
            end_user_id="end-user-1",
            team_id=None,
            org_id=None,
            kwargs={"model": "gpt-4", "custom_llm_provider": "openai"},
            completion_response=MagicMock(),
            start_time=datetime.now(),
            end_time=datetime.now(),
            response_cost=0.1,
        )
        await asyncio.sleep(0)

    strict_redis_backed_cache.async_get_cache.assert_not_called()

    (queued,) = prisma.spend_log_transactions
    projected = _projected(queued)
    assert _rows_of(projected, "LiteLLM_EndUserTable") == [{"spend": 0.1}]
    assert ("LiteLLM_EndUserTable", (("user_id", "end-user-1"),)) in projected
    assert _rows_of(projected, "LiteLLM_UserTable") == []


# ---------------------------------------------------------------------------
# Budget window spend flush (LiteLLM_BudgetWindowSpend)
# ---------------------------------------------------------------------------


class _WindowSpendFakeBatcher:
    def __init__(self):
        self.calls = []

    def execute_raw(self, query, *args):
        self.calls.append((query, args))


class _WindowSpendFakeDB:
    """Minimal prisma_client.db that records the raw statements it is handed."""

    def __init__(self, existing_rows=None):
        self.existing_rows = existing_rows or []
        self.query_raw_calls = []
        self.batcher = _WindowSpendFakeBatcher()

    async def query_raw(self, query, *args):
        self.query_raw_calls.append((query, args))
        if "LiteLLM_BudgetWindowSpend" in query:
            return self.existing_rows
        return []

    @asynccontextmanager
    async def _tx(self):
        yield self

    def tx(self, timeout=None):
        return self._tx()

    @asynccontextmanager
    async def _batch(self):
        yield self.batcher

    def batch_(self):
        return self._batch()


class _WindowSpendFakePrisma:
    def __init__(self, db):
        self.db = db


def _window_spend_upserts(db):
    return [params for query, params in db.batcher.calls if "LiteLLM_BudgetWindowSpend" in query]


@pytest.mark.asyncio
async def test_window_spend_queue_is_flushed_without_redis_buffer():
    """The in-memory window queue must reach the DB on the same scheduler tick
    as the other spend queues when the Redis buffer is off."""
    db_writer = DBSpendUpdateWriter()
    await db_writer.window_spend_update_queue.add_update(
        build_window_spend_transaction(
            entity_type="key",
            entity_id="hashed-token",
            window_duration="30d",
            window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            spend=0.5,
        )
    )
    db = _WindowSpendFakeDB(
        existing_rows=[{"entity_type": "key", "entity_id": "hashed-token", "window_duration": "30d"}]
    )

    await db_writer._commit_window_spend_updates_from_memory(_WindowSpendFakePrisma(db))

    upserts = _window_spend_upserts(db)
    assert len(upserts) == 1
    assert upserts[0][0] == "key"
    assert upserts[0][1] == "hashed-token"
    assert upserts[0][2] == "30d"
    assert upserts[0][5] == pytest.approx(0.5)
    assert db_writer.window_spend_update_queue.update_queue.qsize() == 0


@pytest.mark.asyncio
async def test_window_spend_queue_is_handed_to_the_redis_buffer():
    """Multi-pod deployments buffer through Redis, so the window queue has to
    ride the same rpush path as the daily queues."""
    db_writer = DBSpendUpdateWriter()
    mock_redis_update_buffer = AsyncMock()
    mock_redis_update_buffer.get_window_spend_updates_from_redis = AsyncMock(return_value=())
    db_writer.redis_update_buffer = mock_redis_update_buffer
    db_writer.pod_lock_manager = AsyncMock()
    db_writer.pod_lock_manager.acquire_lock = AsyncMock(return_value=True)

    await db_writer._commit_window_spend_updates_with_redis(MagicMock())

    mock_redis_update_buffer.store_window_spend_updates_in_redis.assert_awaited_once_with(
        db_writer.window_spend_update_queue
    )


@pytest.mark.asyncio
async def test_window_spend_transactions_from_redis_are_committed_by_the_lock_winner():
    db_writer = DBSpendUpdateWriter()
    window_transactions = (
        build_window_spend_transaction(
            entity_type="team",
            entity_id="team-1",
            window_duration="7d",
            window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            spend=2.0,
        ),
    )
    mock_redis_update_buffer = AsyncMock()
    mock_redis_update_buffer.get_window_spend_updates_from_redis = AsyncMock(return_value=window_transactions)
    db_writer.redis_update_buffer = mock_redis_update_buffer
    db_writer.pod_lock_manager = AsyncMock()
    db_writer.pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    db = _WindowSpendFakeDB(existing_rows=[{"entity_type": "team", "entity_id": "team-1", "window_duration": "7d"}])

    await db_writer._commit_window_spend_updates_with_redis(_WindowSpendFakePrisma(db))

    upserts = _window_spend_upserts(db)
    assert len(upserts) == 1
    assert upserts[0][:3] == ("team", "team-1", "7d")
    assert upserts[0][5] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_window_spend_transactions_are_not_committed_without_the_pod_lock():
    """Every pod buffers to Redis but only the lock winner may drain it."""
    db_writer = DBSpendUpdateWriter()
    mock_redis_update_buffer = AsyncMock()
    db_writer.redis_update_buffer = mock_redis_update_buffer
    db_writer.pod_lock_manager = AsyncMock()
    db_writer.pod_lock_manager.acquire_lock = AsyncMock(return_value=False)
    db = _WindowSpendFakeDB()

    await db_writer._commit_window_spend_updates_with_redis(_WindowSpendFakePrisma(db))

    mock_redis_update_buffer.get_window_spend_updates_from_redis.assert_not_called()
    assert _window_spend_upserts(db) == []


@pytest.mark.asyncio
async def test_failed_window_spend_commit_requeues_the_increments_and_continues_the_flush():
    """Budget enforcement trusts a current window row without reconciling it
    against LiteLLM_SpendLogs, so a dropped increment would let the key spend
    past its limit after the next reseed. The increments must go back on the
    queue, and the tool registry flush must still run."""
    db_writer = DBSpendUpdateWriter()
    transaction = build_window_spend_transaction(
        entity_type="key",
        entity_id="hashed-token",
        window_duration="30d",
        window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        spend=0.5,
    )
    await db_writer.window_spend_update_queue.add_update(transaction)
    db = _WindowSpendFakeDB()
    db.query_raw = AsyncMock(side_effect=Exception("connection reset"))

    await db_writer._commit_window_spend_updates_from_memory(_WindowSpendFakePrisma(db))

    requeued = await db_writer.window_spend_update_queue.flush_and_get_aggregated_window_spend_transactions()
    assert requeued == (transaction,)


class _DailySpendFakeDB(_WindowSpendFakeDB):
    """Records the daily rollup upserts it is handed and fails the ones aimed at one table."""

    def __init__(self, failing_table: str | None, failure: Exception | None = None) -> None:
        super().__init__()
        self.failing_table = failing_table
        self.failure = failure
        self.execute_raw_calls: list[Statement] = []

    async def execute_raw(self, query: str, *args: object) -> int:
        if self.failing_table is not None and self.failing_table in query:
            raise self.failure if self.failure is not None else Exception("connection reset")
        self.execute_raw_calls.append((query, args))
        return len(args)


def _postgres_rejection(sqlstate: str) -> RawQueryError:
    return RawQueryError(
        data={"user_facing_error": {"error_code": "P2010", "meta": {"code": sqlstate, "message": "db error"}}}
    )


def _batched_postgres_rejection(sqlstate: str) -> PrismaDataError:
    return PrismaDataError(
        data={
            "user_facing_error": {
                "is_panic": False,
                "message": "Error occurred during query execution:\nConnectorError(ConnectorError { "
                f'user_facing_error: None, kind: QueryError(PostgresError {{ code: "{sqlstate}", '
                'message: "db error", severity: "ERROR" }) })',
                "batch_request_idx": 0,
            }
        }
    )


_REQUEUE_SAFETY_CASES: Final = [
    pytest.param(httpx.ReadTimeout("no reply"), False, id="reply lost after the statement was sent"),
    pytest.param(httpx.ConnectError("refused"), True, id="statement never reached the database"),
    pytest.param(_postgres_rejection("23514"), False, id="postgres refused a constraint violation"),
    pytest.param(_batched_postgres_rejection("23514"), False, id="postgres refused a batched constraint violation"),
    pytest.param(_postgres_rejection("42P01"), True, id="table missing"),
]


@pytest.mark.asyncio
async def test_failed_window_spend_commit_from_redis_is_restored_to_redis():
    """The Redis drain is destructive, so a failed window commit has to push
    the popped increments back exactly like the other spend categories."""
    db_writer = DBSpendUpdateWriter()
    window_transactions = (
        build_window_spend_transaction(
            entity_type="team",
            entity_id="team-1",
            window_duration="7d",
            window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            spend=2.0,
        ),
    )
    mock_redis_update_buffer = AsyncMock()
    mock_redis_update_buffer.get_window_spend_updates_from_redis = AsyncMock(return_value=window_transactions)
    mock_redis_update_buffer.restore_window_spend_updates_to_redis = AsyncMock()
    db_writer.redis_update_buffer = mock_redis_update_buffer
    db_writer.pod_lock_manager = AsyncMock()
    db_writer.pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    db = _WindowSpendFakeDB()
    db.query_raw = AsyncMock(side_effect=Exception("connection reset"))

    await db_writer._commit_window_spend_updates_with_redis(_WindowSpendFakePrisma(db))

    assert _window_spend_upserts(db) == []
    mock_redis_update_buffer.restore_window_spend_updates_to_redis.assert_awaited_once_with(window_transactions)
    db_writer.pod_lock_manager.release_lock.assert_awaited_once()


@pytest.mark.parametrize(("failure", "safe_to_resend"), _REQUEUE_SAFETY_CASES)
@pytest.mark.asyncio
async def test_failed_window_spend_commit_from_redis_is_restored_only_when_safe_to_resend(
    failure: Exception, safe_to_resend: bool
):
    db_writer = DBSpendUpdateWriter()
    window_transactions = (
        build_window_spend_transaction(
            entity_type="team",
            entity_id="team-1",
            window_duration="7d",
            window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            spend=2.0,
        ),
    )
    mock_redis_update_buffer = AsyncMock()
    mock_redis_update_buffer.get_window_spend_updates_from_redis = AsyncMock(return_value=window_transactions)
    mock_redis_update_buffer.restore_window_spend_updates_to_redis = AsyncMock()
    db_writer.redis_update_buffer = mock_redis_update_buffer
    db_writer.pod_lock_manager = AsyncMock()
    db_writer.pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    db = _WindowSpendFakeDB()
    db.query_raw = AsyncMock(side_effect=failure)

    await db_writer._commit_window_spend_updates_with_redis(_WindowSpendFakePrisma(db))

    assert _window_spend_upserts(db) == []
    if safe_to_resend:
        mock_redis_update_buffer.restore_window_spend_updates_to_redis.assert_awaited_once_with(window_transactions)
    else:
        mock_redis_update_buffer.restore_window_spend_updates_to_redis.assert_not_awaited()
    db_writer.pod_lock_manager.release_lock.assert_awaited_once()


@pytest.mark.parametrize(("failure", "safe_to_resend"), _REQUEUE_SAFETY_CASES)
@pytest.mark.asyncio
async def test_failed_window_spend_commit_is_requeued_only_when_the_rows_are_provably_uncommitted(
    failure: Exception, safe_to_resend: bool
):
    class _WindowSpendFailureDB(_WindowSpendFakeDB):
        def __init__(self, failure: Exception | None) -> None:
            super().__init__()
            self.failure = failure

        async def query_raw(self, query, *args):
            if self.failure is not None:
                raise self.failure
            return await super().query_raw(query, *args)

    db_writer = DBSpendUpdateWriter()
    transaction = build_window_spend_transaction(
        entity_type="key",
        entity_id="hashed-token",
        window_duration="30d",
        window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        spend=0.5,
    )
    await db_writer.window_spend_update_queue.add_update(transaction)
    db = _WindowSpendFailureDB(failure)

    await db_writer._commit_window_spend_updates_from_memory(_WindowSpendFakePrisma(db))
    db.failure = None
    await db_writer._commit_window_spend_updates_from_memory(_WindowSpendFakePrisma(db))

    assert len(_window_spend_upserts(db)) == (1 if safe_to_resend else 0)
    assert db_writer.window_spend_update_queue.update_queue.empty()


def _responses_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "native",
                "litellm_params": {"model": "openai/gpt-5.1", "api_key": "sk-test"},
                "model_info": {"id": "dep-native"},
            },
            {
                "model_name": "bridged",
                "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-test"},
                "model_info": {"id": "dep-bridged"},
            },
            {
                "model_name": "forced-bridge",
                "litellm_params": {"model": "openai/gpt-5.1", "api_key": "sk-test", "use_chat_completions_api": True},
                "model_info": {"id": "dep-forced-bridge"},
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type, model_id, expects_flush",
    [
        ("aresponses", "dep-bridged", True),
        ("responses", "dep-bridged", True),
        ("aresponses", "dep-forced-bridge", True),
        ("aresponses", None, True),
        ("aresponses", "dep-native", False),
        ("responses", "dep-native", False),
        ("acompletion", "dep-bridged", False),
    ],
)
async def test_insert_spend_log_asks_for_an_immediate_flush_only_on_rows_the_bridge_reads_back(
    call_type: str, model_id: str | None, expects_flush: bool
):
    """
    A `previous_response_id` chained straight off a bridged turn reads that turn's row from the
    DB, so it cannot sit in this worker's queue until the monitor's next poll. A native Responses
    provider holds the conversation itself, so its row waits for the batch write like a chat row.
    """
    db_writer = DBSpendUpdateWriter()
    prisma = _tool_usage_prisma()
    prisma.spend_log_flush_requested = asyncio.Event()
    payload = {"request_id": "req-1", "call_type": call_type, "model_id": model_id}

    await db_writer._insert_spend_log_to_db(payload=payload, prisma_client=prisma, llm_router=_responses_router())

    assert prisma.spend_log_transactions == [payload]
    assert prisma.spend_log_flush_requested.is_set() is expects_flush


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_a_chatgpt_responses_row_queues_without_a_default_chatgpt_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "no-default-login"))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    own_login: Final = tmp_path / "team" / "auth.json"
    own_login.parent.mkdir()
    own_login.write_text(json.dumps({"access_token": "at-team", "account_id": "acct-team"}))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "chatgpt-team",
                "litellm_params": {"model": "chatgpt/gpt-5.6-terra", "chatgpt_auth_file": str(own_login)},
                "model_info": {"id": "dep-chatgpt"},
            }
        ]
    )
    prisma: Final = _tool_usage_prisma()
    prisma.spend_log_flush_requested = asyncio.Event()
    payload: Final = {"request_id": "req-1", "call_type": "aresponses", "model_id": "dep-chatgpt"}

    await DBSpendUpdateWriter()._insert_spend_log_to_db(payload=payload, prisma_client=prisma, llm_router=router)

    assert prisma.spend_log_transactions == [payload]
    assert prisma.spend_log_flush_requested.is_set() is False


def _batch_cost_payload() -> dict:
    return {
        **_minimal_spend_payload(),
        "request_id": "batch_abc_batch_cost",
        "call_type": "aretrieve_batch",
        "status": "success",
    }


def _spend_logs_prisma(inserted: int, existing: object, taken_over: int = 1) -> MagicMock:
    prisma = _tool_usage_prisma()
    prisma.jsonify_object = lambda data: dict(data)
    prisma.writer_db = prisma.db
    prisma.db.query_raw = AsyncMock(return_value=[])
    prisma.db.litellm_spendlogs.create_many = AsyncMock(return_value=inserted)
    prisma.db.litellm_spendlogs.find_unique = AsyncMock(return_value=existing)
    prisma.db.litellm_spendlogs.update_many = AsyncMock(return_value=taken_over)
    return prisma


async def _update_database_with(
    db_writer: DBSpendUpdateWriter,
    prisma: MagicMock,
    payload: dict,
    disable_spend_logs: bool = False,
    response_cost: float = 0.25,
) -> bool:
    with (
        patch(  # test-quality-ok: update_database reads this proxy_server global at call time, no seam
            "litellm.proxy.proxy_server.disable_spend_logs", disable_spend_logs
        ),
        patch(  # test-quality-ok: update_database reads this proxy_server global at call time, no seam
            "litellm.proxy.proxy_server.prisma_client", prisma
        ),
        patch(  # test-quality-ok: update_database reads this proxy_server global at call time, no seam
            "litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"
        ),
        patch(  # test-quality-ok: update_database imports the payload builder inside its body, no seam
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            return_value=payload,
        ),
    ):
        charged = await db_writer.update_database(
            token="test-token",
            user_id="test-user",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={"model": "gpt-5.6-luna", "call_type": payload["call_type"]},
            completion_response=None,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=response_cost,
        )
        await asyncio.sleep(0)
        return charged


@pytest.mark.asyncio
async def test_update_database_writes_no_ordinary_spend_row_with_spend_logs_disabled():
    """With spend logs off the request still queues its charge, flagged so no SpendLogs row is written for it."""
    db_writer = DBSpendUpdateWriter()
    prisma = _spend_logs_prisma(1, None)
    payload = {**_batch_cost_payload(), "call_type": "acompletion"}

    assert await _update_database_with(db_writer, prisma, payload, True) is True

    prisma.db.query_raw.assert_not_called()
    (queued,) = prisma.spend_log_transactions
    assert queued["accounting_log_disabled"] is True
    assert _rows_of(_projected(queued), "LiteLLM_UserTable") == [{"spend": 0.25}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [{**_batch_cost_payload(), "call_type": "acompletion"}, {**_batch_cost_payload(), "status": "failure"}],
    ids=["not_a_batch_retrieve", "failed_batch_retrieve"],
)
async def test_update_database_queues_every_other_spend_row_for_the_next_flush(payload: dict):
    db_writer = DBSpendUpdateWriter()
    prisma = _spend_logs_prisma(1, None)

    assert await _update_database_with(db_writer, prisma, payload) is True

    prisma.db.query_raw.assert_not_called()
    (queued,) = prisma.spend_log_transactions
    assert queued["accounting_log_disabled"] is False
    assert queued["accounting_delivery_id"]
    assert _rows_of(_projected(queued), "LiteLLM_UserTable") == [{"spend": 0.25}]


class _StallingDailySpendFakeDB(_DailySpendFakeDB):
    """Holds the daily upsert aimed at one table until it is cancelled, like a starved pool does.

    The rollback of that transaction waits for ``rollback_release``: the query engine only
    rolls back once the statement it is running has returned, which behind a lock takes
    as long as the lock is held."""

    def __init__(self, stalled_table: str) -> None:
        super().__init__(failing_table=None)
        self.stalled_table = stalled_table
        self.stalled = asyncio.Event()
        self.rollback_release = asyncio.Event()
        self.rolled_back = asyncio.Event()
        self.transaction_outcomes: list[str] = []

    async def execute_raw(self, query: str, *args: object) -> int:
        if self.stalled_table in query:
            self.stalled.set()
            await asyncio.Event().wait()
        return await super().execute_raw(query, *args)

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator["_StallingDailySpendFakeDB"]:
        try:
            yield self
        except BaseException:
            await self.rollback_release.wait()
            self.transaction_outcomes.append("rollback")
            self.rolled_back.set()
            raise
        self.transaction_outcomes.append("commit")


class _CommittingDailySpendFakeDB(_DailySpendFakeDB):
    """Runs the daily upsert at once but holds the COMMIT until released. A COMMIT that has left
    the client lands on the server whether or not the client keeps waiting for the reply."""

    def __init__(self) -> None:
        super().__init__(failing_table=None)
        self.committing = asyncio.Event()
        self.commit_release = asyncio.Event()
        self.transaction_outcomes: list[str] = []

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator["_CommittingDailySpendFakeDB"]:
        try:
            yield self
        except BaseException:
            self.transaction_outcomes.append("rollback")
            raise
        self.committing.set()
        try:
            await self.commit_release.wait()
        finally:
            self.transaction_outcomes.append("commit")


class _CommitFailingDailySpendFakeDB(_CommittingDailySpendFakeDB):
    """COMMIT leaves for the server but the reply comes back as a failure."""

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator["_CommitFailingDailySpendFakeDB"]:
        yield self
        self.committing.set()
        await self.commit_release.wait()
        self.transaction_outcomes.append("commit_failed")
        raise Exception("connection reset")


def test_record_request_error_feeds_the_failure_rollup_with_the_logged_status() -> None:
    from litellm.proxy.db import db_spend_update_writer as writer_module
    from litellm.proxy.db.request_error_tracking import RequestErrorAccumulator
    from litellm.types.proxy.request_errors import RequestErrorKey

    accumulator: Final = RequestErrorAccumulator()
    prisma_client: Final = MagicMock()
    prisma_client.get_request_status.side_effect = lambda payload: json.loads(payload["metadata"])["status"]
    failed_payload: Final[dict[str, object]] = {
        "startTime": datetime(2026, 10, 8, 23, 59, tzinfo=timezone.utc),
        "api_key": "hash-1",
        "team_id": "team-a",
        "user": "user-a",
        "model_group": "gpt-4o",
        "metadata": json.dumps({"status": "failure", "error_information": {"error_code": "503"}}),
    }
    ok_payload: Final[dict[str, object]] = {**failed_payload, "metadata": json.dumps({"status": "success"})}
    with patch.object(writer_module, "request_error_accumulator", accumulator):
        DBSpendUpdateWriter._record_request_error(payload=failed_payload, prisma_client=prisma_client)
        DBSpendUpdateWriter._record_request_error(payload=ok_payload, prisma_client=prisma_client)
    assert accumulator.drain() == {
        RequestErrorKey(
            date="2026-10-08",
            api_key="hash-1",
            team_id="team-a",
            user_id="user-a",
            model_group="gpt-4o",
            status_code=503,
        ): 1
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_error_logs", [False, True])
@pytest.mark.parametrize("status", ["success", "failure"])
async def test_completed_spend_survives_error_log_suppression(
    monkeypatch: pytest.MonkeyPatch, disable_error_logs: bool, status: str
) -> None:
    from litellm.proxy import proxy_server
    from litellm.types.utils import ModelResponse, Usage

    writer: Final = DBSpendUpdateWriter()
    prisma: Final = _tool_usage_prisma()
    prisma.get_request_status.return_value = status
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "general_settings", {"disable_error_logs": disable_error_logs})
    now: Final = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
    result: Final = ModelResponse(
        model="gpt-4o-mini", usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    assert await writer.update_database(
        token="test-token",
        user_id="test-user",
        end_user_id=None,
        team_id=None,
        org_id=None,
        kwargs={
            "model": "gpt-4o-mini",
            "litellm_call_id": "completed-call",
            "litellm_params": {
                "metadata": {
                    "user_api_key": "test-token",
                    "user_api_key_user_id": "test-user",
                    "status": status,
                }
            },
        },
        completion_response=result,
        start_time=now,
        end_time=now,
        response_cost=0.23,
    )
    (queued,) = prisma.spend_log_transactions
    assert queued["accounting_log_disabled"] is (status == "failure" and disable_error_logs)
    (daily,) = _rows_of(_projected(queued), "LiteLLM_DailyUserSpend")
    assert daily["spend"] == pytest.approx(0.23)
    assert (daily["prompt_tokens"], daily["completion_tokens"]) == (10, 5)


@pytest.mark.asyncio
async def test_a_failed_request_feeds_the_request_error_rollup(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.db import db_spend_update_writer as writer_module
    from litellm.proxy.db.request_error_tracking import RequestErrorAccumulator
    from litellm.types.proxy.request_errors import RequestErrorKey

    accumulator = RequestErrorAccumulator()
    prisma = _tool_usage_prisma()
    prisma.get_request_status.return_value = "failure"
    monkeypatch.setattr(writer_module, "request_error_accumulator", accumulator)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    now = datetime(2026, 11, 2, tzinfo=timezone.utc)
    await DBSpendUpdateWriter().update_database(
        token="key",
        user_id="user",
        team_id=None,
        org_id=None,
        end_user_id=None,
        kwargs={
            "model": "ledger-test",
            "call_type": "acompletion",
            "litellm_call_id": "failed-request",
            "litellm_params": {
                "metadata": {
                    "user_api_key": "key",
                    "user_api_key_user_id": "user",
                    "status": "failure",
                    "error_information": {"error_code": "503"},
                }
            },
        },
        completion_response=None,
        start_time=now,
        end_time=now,
        response_cost=0,
    )
    await drain_logging_tasks()
    assert accumulator.drain() == {
        RequestErrorKey(
            date="2026-11-02",
            api_key=prisma.spend_log_transactions[0]["api_key"],
            team_id="",
            user_id="user",
            model_group="ledger-test",
            status_code=503,
        ): 1
    }


class _DeliveryLedger:
    """LiteLLM_SpendLogDelivery keyed by delivery_id, as transaction A's ON CONFLICT insert sees it."""

    def __init__(self, failure: Exception | None = None) -> None:
        self.delivery_ids: list[str] = []
        self.failure = failure

    async def query_raw(self, query: str, *args: object) -> list[dict[str, object]]:
        if self.failure is not None:
            raise self.failure
        if 'INSERT INTO "LiteLLM_SpendLogDelivery"' not in query:
            return []
        fresh = [row["delivery_id"] for row in json.loads(str(args[0])) if row["delivery_id"] not in self.delivery_ids]
        self.delivery_ids.extend(fresh)
        return [{"delivery_id": delivery_id} for delivery_id in fresh]

    async def execute_raw(self, query: str, *args: object) -> int:
        return 0

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator["_DeliveryLedger"]:
        yield self

    def tx(self, timeout: object = None) -> AbstractAsyncContextManager["_DeliveryLedger"]:
        return self._tx()


def _ledger_prisma(ledger: _DeliveryLedger) -> MagicMock:
    prisma = _tool_usage_prisma()
    prisma.jsonify_object = lambda data: dict(data)
    prisma.writer_db = ledger
    return prisma


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_spend_logs", [False, True])
async def test_a_finished_job_is_charged_once_however_often_it_is_polled(disable_spend_logs: bool):
    ledger = _DeliveryLedger()
    prisma = _ledger_prisma(ledger)

    charged = [
        await _update_database_with(DBSpendUpdateWriter(), prisma, _batch_cost_payload(), disable_spend_logs)
        for _ in range(3)
    ]

    assert charged == [True, False, False]
    assert ledger.delivery_ids == ["batch_abc_batch_cost"]
    assert prisma.spend_log_transactions == []


@pytest.mark.asyncio
async def test_a_poll_of_a_job_that_cost_nothing_leaves_the_charge_to_a_later_poll():
    ledger = _DeliveryLedger()
    prisma = _ledger_prisma(ledger)

    free = await _update_database_with(DBSpendUpdateWriter(), prisma, _batch_cost_payload(), response_cost=0.0)
    paid = await _update_database_with(DBSpendUpdateWriter(), prisma, _batch_cost_payload(), response_cost=0.4)

    assert (free, paid) == (True, True)
    assert ledger.delivery_ids == ["batch_abc_batch_cost"]


@pytest.mark.asyncio
async def test_a_job_charge_the_database_could_not_take_is_queued_for_the_next_flush():
    prisma = _ledger_prisma(_DeliveryLedger(failure=httpx.ConnectError("db down")))

    assert await _update_database_with(DBSpendUpdateWriter(), prisma, _batch_cost_payload()) is True

    (queued,) = prisma.spend_log_transactions
    assert queued["request_id"] == "batch_abc_batch_cost"
    assert "accounting_delivery_id" not in queued
    assert _rows_of(_projected(queued), "LiteLLM_UserTable") == [{"spend": 0.25}]


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_spend_logs", [False, True])
async def test_a_job_cost_row_is_logged_only_while_spend_logs_are_on(disable_spend_logs: bool):
    ledger = _DeliveryLedger()
    prisma = _ledger_prisma(ledger)
    statements: list[str] = []
    record_delivery = ledger.query_raw

    async def record(query: str, *args: object) -> list[dict[str, object]]:
        statements.append(query)
        return await record_delivery(query, *args)

    ledger.query_raw = record

    assert await _update_database_with(DBSpendUpdateWriter(), prisma, _batch_cost_payload(), disable_spend_logs)

    assert ledger.delivery_ids == ["batch_abc_batch_cost"]
    assert any('INSERT INTO "LiteLLM_SpendLogs"' in query for query in statements) is not disable_spend_logs
