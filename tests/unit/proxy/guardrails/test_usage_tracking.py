import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.proxy.guardrails import usage_tracking
from litellm.proxy.guardrails.usage_tracking import (
    _MAX_PENDING_ROWS,
    PendingRollups,
    _capped,
    process_spend_logs_guardrail_usage,
)


def _prisma() -> MagicMock:
    client = MagicMock()
    db = client.db
    db.litellm_dailyguardrailmetrics.upsert = AsyncMock()
    db.litellm_dailyguardrailusageunits.upsert = AsyncMock()
    db.litellm_spendlogguardrailindex.create_many = AsyncMock()
    return client


def _payload(
    request_id: str,
    *,
    team_id: str | None = "team-a",
    api_key: str = "hashed-key-1",
    usage: dict[str, Any] | None = None,
    guardrail_status: str = "success",
    cost_by_unit: dict[str, Any] | None = None,
    cost_in_spend: bool | None = None,
    guardrail_id: str = "bedrock-guard",
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "guardrail_id": guardrail_id,
        "guardrail_status": guardrail_status,
    }
    if usage is not None:
        entry["guardrail_usage"] = usage
    if cost_by_unit is not None:
        entry["guardrail_cost_by_unit"] = cost_by_unit
    if cost_in_spend is not None:
        entry["guardrail_cost_in_spend"] = cost_in_spend
    return {
        "request_id": request_id,
        "startTime": datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc),
        "team_id": team_id,
        "api_key": api_key,
        "metadata": json.dumps({"guardrail_information": [entry]}),
    }


@pytest.mark.asyncio
async def test_flagged_status_counts_as_flagged_not_passed_or_blocked():
    """LIT-6894: a custom code flag() verdict lands in flagged_count on the Monitor rollup."""
    prisma = _prisma()
    logs = [
        _payload("r1", guardrail_status="success"),
        _payload("r2", guardrail_status="guardrail_flagged"),
        _payload("r3", guardrail_status="guardrail_intervened"),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs)

    create = prisma.db.litellm_dailyguardrailmetrics.upsert.call_args.kwargs["data"]["create"]
    assert (create["requests_evaluated"], create["passed_count"], create["flagged_count"], create["blocked_count"]) == (
        3,
        1,
        1,
        1,
    )


def _fake_sleep() -> tuple[AsyncMock, list[float]]:
    delays: list[float] = []
    sleep = AsyncMock(side_effect=lambda delay: delays.append(delay))
    return sleep, delays


@pytest.mark.asyncio
async def test_persistent_upsert_failure_stops_after_three_retries():
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailmetrics.upsert.side_effect = httpx.ConnectError("db down")
    sleep, delays = _fake_sleep()
    pending = PendingRollups()

    await process_spend_logs_guardrail_usage(
        prisma, [_payload("r1", usage={"topicPolicyUnits": 1})], sleep=sleep, pending=pending
    )

    assert prisma.db.litellm_dailyguardrailmetrics.upsert.call_count == 4
    assert delays == [1, 2, 4]
    assert dict(pending.metrics) == {
        ("bedrock-guard", "2026-08-17"): {
            "requests_evaluated": 1,
            "passed_count": 1,
            "blocked_count": 0,
            "flagged_count": 0,
        }
    }


def _metrics_guardrails(prisma: MagicMock) -> list[str]:
    return [
        c.kwargs["data"]["create"]["guardrail_id"]
        for c in prisma.db.litellm_dailyguardrailmetrics.upsert.call_args_list
    ]


@pytest.mark.asyncio
async def test_transient_upsert_failure_is_retried_with_backoff_for_failed_rows_only():
    """A connection error proves the write never reached the database, so only that row is re-sent after a backoff."""
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailmetrics.upsert.side_effect = [httpx.ConnectError("blip"), None, None]
    sleep, delays = _fake_sleep()

    await process_spend_logs_guardrail_usage(
        prisma, [_payload("r1", guardrail_id="a"), _payload("r2", guardrail_id="b")], sleep=sleep
    )

    assert _metrics_guardrails(prisma) == ["a", "b", "a"]
    assert delays == [1]


@pytest.mark.asyncio
async def test_post_send_failure_is_never_retried_so_increments_cannot_double_count():
    """A read timeout may have committed the increment, so it is attempted once while a refused row is retried."""
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailmetrics.upsert.side_effect = [
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectError("refused"),
        None,
    ]
    sleep, delays = _fake_sleep()

    await process_spend_logs_guardrail_usage(
        prisma, [_payload("r1", guardrail_id="a"), _payload("r2", guardrail_id="b")], sleep=sleep
    )

    assert _metrics_guardrails(prisma) == ["a", "b", "b"]
    assert delays == [1]


@pytest.mark.asyncio
async def test_generic_upsert_exception_is_terminal_for_that_row_only():
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailmetrics.upsert.side_effect = [RuntimeError("constraint violation"), None]
    sleep, delays = _fake_sleep()

    await process_spend_logs_guardrail_usage(
        prisma, [_payload("r1", guardrail_id="a"), _payload("r2", guardrail_id="b")], sleep=sleep
    )

    assert _metrics_guardrails(prisma) == ["a", "b"]
    assert delays == []


@pytest.mark.asyncio
async def test_retry_exhausted_rows_are_requeued_and_land_on_the_next_flush():
    """LIT-5761: rows whose connection-error retries exhaust are merged into the next flushed batch."""
    pending = PendingRollups()
    down = _prisma()
    down.db.litellm_dailyguardrailmetrics.upsert.side_effect = httpx.ConnectError("db down")
    sleep, _ = _fake_sleep()

    await process_spend_logs_guardrail_usage(down, [_payload("r1")], sleep=sleep, pending=pending)
    recovered = _prisma()
    await process_spend_logs_guardrail_usage(recovered, [_payload("r2")], sleep=sleep, pending=pending)

    metrics_create = recovered.db.litellm_dailyguardrailmetrics.upsert.call_args.kwargs["data"]["create"]
    assert metrics_create["requests_evaluated"] == 2
    assert not pending.metrics


@pytest.mark.asyncio
async def test_ambiguous_failures_are_never_requeued():
    """A post-send failure may have committed, so requeueing it would double-count."""
    pending = PendingRollups()
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailmetrics.upsert.side_effect = httpx.ReadTimeout("maybe committed")
    sleep, delays = _fake_sleep()

    await process_spend_logs_guardrail_usage(prisma, [_payload("r1")], sleep=sleep, pending=pending)

    assert delays == []
    assert not pending.metrics


@pytest.mark.asyncio
async def test_guardrail_usage_units_are_left_to_the_accounting_ledger():
    prisma = _prisma()

    await process_spend_logs_guardrail_usage(prisma, [_payload("r1", usage={"topicPolicyUnits": 2})])

    prisma.db.litellm_dailyguardrailusageunits.upsert.assert_not_called()
    assert prisma.db.litellm_dailyguardrailmetrics.upsert.await_count == 1


def test_pending_requeue_is_capped_dropping_oldest_rows():
    rows = {index: index for index in range(_MAX_PENDING_ROWS + 5)}

    capped = _capped(rows, "usage unit")

    assert len(capped) == _MAX_PENDING_ROWS
    assert 4 not in capped
    assert _MAX_PENDING_ROWS + 4 in capped


@pytest.mark.asyncio
async def test_not_run_entries_are_indexed_but_not_counted_as_evaluations():
    """
    LIT-6314 records a not_run entry when message scoping leaves a guardrail
    nothing to scan. The guardrail never evaluated the request, so counting it
    as a passed evaluation would inflate daily pass rates; it still gets an
    index row so per-request drill-down finds the spend log.
    """
    prisma = _prisma()
    logs = [_payload("r1", guardrail_status="not_run"), _payload("r2")]

    await process_spend_logs_guardrail_usage(prisma, logs)

    metrics_create = prisma.db.litellm_dailyguardrailmetrics.upsert.call_args.kwargs["data"]["create"]
    assert metrics_create["requests_evaluated"] == 1
    assert metrics_create["passed_count"] == 1
    index_rows = prisma.db.litellm_spendlogguardrailindex.create_many.call_args.kwargs["data"]
    assert sorted(row["request_id"] for row in index_rows) == ["r1", "r2"]


@pytest.mark.asyncio
async def test_not_run_entry_shares_index_key_with_evaluated_sibling_of_same_name():
    """
    The not_run entry from the shared base guardrail carries only guardrail_name,
    while the evaluated entry from the same guardrail (e.g. content filter on the
    output of a logging_only run) carries its guardrail_id. Keying them differently
    lists one request twice in the monitor, once as not_run and once as passed.
    """
    prisma = _prisma()
    payload = _payload("r1")
    payload["metadata"] = json.dumps(
        {
            "guardrail_information": [
                {"guardrail_name": "cf", "guardrail_status": "not_run"},
                {
                    "guardrail_name": "cf",
                    "guardrail_id": "cf-uuid",
                    "policy_id": "pol-1",
                    "guardrail_status": "success",
                },
                {"guardrail_name": "other", "guardrail_status": "not_run"},
            ]
        }
    )

    await process_spend_logs_guardrail_usage(prisma, [payload])

    index_rows = prisma.db.litellm_spendlogguardrailindex.create_many.call_args.kwargs["data"]
    assert sorted((row["guardrail_id"], row["policy_id"]) for row in index_rows) == [
        ("cf-uuid", "pol-1"),
        ("other", None),
    ]
    metrics_create = prisma.db.litellm_dailyguardrailmetrics.upsert.call_args.kwargs["data"]["create"]
    assert (metrics_create["guardrail_id"], metrics_create["requests_evaluated"]) == ("cf-uuid", 1)


@pytest.mark.asyncio
async def test_malformed_not_run_entry_does_not_drop_the_batch():
    prisma = _prisma()
    payload = _payload("r1")
    payload["metadata"] = json.dumps(
        {
            "guardrail_information": [
                {"guardrail_name": ["not", "a", "string"], "guardrail_status": "success"},
                {"guardrail_name": "", "guardrail_id": "cf-uuid", "guardrail_status": "success"},
                {"guardrail_status": "success"},
            ]
        }
    )

    await process_spend_logs_guardrail_usage(prisma, [payload])

    index_rows = prisma.db.litellm_spendlogguardrailindex.create_many.call_args.kwargs["data"]
    assert [row["guardrail_id"] for row in index_rows] == ["cf-uuid"]
    metrics_create = prisma.db.litellm_dailyguardrailmetrics.upsert.call_args.kwargs["data"]["create"]
    assert (metrics_create["guardrail_id"], metrics_create["requests_evaluated"]) == ("cf-uuid", 1)


@pytest.mark.asyncio
async def test_batch_of_only_not_run_entries_writes_no_metrics_row():
    prisma = _prisma()

    await process_spend_logs_guardrail_usage(prisma, [_payload("r1", guardrail_status="not_run")])

    assert prisma.db.litellm_dailyguardrailmetrics.upsert.call_count == 0
    index_rows = prisma.db.litellm_spendlogguardrailindex.create_many.call_args.kwargs["data"]
    assert [row["request_id"] for row in index_rows] == ["r1"]


def _fan_out_payload(request_id: str, guardrail_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "startTime": datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc),
        "team_id": "team-a",
        "api_key": "hashed-key-1",
        "metadata": json.dumps(
            {"guardrail_information": [{"guardrail_id": gid, "guardrail_status": "success"} for gid in guardrail_ids]}
        ),
    }


def _index_rows_written(prisma: MagicMock) -> list[tuple[str, str]]:
    return [
        (row["request_id"], row["guardrail_id"])
        for call in prisma.db.litellm_spendlogguardrailindex.create_many.call_args_list
        for row in call.kwargs["data"]
    ]


@pytest.mark.asyncio
async def test_index_rows_are_written_in_row_bounded_statements(monkeypatch):
    """
    LIT-5931: the drain caps logs, not logs x guardrails, so a fan-out must be
    split into statements the query engine can afford instead of one create_many.
    """
    monkeypatch.setattr(usage_tracking, "SPEND_LOG_WRITE_BATCH_MAX_ROWS", 100)
    prisma = _prisma()
    guardrail_ids = tuple(f"guard-{i}" for i in range(50))
    logs = [_fan_out_payload(f"r{i}", guardrail_ids) for i in range(5)]

    await process_spend_logs_guardrail_usage(prisma, logs, pending=PendingRollups())

    statements = prisma.db.litellm_spendlogguardrailindex.create_many.call_args_list
    assert [len(call.kwargs["data"]) for call in statements] == [100, 100, 50]
    assert all(call.kwargs["skip_duplicates"] is True for call in statements)
    assert _index_rows_written(prisma) == [(f"r{i}", gid) for i in range(5) for gid in guardrail_ids]


@pytest.mark.asyncio
async def test_one_failing_index_statement_does_not_drop_the_others_or_the_rollup(monkeypatch):
    monkeypatch.setattr(usage_tracking, "SPEND_LOG_WRITE_BATCH_MAX_ROWS", 100)
    prisma = _prisma()
    prisma.db.litellm_spendlogguardrailindex.create_many.side_effect = [None, httpx.ReadTimeout("ambiguous"), None]
    guardrail_ids = tuple(f"guard-{i}" for i in range(50))
    logs = [_fan_out_payload(f"r{i}", guardrail_ids) for i in range(5)]

    await process_spend_logs_guardrail_usage(prisma, logs, pending=PendingRollups())

    assert prisma.db.litellm_spendlogguardrailindex.create_many.await_count == 3
    assert prisma.db.litellm_dailyguardrailmetrics.upsert.await_count == len(guardrail_ids)


@pytest.mark.asyncio
async def test_disabled_log_deliveries_do_not_enable_guardrail_analytics():
    prisma = _prisma()
    row = _payload("lean", usage={"topicPolicyUnits": 2})
    row["accounting_payload"] = {"request_id": "lean"}
    row["accounting_log_disabled"] = True
    await process_spend_logs_guardrail_usage(prisma, [row], pending=PendingRollups())
    prisma.db.litellm_dailyguardrailmetrics.upsert.assert_not_awaited()
    prisma.db.litellm_spendlogguardrailindex.create_many.assert_not_awaited()
    prisma.db.litellm_dailyguardrailusageunits.upsert.assert_not_awaited()
