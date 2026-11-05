import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.accounting.projection import project
from litellm.accounting.snapshot import build_accounting_payload
from litellm.constants import BILLED_RETRY_ATTEMPT_METADATA_KEY, INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.proxy_server import proxy_logging_obj
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import PrismaClient
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN, ModelResponse, Usage

_NOW: Final = datetime(2026, 11, 2, tzinfo=timezone.utc)
_FIELDS: Final = TypeAdapter(dict[str, object])


@pytest.mark.parametrize("status", ["success", "failure"])
@pytest.mark.parametrize("origin", [None, "auto_router", "billed_retry"])
@pytest.mark.parametrize("spend", [0.0, 0.25])
@pytest.mark.asyncio
async def test_daily_projection_counts_requests_only_for_the_caller(
    status: str, origin: str | None, spend: float
) -> None:
    payload: Final = get_logging_payload(
        kwargs={
            "model": "projection-test",
            "custom_llm_provider": "openai",
            "call_type": "acompletion",
            "litellm_call_id": "projection-request",
            "litellm_params": {"metadata": {"user_api_key": "key", "user_api_key_user_id": "user", "status": status}},
        },
        response_obj=ModelResponse(
            id="projection-request", model="projection-test", usage=Usage(prompt_tokens=10, completion_tokens=5)
        ),
        start_time=_NOW,
        end_time=_NOW,
    )
    metadata: Final = {
        **_FIELDS.validate_json(payload["metadata"]),
        **({BILLED_RETRY_ATTEMPT_METADATA_KEY: True} if origin == "billed_retry" else {}),
        **({INTERNAL_CALL_ORIGIN_METADATA_KEY: origin} if origin not in (None, "billed_retry") else {}),
    }
    row: Final = {**payload, "spend": spend, "metadata": json.dumps(metadata), "request_duration_ms": 25}
    accounting_input: Final[SpendLogsPayload] = {**row}
    accounting: Final = build_accounting_payload(
        accounting_input,
        entities={"api_key": payload["api_key"], "user": payload["user"]},
        model_access_groups=(),
        windows=(),
        router=lambda: None,
    )
    restored: Final = _FIELDS.validate_json(json.dumps(accounting, default=str))
    stored_daily: Final = next(item for item in project(restored) if item.table == "LiteLLM_DailyUserSpend")
    caller: Final = origin is None
    timed: Final = origin != "auto_router" and status == "success"
    assert dict(stored_daily.increments) == {
        "spend": spend,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "api_requests": int(caller),
        "successful_requests": int(caller and status == "success"),
        "failed_requests": int(caller and status == "failure"),
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "compression_saved_tokens": 0,
        "compression_savings_spend": 0.0,
        "prompt_caching_savings_spend": 0.0,
        "gateway_injected_caching_savings_spend": 0.0,
        "autorouter_savings_spend": 0.0,
        "total_response_time_ms": 25 if timed else 0,
        "timed_requests": int(timed),
    }
    assert dict(stored_daily.key)["date"] == "2026-11-02"
    assert dict(stored_daily.key)["api_key"] == payload["api_key"]
    assert not {"messages", "response", "proxy_server_request", "requester_ip_address"}.intersection(accounting)


def test_projection_uses_frozen_savings_and_keeps_duplicate_entity_tags() -> None:
    row: Final[Mapping[str, object]] = {
        "request_id": "saved",
        "startTime": _NOW,
        "api_key": "key",
        "user": "user",
        "model": "unpriced-projection-test",
        "model_group": "group",
        "custom_llm_provider": "openai",
        "request_tags": '["tag", "tag"]',
        "spend": 0.5,
        "metadata": json.dumps(
            {
                "accounting_savings": {
                    "compression": 1.0,
                    "prompt_caching": -2.0,
                    "gateway_injected_caching": 3.0,
                    "autorouter": 4.0,
                },
                "autorouter_savings": 9000,
            }
        ),
    }
    contributions: Final = project(row)
    tags: Final = tuple(item for item in contributions if item.table == "LiteLLM_TagTable")
    daily_tags: Final = tuple(item for item in contributions if item.table == "LiteLLM_DailyTagSpend")
    assert sum(dict(item.increments)["spend"] for item in tags) == 1.0
    assert len(daily_tags) == 1
    assert {key: value for key, value in daily_tags[0].increments if key.endswith("savings_spend")} == {
        "compression_savings_spend": 1.0,
        "prompt_caching_savings_spend": -2.0,
        "gateway_injected_caching_savings_spend": 3.0,
        "autorouter_savings_spend": 4.0,
    }


def test_projection_carries_windows_and_does_not_reprice_guardrails() -> None:
    row: Final = {
        "request_id": "guarded",
        "startTime": _NOW,
        "api_key": "key",
        "spend": 0.75,
        "metadata": json.dumps(
            {
                "accounting_windows": [
                    {
                        "entity_type": "key",
                        "entity_id": "key",
                        "window_duration": "1d",
                        "window_start": _NOW.isoformat(),
                    }
                ],
                "guardrail_information": [
                    {
                        "guardrail_id": "guard",
                        "guardrail_usage": {"text": 3, "images": 2},
                        "guardrail_cost_by_unit": {"text": 0.3, "images": None},
                    }
                ],
            }
        ),
    }
    contributions: Final = project(row)
    window: Final = next(item for item in contributions if item.table == "LiteLLM_BudgetWindowSpend")
    units: Final = {
        dict(item.key)["usage_unit"]: dict(item.increments)
        for item in contributions
        if item.table == "LiteLLM_DailyGuardrailUsageUnits"
    }
    assert dict(window.increments) == {"spend": 0.75}
    assert dict(window.values)["window_start"] == _NOW.isoformat()
    assert units == {
        "text": {"units": 3, "cost": 0.3, "untracked_units": 0},
        "images": {"units": 2, "cost": 0.0, "untracked_units": 2},
    }


@pytest.mark.asyncio
async def test_a_request_is_charged_once_to_every_entity_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    client: Final = PrismaClient(database_url="postgresql://unused", proxy_logging_obj=proxy_logging_obj)
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(client, "spend_log_transactions", [])
    charged: Final = await DBSpendUpdateWriter().update_database(
        token="sk-projection",
        user_id="user",
        end_user_id="end-user",
        team_id="team",
        org_id="org",
        project_id="project",
        kwargs={
            "model": "projection-test",
            "custom_llm_provider": "openai",
            "call_type": "acompletion",
            "litellm_call_id": "entity-request",
            "agent_id": "agent",
            "user": "end-user",
            "litellm_params": {
                "metadata": {
                    "user_api_key": "sk-projection",
                    "user_api_key_user_id": "user",
                    "user_api_key_team_id": "team",
                    "user_api_key_org_id": "org",
                    "user_api_key_project_id": "project",
                    "user_api_key_end_user_id": "end-user",
                    "tags": ["tag", "tag"],
                },
                "user": "end-user",
            },
        },
        completion_response=ModelResponse(
            id="entity-request", model="projection-test", usage=Usage(prompt_tokens=10, completion_tokens=5)
        ),
        start_time=_NOW,
        end_time=_NOW,
        response_cost=0.25,
    )

    assert charged
    assert len(client.spend_log_transactions) == 1
    projected: Final = project(
        _FIELDS.validate_json(json.dumps(client.spend_log_transactions[0]["accounting_payload"], default=str))
    )
    spend_by_entity: Final = {
        (item.table, item.key): dict(item.increments)["spend"]
        for item in projected
        if not item.table.startswith("LiteLLM_Daily")
    }
    assert spend_by_entity == {
        ("LiteLLM_VerificationToken", (("token", proxy_server.hash_token("sk-projection")),)): 0.25,
        ("LiteLLM_UserTable", (("user_id", "user"),)): 0.25,
        ("LiteLLM_TeamTable", (("team_id", "team"),)): 0.25,
        ("LiteLLM_OrganizationTable", (("organization_id", "org"),)): 0.25,
        ("LiteLLM_EndUserTable", (("user_id", "end-user"),)): 0.25,
        ("LiteLLM_AgentsTable", (("agent_id", "agent"),)): 0.25,
        ("LiteLLM_ProjectTable", (("project_id", "project"),)): 0.25,
        ("LiteLLM_TagTable", (("tag_name", "tag"),)): 0.25,
        ("LiteLLM_TeamMembership", (("team_id", "team"), ("user_id", "user"))): 0.25,
        ("LiteLLM_OrganizationMembership", (("organization_id", "org"), ("user_id", "user"))): 0.25,
    }
    daily: Final = {item.table: dict(item.increments) for item in projected if item.table.startswith("LiteLLM_Daily")}
    assert set(daily) == {
        "LiteLLM_DailyUserSpend",
        "LiteLLM_DailyTeamSpend",
        "LiteLLM_DailyOrganizationSpend",
        "LiteLLM_DailyEndUserSpend",
        "LiteLLM_DailyAgentSpend",
        "LiteLLM_DailyTagSpend",
        "LiteLLM_DailyModelUsage",
    }
    assert all(row["spend"] == 0.25 and row["prompt_tokens"] == 10 for row in daily.values())
    assert daily["LiteLLM_DailyModelUsage"]["request_count"] == 1
    assert daily["LiteLLM_DailyUserSpend"]["api_requests"] == 1


@pytest.mark.parametrize("served_id", ["requested", "fallback", "missing"])
@pytest.mark.asyncio
async def test_projection_preserves_fallback_narrowed_model_access_groups(served_id: str) -> None:
    from litellm import Router
    from litellm.proxy.db.db_spend_update_writer import debitable_model_access_groups

    router: Final = Router(
        model_list=[
            {
                "model_name": "first",
                "litellm_params": {"model": "openai/projection-test", "api_key": "unused"},
                "model_info": {"id": "requested", "access_groups": ["first-group"]},
            },
            {
                "model_name": "second",
                "litellm_params": {"model": "openai/projection-test", "api_key": "unused"},
                "model_info": {"id": "fallback", "access_groups": ["second-group", "unattributed"]},
            },
        ]
    )
    attributed: Final = ("first-group", "second-group", "first-group")
    frozen_groups: Final = debitable_model_access_groups(attributed, served_id, router)
    projected: Final = project(
        {"spend": 0.5, "metadata": json.dumps({"accounting_model_access_groups": frozen_groups})}
    )
    assert (
        sorted((dict(item.key)["access_group_name"], dict(item.increments)["spend"]) for item in projected)
        == {
            "requested": [("first-group", 0.5)],
            "fallback": [("second-group", 0.5)],
            "missing": [("first-group", 0.5), ("second-group", 0.5)],
        }[served_id]
    )
    assert "unattributed" not in frozen_groups


def test_guardrail_projection_skips_unmeasured_counters() -> None:
    row: Final = {
        "request_id": "guarded",
        "startTime": _NOW,
        "metadata": json.dumps(
            {
                "guardrail_information": [
                    {"guardrail_usage": {"text": 3}},
                    {
                        "guardrail_id": "guard",
                        "guardrail_usage": {"zero": 0, "negative": -1, "bool": True, "unknown": None, "text": 2},
                    },
                ]
            }
        ),
    }
    units: Final = tuple(item for item in project(row) if item.table == "LiteLLM_DailyGuardrailUsageUnits")
    assert len(units) == 1
    assert dict(units[0].key)["usage_unit"] == "text"
    assert dict(units[0].increments) == {"units": 2, "cost": 0.0, "untracked_units": 2}


def test_lean_row_without_model_does_not_fabricate_daily_entity_usage() -> None:
    projected: Final = project(
        {
            "request_id": "lean",
            "api_key": "key",
            "startTime": _NOW,
            "spend": 0.5,
            "metadata": {"accounting_entities": {"api_key": "key"}},
        }
    )
    assert tuple(item.table for item in projected) == ("LiteLLM_VerificationToken", "LiteLLM_DailyModelUsage")


@pytest.mark.parametrize(
    "cost_fields",
    [
        {"guardrail_cost_in_spend": False, "guardrail_cost_by_unit": {"text": 1.0}},
        {"guardrail_cost_by_unit": {"text": -1.0}},
    ],
)
def test_report_only_or_invalid_guardrail_prices_remain_untracked(cost_fields: Mapping[str, object]) -> None:
    row: Final = {
        "request_id": "guarded",
        "startTime": _NOW,
        "request_tags": '["task:debugging"]',
        "metadata": json.dumps(
            {
                "guardrail_information": [
                    {"guardrail_id": "guard", "guardrail_usage": {"text": 2}, **cost_fields},
                ]
            }
        ),
    }
    contributions: Final = project(row)
    guardrail: Final = next(item for item in contributions if item.table == "LiteLLM_DailyGuardrailUsageUnits")
    model: Final = next(item for item in contributions if item.table == "LiteLLM_DailyModelUsage")
    assert dict(guardrail.increments) == {"units": 2, "cost": 0.0, "untracked_units": 2}
    assert dict(model.key)["task_type"] == "debugging"


@pytest.mark.parametrize("organization_id", [None, "", "charged-org"])
def test_daily_organization_uses_frozen_charged_identity(organization_id: str | None) -> None:
    row: Final[Mapping[str, object]] = {
        "startTime": _NOW,
        "api_key": "key",
        "model": "projection-test",
        "model_group": "group",
        "custom_llm_provider": "openai",
        "organization_id": "stored-org",
        "spend": 0.25,
        "metadata": {"accounting_entities": {"organization_id": organization_id}},
    }
    daily: Final = tuple(item for item in project(row) if item.table == "LiteLLM_DailyOrganizationSpend")
    if organization_id is None:
        assert daily == ()
    else:
        assert len(daily) == 1
        assert dict(daily[0].key)["organization_id"] == organization_id
        assert dict(daily[0].increments)["spend"] == 0.25


@pytest.mark.parametrize("start", [_NOW, "2026-11-02T12:00:00+00:00", "2026-11-02 12:00:00+00:00"])
def test_every_daily_row_uses_the_calendar_date_of_the_request(start: object) -> None:
    row: Final[Mapping[str, object]] = {
        "request_id": "dated",
        "startTime": start,
        "api_key": "key",
        "user": "user",
        "team_id": "team",
        "model": "unpriced-projection-test",
        "model_group": "group",
        "custom_llm_provider": "openai",
        "spend": 0.5,
        "metadata": "{}",
    }
    dates: Final = {
        item.table: dict(item.key)["date"] for item in project(row) if item.table.startswith("LiteLLM_Daily")
    }
    assert {"LiteLLM_DailyUserSpend", "LiteLLM_DailyModelUsage"} <= dates.keys()
    assert set(dates.values()) == {"2026-11-02"}


def test_unparseable_start_time_skips_only_guardrail_usage() -> None:
    projected: Final = project(
        {
            "request_id": "bad-time",
            "startTime": "yesterday at noon",
            "spend": 0.5,
            "metadata": {
                "accounting_entities": {"api_key": "key"},
                "guardrail_information": [{"guardrail_id": "guard", "guardrail_usage": {"text": 2}}],
            },
        }
    )
    tables: Final = {item.table for item in projected}
    assert "LiteLLM_VerificationToken" in tables
    assert "LiteLLM_DailyGuardrailUsageUnits" not in tables


def test_guardrail_usage_is_dated_in_utc() -> None:
    projected: Final = project(
        {
            "request_id": "offset",
            "startTime": "2026-11-02T22:30:00-05:00",
            "metadata": {"guardrail_information": [{"guardrail_id": "guard", "guardrail_usage": {"text": 1}}]},
        }
    )
    assert [dict(item.key)["date"] for item in projected if item.table == "LiteLLM_DailyGuardrailUsageUnits"] == [
        "2026-11-03"
    ]


@pytest.mark.parametrize(
    ("untrusted", "metadata"),
    [
        ({"request_tags": json.dumps(["x" * 16384, "normal"])}, {}),
        ({"end_user": "x" * 3072}, {}),
        ({}, {"compression_savings": {"tokens_saved": 2**63}}),
    ],
)
def test_untrusted_analytics_values_never_block_the_charge(
    untrusted: Mapping[str, str], metadata: Mapping[str, object]
) -> None:
    row: Final[Mapping[str, object]] = {
        "request_id": "untrusted",
        "startTime": _NOW,
        "api_key": "key",
        "user": "user",
        "team_id": "team",
        "model": "unpriced-projection-test",
        "model_group": "group",
        "custom_llm_provider": "openai",
        "spend": 0.5,
        **untrusted,
        "metadata": {
            **metadata,
            "accounting_entities": {
                "api_key": "key",
                "user": "user",
                "team_id": "team",
                "end_user": untrusted.get("end_user"),
            },
        },
    }
    contributions: Final = project(row)
    texts: Final = tuple(value for item in contributions for _, value in (*item.key, *item.values) if value)
    numbers: Final = tuple(value for item in contributions for _, value in item.increments)
    assert {"LiteLLM_VerificationToken", "LiteLLM_TeamTable", "LiteLLM_UserTable"} <= {
        item.table for item in contributions
    }
    assert "LiteLLM_DailyUserSpend" in {item.table for item in contributions}
    assert all(len(text.encode()) <= 2000 for text in texts)
    assert all(abs(number) <= 2**40 for number in numbers)


def test_stored_payload_keeps_guardrail_billing_but_not_guardrail_content() -> None:
    payload: Final = get_logging_payload(
        kwargs={
            "model": "projection-test",
            "custom_llm_provider": "openai",
            "call_type": "acompletion",
            "litellm_call_id": "guardrail-request",
            "litellm_params": {"metadata": {"user_api_key": "key", "user_api_key_user_id": "user"}},
        },
        response_obj=ModelResponse(
            id="guardrail-request", model="projection-test", usage=Usage(prompt_tokens=10, completion_tokens=5)
        ),
        start_time=_NOW,
        end_time=_NOW,
    )
    guardrail: Final = {
        "guardrail_name": "pii",
        "guardrail_usage": {"requests": 1},
        "guardrail_cost_by_unit": {"requests": 0.25},
        "guardrail_response": "card 4111 1111 1111 1111",
        "match_details": [{"matched": "4111 1111 1111 1111"}],
    }
    headroom: Final = {
        "guardrail_name": "compress",
        "guardrail_provider": "headroom",
        "guardrail_response": {"tokens_saved": 7, "compressed_prompt": "card 4111 1111 1111 1111"},
    }
    metadata: Final = {**_FIELDS.validate_json(payload["metadata"]), "guardrail_information": [guardrail, headroom]}
    accounting_input: Final[SpendLogsPayload] = {
        **payload,
        "spend": 0.5,
        "metadata": json.dumps(metadata),
        "request_tags": '["nul\x00tag"]',
    }
    accounting: Final = build_accounting_payload(
        accounting_input, entities={"api_key": "key"}, model_access_groups=(), windows=(), router=lambda: None
    )
    stored: Final = json.dumps(accounting, default=str)
    restored: Final = _FIELDS.validate_json(stored)
    usage_rows: Final = tuple(item for item in project(restored) if item.table == "LiteLLM_DailyGuardrailUsageUnits")
    assert "4111" not in stored
    assert "\\u0000" not in stored
    assert [dict(item.increments) for item in usage_rows] == [{"units": 1, "cost": 0.25, "untracked_units": 0}]
    daily_user: Final = next(item for item in project(restored) if item.table == "LiteLLM_DailyUserSpend")
    assert dict(daily_user.increments)["compression_saved_tokens"] == 7


def test_caller_supplied_savings_beyond_json_range_are_clamped_and_never_drop_the_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setitem(
        litellm.model_cost,
        "savings-overflow-test",
        {"input_cost_per_token": 0.002, "output_cost_per_token": 0.004, "litellm_provider": "openai", "mode": "chat"},
    )
    payload: Final[SpendLogsPayload] = {
        **get_logging_payload(
            kwargs={
                "model": "savings-overflow-test",
                "custom_llm_provider": "openai",
                "call_type": "acompletion",
                "litellm_call_id": "overflow-request",
                "litellm_params": {"metadata": {"user_api_key": "key"}},
            },
            response_obj=ModelResponse(
                id="overflow-request", model="savings-overflow-test", usage=Usage(prompt_tokens=10, completion_tokens=5)
            ),
            start_time=_NOW,
            end_time=_NOW,
        ),
        "spend": 0.5,
        "metadata": json.dumps({"compression_savings": {"tokens_saved": 10**400}}),
    }
    accounting: Final = build_accounting_payload(
        payload, entities={"api_key": "key"}, model_access_groups=(), windows=(), router=lambda: None
    )
    restored: Final = _FIELDS.validate_json(json.dumps(accounting, default=str))
    assert dict(restored["metadata"])["accounting_savings"]["compression"] == 0.0
    assert dict(restored["metadata"])["compression_savings"] == {"tokens_saved": 2**53 - 1}
    assert dict(restored["metadata"])["compression_savings"] == {"tokens_saved": 2**53 - 1}
    assert [
        dict(item.increments)["spend"] for item in project(restored) if item.table == "LiteLLM_VerificationToken"
    ] == [0.5]


def _guarded_row(
    request_id: str | None,
    usage: Mapping[str, object],
    *,
    team_id: str | None = "team-a",
    api_key: str = "hashed-key-1",
    status: str = "success",
    cost_by_unit: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    entry: Final = {
        "guardrail_id": "bedrock-guard",
        "guardrail_status": status,
        "guardrail_usage": usage,
        **({} if cost_by_unit is None else {"guardrail_cost_by_unit": cost_by_unit}),
    }
    return {
        "request_id": request_id,
        "startTime": "2026-08-17T12:00:00+00:00",
        "team_id": team_id,
        "api_key": api_key,
        "metadata": {"guardrail_information": [entry]},
    }


def _guardrail_unit_rows(*rows: Mapping[str, object]) -> Mapping[tuple[str, ...], Mapping[str, int | float]]:
    from litellm.accounting.delivery_projection import project_deliveries
    from litellm.repositories.accounting_delivery_repository import DeliveryProjection

    return {
        tuple(value for _, value in item.key): dict(item.increments)
        for item in project_deliveries(DeliveryProjection(row) for row in rows)
        if item.table == "LiteLLM_DailyGuardrailUsageUnits"
    }


def test_guardrail_units_sum_per_guardrail_team_key_and_date_whatever_the_verdict() -> None:
    """LIT-5650: Bedrock bills blocked invocations like passed ones, and a team-less request gets its own row."""
    rows: Final = _guardrail_unit_rows(
        _guarded_row("r1", {"topicPolicyUnits": 1, "contentPolicyUnits": 1}),
        _guarded_row("r2", {"topicPolicyUnits": 1, "contentPolicyUnits": 2}, status="guardrail_intervened"),
        _guarded_row("r3", {"topicPolicyUnits": 1}, team_id=None, api_key="hashed-key-2"),
        _guarded_row(None, {"topicPolicyUnits": 9}),
    )

    assert {key: counters["units"] for key, counters in rows.items()} == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "topicPolicyUnits"): 2,
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "contentPolicyUnits"): 3,
        ("bedrock-guard", "2026-08-17", "", "hashed-key-2", "topicPolicyUnits"): 1,
    }


def test_guardrail_cost_sums_per_counter_and_unpriced_units_stay_untracked() -> None:
    """LIT-5652: priced units keep their subtotal while units no price covers are counted as untracked, never free."""
    rows: Final = _guardrail_unit_rows(
        _guarded_row(
            "r1",
            {"contentPolicyUnits": 1000, "wordPolicyUnits": 50},
            cost_by_unit={"contentPolicyUnits": 0.15, "wordPolicyUnits": 0.0},
        ),
        _guarded_row(
            "r2", {"contentPolicyUnits": 700, "someFutureCounter": 3}, cost_by_unit={"someFutureCounter": None}
        ),
        _guarded_row("r3", {"contentPolicyUnits": 300}),
    )

    by_unit: Final = {key[-1]: counters for key, counters in rows.items()}
    assert by_unit["contentPolicyUnits"] == {"units": 2000, "cost": pytest.approx(0.15), "untracked_units": 1000}
    assert by_unit["wordPolicyUnits"] == {"units": 50, "cost": 0.0, "untracked_units": 0}
    assert by_unit["someFutureCounter"] == {"units": 3, "cost": 0.0, "untracked_units": 3}


@pytest.mark.parametrize(
    ("overrides", "counted"),
    [
        ({}, (1, 1, 0)),
        ({"status": "failure"}, (1, 0, 1)),
        ({INTERNAL_CALL_ORIGIN_METADATA_KEY: "health_check"}, None),
        ({INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN}, (0, 0, 0)),
    ],
    ids=["success", "failure", "internal_call", "background_billing"],
)
def test_model_usage_counts_spend_always_but_requests_only_for_the_caller(
    overrides: Mapping[str, object], counted: tuple[int, int, int] | None
) -> None:
    """A background response's billing row adds its spend and tokens but not a second request, and other internal
    calls are left out of model insights entirely."""
    metadata: Final = {key: value for key, value in overrides.items() if key != "status"}
    row: Final = {
        "request_id": "r",
        "startTime": "2026-09-28T01:02:03Z",
        "spend": 0.25,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "model": "",
        "model_group": None,
        "custom_llm_provider": None,
        "status": overrides.get("status", "success"),
        "request_tags": ["task:debugging"],
        "metadata": metadata,
    }

    usage: Final = tuple(item for item in project(row) if item.table == "LiteLLM_DailyModelUsage")

    if counted is None:
        assert usage == ()
        return
    (only,) = usage
    assert dict(only.key) == {
        "date": "2026-09-28",
        "model_group": "unknown",
        "model": "unknown",
        "custom_llm_provider": "unknown",
        "task_type": "debugging",
    }
    assert dict(only.increments) == {
        "spend": 0.25,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "request_count": counted[0],
        "successful_requests": counted[1],
        "failed_requests": counted[2],
    }


def _daily_user_row(payload: Mapping[str, object], status: str) -> Mapping[str, object]:
    metadata: Final = {**_FIELDS.validate_json(str(payload["metadata"])), "status": status}
    row: Final[SpendLogsPayload] = {
        "endTime": payload["startTime"],
        "model_id": None,
        "request_tags": "[]",
        "agent_id": None,
        **payload,
        "status": status,
        "metadata": json.dumps(metadata),
    }
    accounting: Final = build_accounting_payload(
        row,
        entities={"api_key": payload["api_key"], "user": payload["user"]},
        model_access_groups=(),
        windows=(),
        router=lambda: None,
    )
    daily: Final = next(
        item
        for item in project(_FIELDS.validate_json(json.dumps(accounting, default=str)))
        if item.table == "LiteLLM_DailyUserSpend"
    )
    return {**dict(daily.key), **dict(daily.increments)}


def test_daily_transaction_carries_compression_saved_tokens():
    """
    The daily transaction built from a SpendLog payload must carry
    compression_saved_tokens summed from both the native compression_savings
    metadata key and Headroom guardrail entries, alongside the cache token
    fields extracted from usage_object.
    """

    metadata = {
        "usage_object": {"cache_read_input_tokens": 40, "cache_creation_input_tokens": 15},
        "litellm_gateway_injected_cache": "dep-of-the-compression-row",
        "compression_savings": {
            "tokens_before": 12000,
            "tokens_after": 5000,
            "tokens_saved": 7000,
            "source": "compression_interception",
        },
        "guardrail_information": [
            {
                "guardrail_name": "headroom-compressor",
                "guardrail_provider": "headroom",
                "guardrail_status": "success",
                "guardrail_response": {"tokens_before": 1000, "tokens_after": 400, "tokens_saved": 600},
            }
        ],
    }
    payload = {
        "request_id": "req-compression-1",
        "user": "test-user",
        "startTime": "2026-07-17T00:00:00",
        "api_key": "test-key",
        "model": "claude-sonnet-5",
        "custom_llm_provider": "anthropic",
        "model_group": "claude-sonnet-5",
        "model_id": "dep-of-the-compression-row",
        "call_type": "anthropic_messages",
        "prompt_tokens": 5000,
        "completion_tokens": 10,
        "spend": 0.05,
        "metadata": json.dumps(metadata),
    }

    transaction = _daily_user_row(payload, "success")

    assert transaction is not None
    assert transaction["compression_saved_tokens"] == 7600
    assert transaction["cache_read_input_tokens"] == 40
    assert transaction["cache_creation_input_tokens"] == 15

    model_info = litellm.get_model_info(model="claude-sonnet-5", custom_llm_provider="anthropic")
    input_cost = model_info["input_cost_per_token"] or 0.0
    cache_read_cost = model_info.get("cache_read_input_token_cost") or input_cost
    cache_write_cost = model_info.get("cache_creation_input_token_cost") or input_cost
    assert transaction["compression_savings_spend"] == pytest.approx(7600 * input_cost)
    assert transaction["prompt_caching_savings_spend"] == pytest.approx(
        40 * max(input_cost - cache_read_cost, 0.0) - 15 * (cache_write_cost - input_cost)
    )
    assert transaction["compression_savings_spend"] > 0
    assert transaction["prompt_caching_savings_spend"] > 0


@pytest.mark.parametrize(
    "estimate, recorded_savings, expected",
    [
        pytest.param(None, None, -0.005, id="plain-classifier-cost"),
        pytest.param({"version": 1, "status": "unknown"}, None, 0.0, id="unknown"),
        pytest.param({"version": 2, "status": "unknown"}, None, 0.0, id="unknown-v2"),
        pytest.param({"version": 1, "status": "unknown"}, -0.003, 0.0, id="unknown-stale-value"),
        pytest.param({"version": 0, "status": "estimated"}, -0.003, 0.0, id="unsupported-version"),
        pytest.param({"version": 1, "status": "estimated"}, -0.003, -0.003, id="estimated"),
        pytest.param(None, -0.003, -0.003, id="legacy"),
    ],
)
def test_daily_transaction_compression_saved_tokens_zero_when_absent(
    estimate: dict[str, object] | None,
    recorded_savings: float | None,
    expected: float,
) -> None:
    """Requests without any compression metadata produce a zero count."""

    payload = {
        "request_id": "req-no-compression",
        "user": "test-user",
        "startTime": "2026-07-17T00:00:00",
        "api_key": "test-key",
        "model": "claude-sonnet-5",
        "custom_llm_provider": "anthropic",
        "model_group": "claude-sonnet-5",
        "call_type": "anthropic_messages",
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "spend": 0.01,
        "metadata": json.dumps(
            {
                "usage_object": {"prompt_tokens": 100, "completion_tokens": 10},
                "routing_decision": {"savings_baseline_model": "anthropic/claude-sonnet-5", "classifier_cost": 0.005},
                "autorouter_savings": recorded_savings,
                "autorouter_savings_estimate": estimate,
            }
        ),
    }

    transaction = _daily_user_row(payload, "success")

    assert transaction is not None
    assert transaction["compression_saved_tokens"] == 0
    assert transaction["compression_savings_spend"] == 0
    assert transaction["prompt_caching_savings_spend"] == 0
    assert transaction["spend"] == 0.01
    assert transaction["autorouter_savings_spend"] == expected


def test_daily_transaction_internal_call_keeps_spend_but_not_request_counts():
    """Internal sub-calls (auto-router classifier, shadow eval's shadow and judge) bill
    spend and tokens to the key but are not requests the caller made: api_requests,
    successful_requests, and autorouter_savings_spend must all stay zero for them."""

    def _payload(metadata: dict) -> dict:
        return {
            "request_id": "req-internal-1",
            "user": "test-user",
            "startTime": "2026-08-11T00:00:00",
            "api_key": "test-key",
            "model": "claude-sonnet-5",
            "custom_llm_provider": "anthropic",
            "model_group": "claude-sonnet-5",
            "call_type": "acompletion",
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "spend": 0.05,
            "metadata": json.dumps(metadata),
        }

    internal = _daily_user_row(_payload({"internal_call_origin": "shadow_eval_judge"}), "success")
    user_sent = _daily_user_row(_payload({}), "success")

    assert internal is not None and user_sent is not None
    assert internal["spend"] == 0.05
    assert internal["prompt_tokens"] == 100
    assert internal["api_requests"] == 0
    assert internal["successful_requests"] == 0
    assert internal["failed_requests"] == 0
    assert internal["autorouter_savings_spend"] == 0.0
    assert user_sent["api_requests"] == 1
    assert user_sent["successful_requests"] == 1


def _response_time_payload(request_duration_ms: object, metadata: dict | None = None) -> dict:
    return {
        "request_id": "req-timed-1",
        "user": "test-user",
        "startTime": "2026-09-15T00:00:00",
        "api_key": "test-key",
        "model": "gpt-5.5",
        "custom_llm_provider": "openai",
        "model_group": "gpt-5.5",
        "call_type": "acompletion",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "spend": 0.01,
        "request_duration_ms": request_duration_ms,
        "metadata": json.dumps(metadata or {}),
    }


@pytest.mark.parametrize("request_duration_ms", [1234, 0])
def test_daily_transaction_rolls_up_response_time_for_successful_requests(request_duration_ms: int):
    """A successful user-sent request contributes its request_duration_ms to the daily
    response-time sum and counts as one timed request, including a 0 ms duration."""

    transaction = _daily_user_row(_response_time_payload(request_duration_ms), "success")

    assert transaction is not None
    assert transaction["total_response_time_ms"] == request_duration_ms
    assert transaction["timed_requests"] == 1


@pytest.mark.parametrize(
    ("request_status", "request_duration_ms", "metadata"),
    [
        ("failure", 1234, {}),
        ("success", None, {}),
        ("success", -5, {}),
        ("success", "1234", {}),
        ("success", 1234, {"internal_call_origin": "shadow_eval_judge"}),
    ],
    ids=["failed", "missing", "negative", "non_int", "internal_call"],
)
def test_daily_transaction_excludes_untimed_requests_from_response_time(
    request_status: str, request_duration_ms: object, metadata: dict
):
    """Failed, internal, and missing/invalid-duration requests never enter the response-time
    average: both the duration sum and the timed_requests denominator stay at zero."""

    transaction = _daily_user_row(_response_time_payload(request_duration_ms, metadata), request_status)

    assert transaction is not None
    assert transaction["total_response_time_ms"] == 0
    assert transaction["timed_requests"] == 0


@pytest.mark.parametrize(
    "injected_deployment, attributed",
    [
        pytest.param("dep-of-this-row", True, id="this-deployment-injected"),
        pytest.param("dep-of-a-sibling-leg", False, id="a-sibling-deployment-injected"),
        pytest.param("", True, id="injected-before-a-deployment-was-chosen"),
    ],
)
def test_caching_savings_are_attributed_to_the_deployment_that_was_injected(injected_deployment, attributed):
    """Retries, same-group failover and cross-model-group fallbacks all reuse one metadata
    bucket and one litellm_call_id, so a marker written by the leg that injected is
    visible to every sibling and nothing request-scoped can tell them apart.

    Naming the deployment it injected for is what keeps the credit on that leg: a row
    billed for a different deployment reads it as no injection, so no seam has to strip
    it and a deployment that injected nothing is never credited for the one that did.

    An injection that ran before any deployment was chosen, which is what the proxy does
    for prompt templates, is written into the payload every leg goes on to send, so it
    marks the request for all of them and each leg keeps the credit.
    """

    payload = {
        "request_id": "req-fallback-leg",
        "user": "test-user",
        "startTime": "2026-07-17T00:00:00",
        "api_key": "test-key",
        "model": "claude-sonnet-5",
        "custom_llm_provider": "anthropic",
        "model_group": "claude-sonnet-5",
        "model_id": "dep-of-this-row",
        "call_type": "anthropic_messages",
        "prompt_tokens": 5000,
        "completion_tokens": 10,
        "spend": 0.05,
        "metadata": json.dumps(
            {
                "usage_object": {"cache_read_input_tokens": 4242, "cache_creation_input_tokens": 1111},
                "litellm_gateway_injected_cache": injected_deployment,
            }
        ),
    }

    transaction = _daily_user_row(payload, "success")

    assert transaction is not None
    assert transaction["prompt_caching_savings_spend"] != 0.0
    assert (transaction["gateway_injected_caching_savings_spend"] != 0.0) is attributed


def test_daily_transaction_attributes_caching_savings_only_with_an_injection_marker():
    """Cached usage with no litellm_gateway_injected_cache marker is still a real saving.

    Client-sent cache_control and implicit provider caching leave no marker, so the row
    keeps the total the customer actually got while the gateway-attributed column stays
    empty, which is what separates what caching saved from what litellm can claim.
    """

    payload = {
        "request_id": "req-ungated-caching",
        "user": "test-user",
        "startTime": "2026-07-17T00:00:00",
        "api_key": "test-key",
        "model": "claude-sonnet-5",
        "custom_llm_provider": "anthropic",
        "model_group": "claude-sonnet-5",
        "call_type": "anthropic_messages",
        "prompt_tokens": 5000,
        "completion_tokens": 10,
        "spend": 0.05,
        "metadata": json.dumps(
            {"usage_object": {"cache_read_input_tokens": 4242, "cache_creation_input_tokens": 1111}}
        ),
    }

    transaction = _daily_user_row(payload, "success")

    assert transaction is not None
    assert transaction["cache_read_input_tokens"] == 4242
    assert transaction["cache_creation_input_tokens"] == 1111
    assert transaction["prompt_caching_savings_spend"] != 0.0
    assert transaction["gateway_injected_caching_savings_spend"] == 0.0
