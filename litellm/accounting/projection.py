from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.accounting.payload import AccountingMetadata, AccountingPayload
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import GuardrailCostByUnitEntry
from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, daily_spend_entity_ids
from litellm.proxy.db.model_insights_tasks import model_insight_task_type
from litellm.proxy.db.model_usage_rollup import date_from_start_time
from litellm.proxy.route_llm_request import ROUTE_ENDPOINT_MAPPING
from litellm.proxy.spend_tracking.compression_savings import extract_compression_saved_tokens
from litellm.proxy.spend_tracking.savings import extract_cache_creation_tokens, extract_cache_read_tokens
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN

_MAX_INDEXED_TEXT_BYTES: Final = 2000
_MAX_COUNTER: Final = 2**40
_NO_USAGE: Final[Mapping[str, object]] = MappingProxyType({})
_USAGE: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class Contribution:
    table: str
    key: tuple[tuple[str, str], ...]
    increments: tuple[tuple[str, int | float], ...]
    values: tuple[tuple[str, str | None], ...] = ()


def _text(value: str | None) -> str:
    return "" if value is None else value


def _spend(table: str, key: tuple[tuple[str, str], ...], spend: float) -> Contribution:
    return Contribution(table, key, (("spend", spend),))


def _entity_contributions(payload: AccountingPayload) -> tuple[Contribution, ...]:
    entities: Final = payload.metadata.entities
    identities: Final = (
        ("LiteLLM_VerificationToken", "token", entities.api_key),
        ("LiteLLM_UserTable", "user_id", entities.user),
        ("LiteLLM_UserTable", "user_id", entities.proxy_budget_name),
        ("LiteLLM_TeamTable", "team_id", entities.team_id),
        ("LiteLLM_OrganizationTable", "organization_id", entities.organization_id),
        ("LiteLLM_EndUserTable", "user_id", entities.end_user),
        ("LiteLLM_AgentsTable", "agent_id", entities.agent_id),
        ("LiteLLM_ProjectTable", "project_id", entities.project_id),
    )
    owners: Final = (
        ("LiteLLM_TeamMembership", "team_id", entities.team_id),
        ("LiteLLM_OrganizationMembership", "organization_id", entities.organization_id),
    )
    spend: Final = payload.spend
    return (
        *(
            _spend(table, ((column, identity),), spend)
            for table, column, identity in identities
            if identity is not None
        ),
        *(_spend("LiteLLM_TagTable", (("tag_name", tag),), spend) for tag in payload.request_tags if tag),
        *(
            _spend("LiteLLM_ModelAccessGroupBudgetTable", (("access_group_name", group),), spend)
            for group in payload.metadata.model_access_groups
        ),
        *(
            _spend(table, ((column, identity), ("user_id", entities.user)), spend)
            for table, column, identity in owners
            if entities.user is not None and identity is not None
        ),
    )


def _daily_counters(payload: AccountingPayload) -> tuple[tuple[str, int | float], ...]:
    metadata: Final = payload.metadata
    usage: Final = metadata.usage_object or _NO_USAGE
    internal: Final = bool(metadata.internal_call_origin)
    caller: Final = not internal and not metadata.billed_retry_attempt
    success: Final = metadata.status != "failure"
    duration: Final = payload.request_duration_ms
    timed: Final = not internal and success and duration is not None and duration >= 0
    return (
        ("spend", payload.spend),
        ("prompt_tokens", payload.prompt_tokens),
        ("completion_tokens", payload.completion_tokens),
        ("api_requests", int(caller)),
        ("successful_requests", int(caller and success)),
        ("failed_requests", int(caller and not success)),
        ("cache_read_input_tokens", extract_cache_read_tokens(usage)),
        ("cache_creation_input_tokens", extract_cache_creation_tokens(usage)),
        ("compression_saved_tokens", _compression_saved_tokens(metadata)),
        ("compression_savings_spend", metadata.savings.compression),
        ("prompt_caching_savings_spend", metadata.savings.prompt_caching),
        ("gateway_injected_caching_savings_spend", metadata.savings.gateway_injected_caching),
        ("autorouter_savings_spend", 0.0 if internal else metadata.savings.autorouter),
        ("total_response_time_ms", duration if timed and duration is not None else 0),
        ("timed_requests", int(timed)),
    )


def _compression_saved_tokens(metadata: AccountingMetadata) -> int:
    return extract_compression_saved_tokens(
        {
            "compression_savings": metadata.compression_savings,
            "guardrail_information": list(metadata.guardrail_information),
        }
    )


def _daily_contributions(payload: AccountingPayload) -> tuple[Contribution, ...]:
    date: Final = _date(payload)
    if date is None:
        return ()
    shared_key: Final = (
        ("date", date),
        ("api_key", _text(payload.api_key)),
        ("model", _text(payload.model)),
        ("custom_llm_provider", _text(payload.custom_llm_provider)),
        ("mcp_namespaced_tool_name", _text(payload.mcp_namespaced_tool_name)),
        ("endpoint", _text(ROUTE_ENDPOINT_MAPPING.get(_text(payload.call_type)))),
    )
    entity_ids: Final = {
        "user": payload.user,
        "team_id": payload.team_id,
        "organization_id": payload.metadata.entities.organization_id,
        "end_user": payload.end_user,
        "agent_id": payload.agent_id,
        "request_tags": list(payload.request_tags),
    }
    return tuple(_daily_rows(payload, entity_ids, shared_key, _daily_counters(payload)))


def _daily_rows(
    payload: AccountingPayload,
    entity_ids: Mapping[str, object],
    shared_key: tuple[tuple[str, str], ...],
    counters: tuple[tuple[str, int | float], ...],
) -> Iterator[Contribution]:
    for entity, table in DAILY_SPEND_TABLES.items():
        for identity in daily_spend_entity_ids(entity_ids, entity):
            yield Contribution(
                table.name,
                ((table.entity_id_column, _text(identity)), *shared_key),
                counters,
                (
                    ("model_group", payload.model_group),
                    *((("request_id", payload.request_id),) if table.carries_request_id else ()),
                ),
            )


def _window_contributions(payload: AccountingPayload) -> tuple[Contribution, ...]:
    return tuple(
        Contribution(
            "LiteLLM_BudgetWindowSpend",
            (
                ("entity_type", window.entity_type),
                ("entity_id", window.entity_id),
                ("window_duration", window.window_duration),
            ),
            (("spend", payload.spend),),
            (("window_start", window.window_start), ("started_at", _text(payload.start_time))),
        )
        for window in payload.metadata.windows
    )


def _model_contributions(payload: AccountingPayload) -> tuple[Contribution, ...]:
    date: Final = _date(payload)
    origin: Final = payload.metadata.internal_call_origin
    if date is None or (origin and origin != BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN):
        return ()
    model: Final = _text(payload.model) or "unknown"
    caller: Final = not origin
    success: Final = payload.status == "success"
    return (
        Contribution(
            "LiteLLM_DailyModelUsage",
            (
                ("date", date),
                ("model_group", _text(payload.model_group) or model),
                ("model", model),
                ("custom_llm_provider", _text(payload.custom_llm_provider) or "unknown"),
                ("task_type", model_insight_task_type(payload.request_tags)),
            ),
            (
                ("spend", payload.spend),
                ("prompt_tokens", payload.prompt_tokens),
                ("completion_tokens", payload.completion_tokens),
                ("request_count", int(caller)),
                ("successful_requests", int(caller and success)),
                ("failed_requests", int(caller and not success)),
            ),
        ),
    )


def _date(payload: AccountingPayload) -> str | None:
    return None if payload.start_time is None else date_from_start_time(payload.start_time)


def _utc_date(payload: AccountingPayload) -> str | None:
    if payload.start_time is None:
        return None
    try:
        parsed: Final = datetime.fromisoformat(payload.start_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (
        (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed)
        .astimezone(timezone.utc)
        .date()
        .isoformat()
    )


def _guardrail_contributions(payload: AccountingPayload) -> tuple[Contribution, ...]:
    date: Final = _utc_date(payload)
    entries: Final = payload.metadata.guardrail_information or payload.metadata.standard_logging_guardrail_information
    if not payload.request_id or date is None:
        return ()
    return tuple(chain.from_iterable(_guardrail_entry(payload, date, entry) for entry in entries))


def _guardrail_entry(payload: AccountingPayload, date: str, entry: Mapping[str, object]) -> tuple[Contribution, ...]:
    identity: Final = entry.get("guardrail_id") or entry.get("guardrail_name")
    usage: Final = entry.get("guardrail_usage")
    if not identity or not isinstance(usage, Mapping):
        return ()
    costs: Final = _guardrail_unit_costs(entry)
    return tuple(
        _guardrail_unit(payload, (str(identity), date, unit), units, None if costs is None else costs.get(unit))
        for unit, units in _USAGE.validate_python(usage).items()
        if isinstance(units, int) and not isinstance(units, bool) and units > 0
    )


def _guardrail_unit(
    payload: AccountingPayload, identity: tuple[str, str, str], units: int, cost: float | None
) -> Contribution:
    guardrail_id, date, unit = identity
    return Contribution(
        "LiteLLM_DailyGuardrailUsageUnits",
        (
            ("guardrail_id", guardrail_id),
            ("date", date),
            ("team_id", _text(payload.team_id)),
            ("api_key", _text(payload.api_key)),
            ("usage_unit", unit),
        ),
        (("units", units), ("cost", cost or 0.0), ("untracked_units", units if cost is None else 0)),
    )


def _guardrail_unit_costs(entry: Mapping[str, object]) -> Mapping[str, float | None] | None:
    try:
        cost: Final = GuardrailCostByUnitEntry.model_validate(entry)
    except ValidationError:
        return None
    return None if cost.guardrail_cost_in_spend is False else cost.guardrail_cost_by_unit


def _indexable(contribution: Contribution) -> bool:
    texts: Final = (value for _, value in (*contribution.key, *contribution.values) if value is not None)
    return all(len(text.encode("utf-8", "surrogatepass")) <= _MAX_INDEXED_TEXT_BYTES for text in texts)


def _capped(contribution: Contribution) -> Contribution:
    """Counters are summed into bigint columns in the charge's transaction, so one capped far below that range
    can never overflow a day's total."""
    return replace(
        contribution,
        increments=tuple(
            (name, max(-_MAX_COUNTER, min(_MAX_COUNTER, value)) if isinstance(value, int) else value)
            for name, value in contribution.increments
        ),
    )


def project(payload: AccountingPayload) -> tuple[Contribution, ...]:
    contributions: Final = (
        *_entity_contributions(payload),
        *_daily_contributions(payload),
        *_window_contributions(payload),
        *_model_contributions(payload),
        *_guardrail_contributions(payload),
    )
    return tuple(_capped(contribution) for contribution in contributions if _indexable(contribution))
