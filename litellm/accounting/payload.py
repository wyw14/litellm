from collections.abc import Mapping
from typing import Annotated, Final, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter, ValidationError

_FIELDS: Final = TypeAdapter(dict[str, object])
_VALUES: Final = TypeAdapter(tuple[object, ...])
_JSON: Final = TypeAdapter(object)


def _decoded_json(value: str) -> object:
    try:
        return _JSON.validate_json(value)
    except ValidationError:
        return None


def _mapping_or_none(value: object) -> Mapping[str, object] | None:
    return _FIELDS.validate_python(value) if isinstance(value, Mapping) else None


def _number_or_zero(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _is_true(value: object) -> bool:
    return value is True


def _texts(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return _texts(_decoded_json(value))
    try:
        return tuple(item for item in _VALUES.validate_python(value) if isinstance(item, str))
    except ValidationError:
        return ()


def _entries(value: object) -> tuple[Mapping[str, object], ...]:
    """A bare dict is one entry, as the spend log redactor normalizes that legacy shape."""
    if isinstance(value, Mapping):
        return (_FIELDS.validate_python(value),)
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        _FIELDS.validate_python(entry) for entry in _VALUES.validate_python(value) if isinstance(entry, Mapping)
    )


_Mapping = Annotated[Mapping[str, object] | None, BeforeValidator(_mapping_or_none)]
_Number = Annotated[float, BeforeValidator(_number_or_zero)]
_Count = Annotated[int, BeforeValidator(_int_or_zero)]
_Text = Annotated[str | None, BeforeValidator(_text_or_none)]
_Texts = Annotated[tuple[str, ...], BeforeValidator(_texts)]
_Entries = Annotated[tuple[Mapping[str, object], ...], BeforeValidator(_entries)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)


class AccountingEntities(_Frozen):
    api_key: _Text = None
    user: _Text = None
    end_user: _Text = None
    team_id: _Text = None
    organization_id: _Text = None
    project_id: _Text = None
    agent_id: _Text = None
    proxy_budget_name: _Text = None


class AccountingSavings(_Frozen):
    compression: _Number = 0.0
    prompt_caching: _Number = 0.0
    autorouter: _Number = 0.0
    gateway_injected_caching: _Number = 0.0


class AccountingWindow(_Frozen):
    entity_type: str
    entity_id: str
    window_duration: str
    window_start: str


class AccountingMetadata(_Frozen):
    entities: AccountingEntities = Field(default_factory=AccountingEntities, alias="accounting_entities")
    savings: AccountingSavings = Field(default_factory=AccountingSavings, alias="accounting_savings")
    model_access_groups: _Texts = Field(default=(), alias="accounting_model_access_groups")
    windows: tuple[AccountingWindow, ...] = Field(default=(), alias="accounting_windows")
    status: _Text = None
    usage_object: _Mapping = None
    compression_savings: _Mapping = None
    guardrail_information: _Entries = ()
    standard_logging_guardrail_information: _Entries = ()
    internal_call_origin: _Text = None
    billed_retry_attempt: Annotated[bool, BeforeValidator(_is_true)] = False


def _metadata(value: object) -> Mapping[str, object]:
    if isinstance(value, str):
        return _metadata(_decoded_json(value))
    return _mapping_or_none(value) or {}


class AccountingPayload(_Frozen):
    """The charge a request stores in its delivery, read back by the worker. Fields a caller can influence are
    normalized instead of rejected, so odd caller data never parks a paid request."""

    request_id: str = ""
    call_type: _Text = None
    api_key: _Text = None
    spend: _Number = 0.0
    prompt_tokens: _Count = 0
    completion_tokens: _Count = 0
    start_time: _Text = Field(default=None, alias="startTime")
    model: _Text = None
    model_group: _Text = None
    custom_llm_provider: _Text = None
    user: _Text = None
    team_id: _Text = None
    organization_id: _Text = None
    end_user: _Text = None
    agent_id: _Text = None
    request_tags: _Texts = ()
    status: _Text = None
    request_duration_ms: Annotated[int | None, BeforeValidator(_int_or_none)] = None
    mcp_namespaced_tool_name: _Text = None
    metadata: Annotated[AccountingMetadata, BeforeValidator(_metadata)] = Field(default_factory=AccountingMetadata)


ChargeKind = Literal["request", "job", "log_only"]


def charge_kind(row: Mapping[str, object]) -> ChargeKind:
    """What a buffered spend log row is owed: a request charge keyed by its own delivery id, a batch or video
    charge keyed by its cost row id so it settles once, or nothing because a pre-ledger proxy already charged it."""
    if "accounting_payload" not in row:
        return "log_only"
    return "request" if "accounting_delivery_id" in row else "job"


_CHARGE_FIELDS: Final = frozenset({"accounting_payload", "accounting_log_disabled", "accounting_delivery_id"})


def spend_log_columns(row: Mapping[str, object]) -> Mapping[str, object]:
    """The row as LiteLLM_SpendLogs stores it, without the charge it carries to the ledger."""
    return {key: value for key, value in row.items() if key not in _CHARGE_FIELDS}
