"""
Module responsible for

1. Writing spend increments to either in memory list of transactions or to redis
2. Reading increments from redis or in memory list of transactions and committing them to db
"""

import asyncio
import dataclasses
import json
import os
import random
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeAlias, cast
from urllib.parse import unquote

from pydantic import TypeAdapter
from typing_extensions import LiteralString, ReadOnly, TypedDict

import litellm
from litellm._internal_context import service_target, with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.constants import (
    BILLED_RETRY_ATTEMPT_METADATA_KEY,
    DB_SPEND_UPDATE_JOB_NAME,
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
)
from litellm.litellm_core_utils.batch_utils import BATCH_COST_REQUEST_ID_SUFFIX
from litellm.litellm_core_utils.core_helpers import finite_cost, single_log_line
from litellm.litellm_core_utils.litellm_logging import coerce_model_access_groups
from litellm.proxy._types import (
    DB_RETRY_SAFE_ERROR_TYPES,
    DBSpendUpdateTransactions,
    SpendLogsMetadata,
    SpendLogsPayload,
    ToolDiscoveryQueueItem,
)
from litellm.proxy.common_utils.user_api_key_cache import AUTH_OBJECTS_TARGET, project_cache_key
from litellm.proxy.db.daily_spend_bulk_upsert import (
    DAILY_SPEND_TABLES,
    daily_spend_entity_ids,
)
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.db.db_transaction_queue.tool_discovery_queue import (
    ToolDiscoveryQueue,
)
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
    WindowSpendTransaction,
    WindowSpendUpdateQueue,
)
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler, db_write_failure_is_requeue_safe
from litellm.proxy.db.request_error_tracking import request_error_accumulator
from litellm.proxy.db.rollup_lock_timeout import apply_rollup_lock_timeout
from litellm.proxy.spend_tracking.savings import (
    compute_savings_spend,
    marks_gateway_injection,
)
from litellm.proxy.spend_tracking.spend_log_error_logger import spend_log_error
from litellm.repositories.prisma_protocols import BatchTable
from litellm.types.utils import CallTypes
from litellm.types.videos.utils import VIDEO_COST_REQUEST_ID_SUFFIX, VIDEO_STATUS_CALL_TYPES

if TYPE_CHECKING:
    from litellm.accounting.projection import Contribution
    from litellm.proxy.db.autorouter_session_rollup import AutoRouterTurnTransaction
    from litellm.proxy.db.baseline_accounting import DailyBaselineAttribution
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router
else:
    PrismaClient = Any
    ProxyLogging = Any


RESPONSES_SESSION_CALL_TYPES: Final = frozenset({CallTypes.responses.value, CallTypes.aresponses.value})
_SPEND_METADATA_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_CALL_KWARG_VALUE: Final = TypeAdapter(object)


def _turn_chains_through_the_db(payload: Mapping[str, object], llm_router: "Router | None") -> bool:
    """A Responses turn the chat-completions bridge replays from its spend log on the next `previous_response_id`.

    Native Responses providers hold the conversation themselves, so their rows can wait for the batch write.
    A row with no router deployment behind it is treated as bridged, which keeps today's immediate flush.
    """
    if payload.get("call_type") not in RESPONSES_SESSION_CALL_TYPES:
        return False
    model_id: Final = payload.get("model_id")
    deployment: Final = (
        llm_router.registry().deployment(model_id) if llm_router is not None and isinstance(model_id, str) else None
    )
    if deployment is None:
        return True
    from litellm.responses.main import provider_stores_the_conversation

    litellm_params: Final = deployment.litellm_params
    return not provider_stores_the_conversation(
        litellm_params.model,
        litellm_params.custom_llm_provider,
        {
            "use_chat_completions_api": litellm_params.use_chat_completions_api,
            "model_info": deployment.model_info.model_dump(),
            "api_base": litellm_params.api_base,
        },
    )


def _persistable_api_key(token: str | None, logged_api_key: str) -> str | None:
    from litellm.litellm_core_utils.litellm_logging import is_valid_sha256_hash
    from litellm.proxy.spend_tracking.spend_tracking_utils import redact_logged_api_key

    if token is None or token == logged_api_key or is_valid_sha256_hash(token):
        return token
    return redact_logged_api_key(token)


async def _track_guardrail_usage_of_cost_row(
    payload: SpendLogsPayload, prisma_client: "PrismaClient", disable_spend_logs: bool
) -> None:
    """A job's cost row skips the spend log queue, so its guardrail runs are counted here instead of at the flush."""
    if disable_spend_logs:
        return
    from litellm.proxy.guardrails.usage_tracking import process_spend_logs_guardrail_usage

    try:
        await process_spend_logs_guardrail_usage(prisma_client=prisma_client, logs_to_process=(payload,))
    except Exception as e:  # noqa: BLE001  # guardrail usage is best effort, as at the spend log flush
        verbose_proxy_logger.warning(
            "Guardrail usage tracking failed for spend row %s (non-fatal): %s", payload["request_id"], e
        )


def _is_settle_once_cost_row(payload: SpendLogsPayload) -> bool:
    """A poll of a job whose cost is charged once, by the first poll that sees it, however often it is polled.

    A batch's cost is known only when a retrieve sees it finished (LIT-7048). A video's cost is
    charged at create by the built-in providers, but a custom provider may report it on the
    status poll that sees the video completed. Only a row carrying the job's cost suffix settles
    the job, so a poll that pays only its guardrails, under an id that client call ids cannot
    turn into the job's, is charged like any other request.
    """
    if payload.get("status") != "success":
        return False
    call_type: Final = payload.get("call_type")
    request_id: Final = payload["request_id"]
    if call_type == CallTypes.aretrieve_batch.value:
        return request_id.endswith(BATCH_COST_REQUEST_ID_SUFFIX)
    return call_type in VIDEO_STATUS_CALL_TYPES and request_id.endswith(VIDEO_COST_REQUEST_ID_SUFFIX)


def _cost_row_to_write(payload: SpendLogsPayload, disable_spend_logs: bool) -> Mapping[str, object]:
    """A job's cost row with logging off keeps only what its delivery needs, so no request detail is stored."""
    if disable_spend_logs is False:
        return payload
    return {
        "request_id": payload["request_id"],
        "call_type": payload["call_type"],
        "accounting_payload": payload["accounting_payload"],
        "accounting_log_disabled": True,
    }


class _SpendIncrement(TypedDict):
    increment: ReadOnly[float]


class _MemberSpendRow(TypedDict):
    user_id: ReadOnly[str]
    team_id: ReadOnly[str]
    cost: ReadOnly[float]


class SpendUpsertTable(Protocol):
    def upsert(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None: ...


class _SpendBatch(Protocol):
    litellm_usertable: BatchTable
    litellm_endusertable: SpendUpsertTable
    litellm_dailymodelusage: SpendUpsertTable

    def execute_raw(self, query: LiteralString, *args: object) -> None: ...

    litellm_verificationtoken: BatchTable
    litellm_teamtable: BatchTable
    litellm_teammembership: BatchTable
    litellm_organizationtable: BatchTable
    litellm_organizationmembership: BatchTable
    litellm_projecttable: BatchTable
    litellm_tagtable: BatchTable
    litellm_agentstable: BatchTable
    litellm_modelaccessgroupbudgettable: BatchTable


_EntitySpendTable: TypeAlias = Literal[
    "litellm_tagtable", "litellm_agentstable", "litellm_modelaccessgroupbudgettable", "litellm_projecttable"
]


_ENTITY_SPEND_TABLES: Final[Mapping[_EntitySpendTable, Callable[[_SpendBatch], BatchTable]]] = MappingProxyType(
    {
        "litellm_tagtable": lambda batcher: batcher.litellm_tagtable,
        "litellm_agentstable": lambda batcher: batcher.litellm_agentstable,
        "litellm_modelaccessgroupbudgettable": lambda batcher: batcher.litellm_modelaccessgroupbudgettable,
        "litellm_projecttable": lambda batcher: batcher.litellm_projecttable,
    }
)


def _entity_spend_table(batcher: _SpendBatch, table_accessor: _EntitySpendTable) -> BatchTable:
    return _ENTITY_SPEND_TABLES[table_accessor](batcher)


class SpendBatchManager(Protocol):
    async def __aenter__(self) -> _SpendBatch: ...

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...


class _SpendTransaction(Protocol):
    def batch_(self) -> SpendBatchManager: ...

    async def execute_raw(self, query: LiteralString, *args: object) -> int: ...


class _SpendTransactionManager(Protocol):
    async def __aenter__(self) -> _SpendTransaction: ...

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...


_SpendTableName = Literal[
    "user_list_transactions",
    "end_user_list_transactions",
    "key_list_transactions",
    "team_list_transactions",
    "team_member_list_transactions",
    "org_list_transactions",
    "org_member_list_transactions",
    "project_list_transactions",
    "tag_list_transactions",
    "model_access_group_list_transactions",
    "agent_list_transactions",
]


def _report_spend_update_step_failure(step_name: str, *, payload: Mapping[str, object], exc: Exception) -> None:
    spend_log_error(
        "Spend tracking - %s failed, this request is missing from that spend rollup. request_id=%s api_key=%s",
        step_name,
        payload.get("request_id"),
        payload.get("api_key"),
        exc=exc,
    )


_EMPTY_SPEND_LOG_METADATA: Final = cast(SpendLogsMetadata, MappingProxyType({}))  # cast-ok: empty read-only metadata


def _parse_spend_log_metadata(payload: Mapping[str, object]) -> SpendLogsMetadata:
    """Malformed metadata costs the request its metadata-derived fields, never its daily rollup row."""
    raw: Final = payload.get("metadata")
    if not isinstance(raw, str):
        return _EMPTY_SPEND_LOG_METADATA
    try:
        parsed: Final[SpendLogsMetadata] = json.loads(raw)
    except json.JSONDecodeError as e:
        spend_log_error(
            "Spend tracking - unreadable spend log metadata, rollups for this request are written without "
            "metadata-derived fields. request_id=%s api_key=%s",
            payload.get("request_id"),
            payload.get("api_key"),
            exc=e,
        )
        return _EMPTY_SPEND_LOG_METADATA
    return parsed


def _is_billed_retry_attempt(metadata: SpendLogsMetadata) -> bool:
    """An earlier billed retry attempt's own spend row, not a request the caller sent."""
    return BILLED_RETRY_ATTEMPT_METADATA_KEY in metadata and metadata[BILLED_RETRY_ATTEMPT_METADATA_KEY] is True


_ENTITY_SPEND_MODELS: Final[Mapping[_EntitySpendTable, str]] = MappingProxyType(
    {
        "litellm_tagtable": "LiteLLM_TagTable",
        "litellm_agentstable": "LiteLLM_AgentsTable",
        "litellm_modelaccessgroupbudgettable": "LiteLLM_ModelAccessGroupBudgetTable",
        "litellm_projecttable": "LiteLLM_ProjectTable",
    }
)


@asynccontextmanager
async def _spend_update_tx(
    prisma_client: PrismaClient,
    table: str,
    call_type: str = "commit_spend_updates",
    outer_transaction: _SpendTransaction | None = None,
) -> AsyncGenerator[_SpendTransaction]:
    if outer_transaction is not None:
        yield outer_transaction
        return
    tx: Final[_SpendTransactionManager] = prisma_client.db.tx(timeout=timedelta(seconds=60))
    async with db_span(call_type, table), tx as transaction:
        await apply_rollup_lock_timeout(transaction)
        yield transaction


_daily_spend_commit_started: Final[ContextVar[asyncio.Event | None]] = ContextVar(
    "_daily_spend_commit_started", default=None
)


# The per-team advisory lock the team endpoints hold while changing a roster (TEAM_ADVISORY_LOCK_SQL),
# so the roster check below cannot interleave with their writes. A row lock would deadlock with the
# access-group endpoints, which lock a team row after an access-group lock.
_TEAM_ADVISORY_LOCK_SQL: Final = "SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS locked"

# One statement adds every member's cost to their membership row. A missing row is created only
# while the user is still on the team's roster, so a spend flush landing after a removal never
# recreates the member. The rows travel as one JSON document, not as a numeric array: Prisma
# types a raw array parameter from the first batch a connection sees, so after an all-$0 batch
# (integers) every later fractional batch on that connection failed with "improper binary format".
# total_spend is floored at spend before adding: rows written before total_spend existed (migration
# 20260421135425, default 0) carry their history only in spend, so a plain increment keeps them behind forever.
_TEAM_MEMBER_SPEND_SQL: Final = """
INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, total_spend)
SELECT member.user_id, member.team_id, member.cost, member.cost
FROM jsonb_to_recordset($1::jsonb) AS member(user_id text, team_id text, cost float8)
WHERE EXISTS (
    SELECT 1 FROM "LiteLLM_TeamTable" t
    WHERE t.team_id = member.team_id
      AND t.members_with_roles @> jsonb_build_array(jsonb_build_object('user_id', member.user_id))
)
   OR EXISTS (
    SELECT 1 FROM "LiteLLM_TeamMembership" m
    WHERE m.user_id = member.user_id AND m.team_id = member.team_id
)
ON CONFLICT (user_id, team_id) DO UPDATE
SET spend = "LiteLLM_TeamMembership".spend + EXCLUDED.spend,
    total_spend = GREATEST("LiteLLM_TeamMembership".total_spend, "LiteLLM_TeamMembership".spend) + EXCLUDED.total_spend
"""


async def _write_team_member_spend(transaction: _SpendTransaction, spend_by_member_key: Mapping[str, float]) -> None:
    # key is "team_id::<value>::user_id::<value>"; locks are taken in sorted team_id order like the team endpoints
    rows: Final = sorted((key.split("::")[1], key.split("::")[3], cost) for key, cost in spend_by_member_key.items())
    for team_id in dict.fromkeys(team_id for team_id, _user_id, _cost in rows):
        _ = await transaction.execute_raw(_TEAM_ADVISORY_LOCK_SQL, team_id)
    members: Final = tuple(
        _MemberSpendRow(user_id=user_id, team_id=team_id, cost=cost) for team_id, user_id, cost in rows
    )
    _ = await transaction.execute_raw(_TEAM_MEMBER_SPEND_SQL, json.dumps(members))


def get_llm_router():
    """The proxy's router, or None outside a running proxy.

    Injected rather than imported where it is used, so the savings computation stays
    a pure function of its arguments and the caller owns where the router comes from.
    """
    try:
        from litellm.proxy.proxy_server import llm_router

        return llm_router
    except Exception:  # noqa: BLE001  # no proxy in scope; savings degrade to zero
        return None


class _DeploymentRegistry(Protocol):
    def deployment_dict(self, id: str) -> Mapping[str, object] | None: ...


class _DeploymentLookup(Protocol):
    def registry(self) -> _DeploymentRegistry: ...


def _served_model_access_groups(
    router: _DeploymentLookup | None,
    served_model_id: str | None,
) -> frozenset[str] | None:
    """Access groups declared by the deployment that actually served the request.

    None when the served deployment cannot be identified, in which case the set
    attributed at auth time stands unchanged.
    """
    if router is None or not served_model_id:
        return None
    deployment: Final = router.registry().deployment_dict(served_model_id)
    if deployment is None:
        return None
    model_info: Final = deployment.get("model_info")
    if not isinstance(model_info, Mapping):
        return None
    declared: Final = model_info.get("access_groups")
    if not isinstance(declared, (list, tuple)):
        return frozenset()
    return frozenset(group for group in declared if isinstance(group, str))


def debitable_model_access_groups(
    attributed: Sequence[str] | None,
    served_model_id: str | None,
    router: _DeploymentLookup | None,
) -> tuple[str, ...]:
    """Groups to debit: the set attributed at auth time, narrowed to those the served model belongs to.

    The router may fall back to a model outside the pool auth reserved against, so the
    attributed set is the hard upper bound: a group absent from it is never debited.
    """
    ordered: Final = coerce_model_access_groups(attributed)
    if not ordered:
        return ()
    served: Final = _served_model_access_groups(router=router, served_model_id=served_model_id)
    if served is None:
        return ordered
    return tuple(group for group in ordered if group in served)


def _billable_cost(response_cost: float | None, payload: SpendLogsPayload) -> float | None:
    billable_cost: Final = finite_cost(response_cost)
    if response_cost is not None and billable_cost is None:
        verbose_proxy_logger.warning(
            "Spend tracking - recording a non-finite response_cost=%s as 0 for request_id=%s",
            response_cost,
            single_log_line(payload["request_id"]),
        )
    return billable_cost


def _iso_timestamp(value: datetime | str) -> str:
    return value.isoformat() if isinstance(value, datetime) else value


def _report_dropped_window_increments(transactions: Sequence[WindowSpendTransaction], error: Exception) -> None:
    spend_log_error(
        "Spend tracking - dropped %d budget window increments: the failed statement may have "
        "applied or the database refused the data, so re-sending it is not safe. Error: %s",
        len(transactions),
        str(error),
        exc=error,
    )


class DBSpendUpdateWriter:
    """
    Module responsible for

    1. Writing spend increments to either in memory list of transactions or to redis
    2. Reading increments from redis or in memory list of transactions and committing them to db
    """

    def __init__(
        self,
        redis_cache: RedisCache | None = None,
    ):
        self.redis_cache = redis_cache
        self.redis_update_buffer = RedisUpdateBuffer(redis_cache=self.redis_cache)
        self.pod_lock_manager = PodLockManager()
        self.tool_discovery_queue = ToolDiscoveryQueue()
        self.window_spend_update_queue = WindowSpendUpdateQueue()

    async def update_database(
        # LiteLLM management object fields
        self,
        token: str | None,
        user_id: str | None,
        end_user_id: str | None,
        team_id: str | None,
        org_id: str | None,
        # Completion object fields
        kwargs: dict | None,
        completion_response: object,
        start_time: datetime,
        end_time: datetime,
        response_cost: float | None,
        project_id: str | None = None,
    ) -> bool:
        """Record the request's spend, answering whether its cost still needs charging.

        False only for a batch or video poll whose cost row another poll already wrote,
        so the caller leaves the key, team, and user counters alone (LIT-7048).
        """
        from litellm.proxy.proxy_server import (
            accounting_window_keys,
            disable_spend_logs,
            general_settings,  # pyright: ignore[reportUnknownVariableType]  # legacy proxy configuration is narrowed below
            litellm_proxy_budget_name,
            prisma_client,
        )
        from litellm.proxy.utils import ProxyUpdateSpend, hash_token

        proxy_settings: Final = cast(  # cast-ok: proxy settings use string keys
            Mapping[str, object], general_settings
        )
        request_fields: Final = _SPEND_METADATA_ADAPTER.validate_python(kwargs or {})
        try:
            verbose_proxy_logger.debug(
                "Enters prisma db call, response_cost: %s, token: %s; user_id: %s; team_id: %s",
                response_cost,
                token,
                user_id,
                team_id,
            )
            if ProxyUpdateSpend.disable_spend_updates() is True:
                return True
            hashed_token: Final = (
                hash_token(token=token) if isinstance(token, str) and token.startswith("sk-") else token
            )

            ## CREATE SPEND LOG PAYLOAD ##
            from litellm.proxy.spend_tracking.spend_tracking_utils import (
                get_logging_payload,
                get_request_model_access_groups,
            )

            llm_router: Final = get_llm_router()
            payload: Final = get_logging_payload(
                kwargs=kwargs,
                response_obj=completion_response,
                start_time=start_time,
                end_time=end_time,
                llm_router=llm_router,
            )
            billable_cost: Final = _billable_cost(response_cost, payload)
            payload["spend"] = billable_cost or 0.0
            payload["startTime"] = _iso_timestamp(payload["startTime"])
            payload["endTime"] = _iso_timestamp(payload["endTime"])
            if org_id:
                payload["organization_id"] = org_id
            if team_id:
                payload["team_id"] = team_id

            from litellm.accounting.snapshot import build_accounting_payload

            if not _is_settle_once_cost_row(payload):
                payload["accounting_delivery_id"] = uuid.uuid4().hex
            payload["accounting_payload"] = build_accounting_payload(
                payload,
                entities={
                    "api_key": _persistable_api_key(hashed_token, payload.get("api_key") or ""),
                    "user": user_id,
                    "end_user": end_user_id,
                    "team_id": team_id,
                    "organization_id": org_id,
                    "project_id": project_id,
                    "agent_id": payload["agent_id"],
                    "proxy_budget_name": litellm_proxy_budget_name if litellm.max_budget > 0 else None,
                },
                model_access_groups=debitable_model_access_groups(
                    get_request_model_access_groups(request_fields), payload["model_id"], llm_router
                ),
                windows=await accounting_window_keys(hashed_token, team_id) if billable_cost else (),
                router=get_llm_router,
            )
            keep_spend_logs: Final = disable_spend_logs is False and not (
                "status" in payload
                and payload["status"] == "failure"
                and proxy_settings.get("disable_error_logs") is True
            )
            if not await self._record_spend_log(
                payload=payload,
                prisma_client=prisma_client,
                disable_spend_logs=not keep_spend_logs,
                llm_router=llm_router,
            ):
                return False

            # The auto-router router-day rollup is an aggregate like the daily spend tables, so it is
            # written whether or not per-request spend logs are kept; per-session rows are not.
            await self._enqueue_autorouter_turn_transaction(
                payload=payload,
                prisma_client=prisma_client,
                spend_logs_kept=keep_spend_logs,
            )
            if keep_spend_logs:
                await self._enqueue_tool_usage_transaction(
                    payload=payload,
                    completion_response=completion_response,
                    prisma_client=prisma_client,
                    kwargs=kwargs,
                )
            else:
                verbose_proxy_logger.debug(
                    "disable_spend_logs=True. Skipping writing spend logs to db. Other spend updates - Key/User/Team table will still occur."
                )

            self._record_request_error_safely(payload=payload, prisma_client=prisma_client)
            self._enqueue_tool_registry_upsert(
                kwargs=kwargs,
                completion_response=completion_response,
                hashed_token=hashed_token,
                team_id=team_id,
            )

            verbose_proxy_logger.debug("Runs spend update on all tables")
            return True
        except Exception:
            spend_log_error(
                "Spend tracking - update_database failed. Spend log insertion or daily transaction enqueue "
                "may not have completed for this request. "
                "response_cost=%s, token=%s, user_id=%s, team_id=%s, org_id=%s, end_user_id=%s",
                response_cost,
                token,
                user_id,
                team_id,
                org_id,
                end_user_id,
            )
            return True

    async def _record_spend_log(
        self,
        payload: SpendLogsPayload,
        prisma_client: "PrismaClient | None",
        disable_spend_logs: bool,
        llm_router: "Router | None" = None,
    ) -> bool:
        if prisma_client is not None and _is_settle_once_cost_row(payload):
            return await self._claim_settle_once_cost_row(
                payload=payload, prisma_client=prisma_client, disable_spend_logs=disable_spend_logs
            )
        await self._insert_spend_log_to_db(
            payload={**payload, "accounting_log_disabled": disable_spend_logs},
            prisma_client=prisma_client,
            llm_router=llm_router,
        )
        return True

    async def _claim_settle_once_cost_row(
        self, payload: SpendLogsPayload, prisma_client: "PrismaClient", disable_spend_logs: bool
    ) -> bool:
        """Charge a batch or video job once, however many times it is polled after completing.

        The job's delivery is keyed by its cost row id, so the first completed poll owns the charge
        and every later poll conflicts and charges nothing (LIT-7048). A $0 poll records no
        delivery, so it cannot block the poll that carries the cost.
        """
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            configured_spend_logs_metadata_fields,
            spend_log_row_with_retained_metadata,
        )
        from litellm.repositories.table_repositories import SpendLogsRepository

        if payload["spend"] <= 0:
            return True
        row: Final = spend_log_row_with_retained_metadata(
            _cost_row_to_write(payload, disable_spend_logs), configured_spend_logs_metadata_fields()
        )
        try:
            accepted: Final = (
                await SpendLogsRepository(prisma_client).insert_with_deliveries([prisma_client.jsonify_object(row)])
                == 1
            )
        except Exception as e:  # noqa: BLE001  # prisma raises its own hierarchy; an unreachable DB queues the row like any other spend log
            verbose_proxy_logger.warning(
                "Could not claim spend row %s for a job's cost, queueing it: %s", payload["request_id"], e
            )
            await self._insert_spend_log_to_db(payload=prisma_client.jsonify_object(row), prisma_client=prisma_client)
            return True
        if accepted:
            await _track_guardrail_usage_of_cost_row(payload, prisma_client, disable_spend_logs)
        return accepted

    async def _enqueue_tool_usage_transaction(
        self,
        payload: SpendLogsPayload,
        completion_response: object,
        prisma_client: "PrismaClient | None",
        kwargs: "dict | None" = None,
    ) -> None:
        try:
            if prisma_client is None:
                return
            from litellm.proxy.db.spend_log_tool_index import (
                build_tool_usage_transaction,
            )

            transaction: Final = build_tool_usage_transaction(
                request_id=payload["request_id"],
                start_time_iso=str(payload["startTime"]),
                mcp_namespaced_tool_name=payload.get("mcp_namespaced_tool_name"),
                spend=payload["spend"],
                total_tokens=payload["total_tokens"],
                completion_response=completion_response,
                realtime_tool_calls=_CALL_KWARG_VALUE.validate_python((kwargs or {}).get("realtime_tool_calls")),
            )
            if transaction is None:
                return
            async with prisma_client._tool_usage_transactions_lock:
                prisma_client.tool_usage_transactions.append(transaction)
        except Exception as e:
            verbose_proxy_logger.debug("_enqueue_tool_usage_transaction error (non-blocking): %s", e)

    async def _enqueue_autorouter_turn_transaction(
        self,
        payload: SpendLogsPayload,
        prisma_client: "PrismaClient | None",
        spend_logs_kept: bool = True,
    ) -> None:
        try:
            if prisma_client is None:
                return
            metadata_raw: Final = payload.get("metadata")
            if not metadata_raw:
                return
            metadata: Final = _SPEND_METADATA_ADAPTER.validate_json(metadata_raw)
            routing_decision: Final = metadata.get("routing_decision")
            if not isinstance(routing_decision, Mapping) or not routing_decision:
                return
            from litellm.proxy.db.autorouter_session_rollup import (
                build_autorouter_turn_transaction,
            )

            usage_object_raw: Final = metadata.get("usage_object")
            cost_breakdown: Final = metadata.get("cost_breakdown")
            savings_estimate: Final = metadata.get("autorouter_savings_estimate")
            savings_spend: Final = compute_savings_spend(
                model=payload.get("model"),
                custom_llm_provider=payload.get("custom_llm_provider"),
                compression_saved_tokens=0,
                gateway_injected_cache=marks_gateway_injection(metadata, payload.get("model_id")),
                routing_decision=routing_decision,
                usage_object=usage_object_raw if isinstance(usage_object_raw, dict) else None,
                model_id=payload.get("model_id"),
                llm_router=get_llm_router,
                cost_breakdown=cost_breakdown if isinstance(cost_breakdown, Mapping) else None,
                recorded_autorouter_savings=metadata.get("autorouter_savings"),
                recorded_autorouter_savings_estimate=(
                    savings_estimate if isinstance(savings_estimate, Mapping) else None
                ),
                billed_at=payload.get("endTime"),
            )
            transaction: Final = build_autorouter_turn_transaction(
                payload=payload,
                metadata=metadata,
                saved_spend=savings_spend.autorouter,
            )
            try:
                # A baseline observation publishes only once its spend log exists, so without spend logs
                # it could never publish; the plain turn still carries this request's recorded savings.
                if spend_logs_kept and await self._enqueue_baseline_accounting(
                    payload, metadata, transaction, prisma_client
                ):
                    return
            except Exception:  # noqa: BLE001  # optional baseline capture must preserve the original actual-spend rollup
                verbose_proxy_logger.warning("Auto-router baseline observation was unavailable; actual turn retained")
            if transaction is None:
                return
            # Without spend logs only the router-day aggregate is kept: an empty session id makes the
            # session upserts skip the row, so no per-session record is stored.
            kept: Final = transaction if spend_logs_kept else dataclasses.replace(transaction, session_id="")
            async with prisma_client._autorouter_turn_transactions_lock:
                prisma_client.autorouter_turn_transactions.append(kept)
        except Exception as e:  # noqa: BLE001  # a metrics enqueue must never fail the spend write
            verbose_proxy_logger.debug("_enqueue_autorouter_turn_transaction error (non-blocking): %s", e)

    async def _enqueue_baseline_accounting(
        self,
        payload: SpendLogsPayload,
        metadata: Mapping[str, object],
        turn: "AutoRouterTurnTransaction | None",
        prisma_client: "PrismaClient",
    ) -> bool:
        from litellm.proxy.db.baseline_accounting import (
            BaselineAccountingRecord,
        )
        from litellm.proxy.hooks.autorouter_baseline_cache import CapturedBaselineObservation
        from litellm.proxy.spend_tracking.savings import baseline_cost_snapshot

        serialized: Final = metadata.get("autorouter_baseline_observation")
        if not isinstance(serialized, str):
            return False
        captured: Final = CapturedBaselineObservation.model_validate_json(serialized)
        if captured.api_key != payload["api_key"] or captured.session_id != payload["session_id"]:
            return False
        decision: Final = _SPEND_METADATA_ADAPTER.validate_python(
            metadata.get("routing_decision") or MappingProxyType({})
        )
        breakdown: Final = _SPEND_METADATA_ADAPTER.validate_python(
            metadata.get("cost_breakdown") or MappingProxyType({})
        )
        daily: Final = await self._baseline_daily_attribution(payload, prisma_client)
        record: Final = BaselineAccountingRecord(
            scope=captured.scope,
            api_key=captured.api_key,
            session_id=captured.session_id,
            router_name=captured.router_name,
            baseline_model=captured.baseline_model,
            observation=captured.observation.model_copy(update=MappingProxyType({"request_id": payload["request_id"]})),
            pricing=baseline_cost_snapshot(captured.model, captured.prices, payload["spend"], breakdown, decision),
            turn=turn,
            daily=daily,
        )
        async with prisma_client.baseline_accounting_lock:
            if len(prisma_client.baseline_accounting_transactions) >= 10000:
                verbose_proxy_logger.warning("Auto-router baseline observation queue is full")
                return False
            prisma_client.baseline_accounting_transactions.append(record)
        from litellm.proxy.utils import request_spend_log_flush

        request_spend_log_flush(prisma_client)
        return True

    async def _baseline_daily_attribution(
        self,
        payload: SpendLogsPayload,
        prisma_client: "PrismaClient",
    ) -> "DailyBaselineAttribution | None":
        from litellm.proxy.db.baseline_accounting import DailyBaselineAttribution, DailyBaselineTarget

        normalized: Final = cast(SpendLogsPayload, MappingProxyType({**payload, "end_user_id": payload["end_user"]}))
        bases: Final = tuple(
            zip(
                DAILY_SPEND_TABLES,
                await asyncio.gather(
                    *(
                        self._common_add_spend_log_transaction_to_daily_transaction(  # pyright: ignore[reportUnknownMemberType]  # legacy payload union; this caller supplies a validated spend payload
                            normalized,
                            prisma_client,
                            "request_tags" if entity == "tag" else entity,
                        )
                        for entity in DAILY_SPEND_TABLES
                    )
                ),
            )
        )
        base: Final = next((base for _, base in bases if base is not None), None)
        if base is None:
            return None
        return DailyBaselineAttribution(
            date=base["date"],
            api_key=base["api_key"],
            model=base.get("model"),
            custom_llm_provider=base.get("custom_llm_provider"),
            model_group=base.get("model_group"),
            endpoint=base.get("endpoint"),
            mcp_namespaced_tool_name=base.get("mcp_namespaced_tool_name"),
            targets=tuple(
                DailyBaselineTarget(entity=entity, entity_id=identity)
                for entity, values in bases
                if values is not None
                for identity in daily_spend_entity_ids(payload, entity)
            ),
        )

    def _enqueue_tool_registry_upsert(
        self,
        kwargs: dict | None,
        completion_response: object,
        hashed_token: str | None = None,
        team_id: str | None = None,
    ) -> None:
        """
        Extract tool names from the LLM request and response and enqueue them
        for upsert into LiteLLM_ToolTable via ToolDiscoveryQueue.

        Handles four sources:
        - MCP tools: standard_logging_object.mcp_tool_call_metadata.namespaced_tool_name
        - Response tool_calls (OpenAI / Anthropic pass-through converted to OpenAI format):
            completion_response.choices[].message.tool_calls[].function.name
        - Request tools array (OpenAI format): kwargs["tools"][].function.name
        - Request tools array (Anthropic /messages format): kwargs["passthrough_logging_payload"]
            ["request_body"]["tools"][].name
        """
        try:
            if kwargs is None:
                return

            # Extract key_alias from kwargs metadata if available
            key_alias: str | None = None
            _litellm_params: Final = kwargs.get("litellm_params") or {}
            _metadata: Final = _litellm_params.get("metadata") or {}
            key_alias = _metadata.get("user_api_key_alias") or None
            user_agent: Final = _metadata.get("user_agent") or None

            def _enqueue(tool_name: str, origin: str = "user_defined") -> None:
                self.tool_discovery_queue.add_update(
                    ToolDiscoveryQueueItem(
                        tool_name=tool_name,
                        origin=origin,
                        key_hash=hashed_token,
                        team_id=team_id,
                        key_alias=key_alias,
                        user_agent=user_agent,
                    )
                )

            # --- MCP tool calls ---
            sl_object: Final = kwargs.get("standard_logging_object")
            if sl_object is not None:
                mcp_metadata: Final = (sl_object.get("metadata", {}) or {}).get("mcp_tool_call_metadata")
                if mcp_metadata and isinstance(mcp_metadata, dict):
                    tool_name = mcp_metadata.get("namespaced_tool_name") or mcp_metadata.get("name")
                    mcp_server_name: Final = mcp_metadata.get("mcp_server_name")
                    if tool_name:
                        _enqueue(tool_name, origin=mcp_server_name or "user_defined")

            # --- Tools from request body (OpenAI format: tools[].function.name) ---
            request_tools: Final = kwargs.get("tools") or []
            for tool_def in request_tools:
                if not isinstance(tool_def, dict):
                    continue
                fn = tool_def.get("function") or {}
                name = fn.get("name") if isinstance(fn, dict) else None
                if name:
                    _enqueue(name)

            # --- Tools from Anthropic /messages pass-through request body
            #     (Anthropic format: tools[].name, no "function" wrapper) ---
            passthrough_payload: Final = kwargs.get("passthrough_logging_payload") or {}
            request_body: Final = (
                passthrough_payload.get("request_body") if isinstance(passthrough_payload, dict) else None
            ) or {}
            for tool_def in request_body.get("tools") or []:
                if not isinstance(tool_def, dict):
                    continue
                name = tool_def.get("name")
                if name:
                    _enqueue(name)

            # --- Response tool_calls (OpenAI format; Anthropic pass-through converts tool_use here) ---
            from litellm.proxy.db.spend_log_tool_index import response_tool_call_names

            for tool_name in response_tool_call_names(completion_response):
                _enqueue(tool_name)
        except Exception as e:
            verbose_proxy_logger.debug("_enqueue_tool_registry_upsert error (non-blocking): %s", e)

    async def _insert_spend_log_to_db(
        self,
        payload: Mapping[str, object],
        prisma_client: PrismaClient | None = None,
        spend_logs_url: str | None = os.getenv("SPEND_LOGS_URL"),
        llm_router: "Router | None" = None,
    ) -> PrismaClient | None:
        verbose_proxy_logger.debug(
            "Writing spend log to db - request_id: {}, spend: {}".format(
                payload.get("request_id"), payload.get("spend")
            )
        )
        if prisma_client is not None and spend_logs_url is not None or prisma_client is not None:
            from litellm.proxy.utils import enqueue_spend_logs, request_spend_log_flush

            await enqueue_spend_logs(prisma_client, (payload,))
            if _turn_chains_through_the_db(_SPEND_METADATA_ADAPTER.validate_python(payload), llm_router):
                request_spend_log_flush(prisma_client)
        else:
            verbose_proxy_logger.debug("prisma_client is None. Skipping writing spend logs to db.")

        return prisma_client

    async def db_update_spend_transaction_handler(
        self,
        prisma_client: PrismaClient,
        n_retry_times: int,
        proxy_logging_obj: ProxyLogging,
    ):
        """Apply pending accounting deliveries, then commit the model budget limiter's window increments."""
        from litellm.accounting.worker import flush_accounting_deliveries

        try:
            await flush_accounting_deliveries(self, prisma_client, proxy_logging_obj)
        except Exception:  # noqa: BLE001  # pending deliveries survive outages while window increments must still drain
            verbose_proxy_logger.exception("Accounting delivery flush failed; pending work will retry")

        if RedisUpdateBuffer.should_commit_spend_updates_to_redis():
            await self._commit_window_spend_updates_with_redis(prisma_client)
        else:
            await self._commit_window_spend_updates_from_memory(prisma_client)

    async def _commit_window_spend_updates_with_redis(self, prisma_client: PrismaClient) -> None:
        """Every pod pushes its increments to Redis and only the pod holding the lock writes them, which avoids
        cross-pod deadlocks on LiteLLM_BudgetWindowSpend."""
        await self.redis_update_buffer.store_window_spend_updates_in_redis(self.window_spend_update_queue)
        if not await self.pod_lock_manager.acquire_lock(cronjob_id=DB_SPEND_UPDATE_JOB_NAME):
            return
        try:
            transactions: Final = await self.redis_update_buffer.get_window_spend_updates_from_redis()
            try:
                await DBSpendUpdateWriter._commit_window_spend_updates(
                    prisma_client=prisma_client, window_spend_transactions=transactions
                )
            except Exception as e:  # noqa: BLE001  # requeue-safe failures go back to Redis for the next tick
                if db_write_failure_is_requeue_safe(e):
                    spend_log_error(
                        "Spend tracking - failed to commit %d budget window increments from Redis, re-queued them "
                        "for the next tick. Error: %s",
                        len(transactions),
                        str(e),
                        exc=e,
                    )
                    await self.redis_update_buffer.restore_window_spend_updates_to_redis(transactions)
                else:
                    _report_dropped_window_increments(transactions, e)
        finally:
            await self.pod_lock_manager.release_lock(cronjob_id=DB_SPEND_UPDATE_JOB_NAME)

    async def _commit_window_spend_updates_from_memory(self, prisma_client: PrismaClient) -> None:
        transactions: Final = await self.window_spend_update_queue.flush_and_get_aggregated_window_spend_transactions()
        try:
            await DBSpendUpdateWriter._commit_window_spend_updates(
                prisma_client=prisma_client, window_spend_transactions=transactions
            )
        except Exception as e:  # noqa: BLE001  # the increments go back on the queue; the rest of the flush must run
            if db_write_failure_is_requeue_safe(e):
                spend_log_error(
                    "Spend tracking - failed to commit budget window spend updates. "
                    "Re-queued %d window increments for retry on next tick. Error: %s",
                    len(transactions),
                    str(e),
                    exc=e,
                )
                await self.window_spend_update_queue.update_queue.put(transactions)
            else:
                _report_dropped_window_increments(transactions, e)

    @staticmethod
    async def _commit_window_spend_updates(
        prisma_client: PrismaClient,
        window_spend_transactions: Sequence[WindowSpendTransaction],
    ) -> None:
        """
        Commit per-budget-window spend increments to LiteLLM_BudgetWindowSpend.

        Raises on failure so the caller re-queues the increments: budget
        enforcement trusts a current row without reconciling it against
        LiteLLM_SpendLogs, so a dropped increment would let the entity spend
        past its window limit after the next counter reseed.
        """
        from litellm.proxy.db.budget_window_spend_writer import (
            commit_window_spend_updates,
        )

        await commit_window_spend_updates(
            prisma_client=prisma_client,
            transactions=window_spend_transactions,
        )

    @staticmethod
    async def handle_spend_update_failure(
        e: Exception,
        attempt: int,
        n_retry_times: int,
        start_time: float,
        proxy_logging_obj: ProxyLogging,
    ) -> None:
        """Retry a failed spend-update transaction on connection errors, deadlocks or a rollup
        ``lock_timeout`` (55P03), else re-raise. All three roll the transaction back before any
        increment applied, so re-sending the same batch cannot double-count."""
        from litellm.proxy.utils import raise_failed_update_spend_exception

        is_retryable = (
            isinstance(e, DB_RETRY_SAFE_ERROR_TYPES)
            or PrismaDBExceptionHandler.is_deadlock_error(e)
            or PrismaDBExceptionHandler.is_lock_timeout_error(e)
        )
        if not is_retryable or attempt >= n_retry_times:
            raise_failed_update_spend_exception(e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj)
        verbose_proxy_logger.warning(
            "Retrying spend update after retryable DB error (attempt %s/%s): %s",
            attempt + 1,
            n_retry_times,
            e,
        )
        await asyncio.sleep(random.uniform(2.0**attempt, 2.0 ** (attempt + 1)))

    _handle_spend_update_failure = handle_spend_update_failure

    @staticmethod
    async def invalidate_accounting_entities(
        contributions: tuple["Contribution", ...], proxy_logging_obj: ProxyLogging
    ) -> None:
        await DBSpendUpdateWriter._invalidate_project_caches(
            tuple(dict(item.key)["project_id"] for item in contributions if item.table == "LiteLLM_ProjectTable"),
            proxy_logging_obj,
        )
        from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

        details: Final = _SPEND_METADATA_ADAPTER.validate_python(
            proxy_logging_obj.call_details  # pyright: ignore[reportUnknownMemberType]  # legacy logging metadata is validated at this boundary
        )
        user_api_key_cache: Final = details.get("user_api_key_cache")
        if isinstance(user_api_key_cache, UserApiKeyCache):
            with service_target(AUTH_OBJECTS_TARGET):
                for item in contributions:
                    if item.table == "LiteLLM_TeamMembership":
                        await user_api_key_cache.async_delete_cache(
                            key=f"team_membership:{dict(item.key)['user_id']}:{dict(item.key)['team_id']}"
                        )

    async def apply_accounting_entities(
        self,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        transaction: _SpendTransaction,
        updates: DBSpendUpdateTransactions,
    ) -> None:
        await self._commit_spend_updates_to_db(
            prisma_client, 0, proxy_logging_obj, updates, outer_transaction=transaction
        )

    async def _commit_spend_updates_to_db(
        self,
        prisma_client: PrismaClient,
        n_retry_times: int,
        proxy_logging_obj: ProxyLogging,
        db_spend_update_transactions: DBSpendUpdateTransactions,
        on_table_committed: Callable[[_SpendTableName], None] | None = None,
        outer_transaction: _SpendTransaction | None = None,
    ):
        """
        Commits all the spend `UPDATE` transactions to the Database

        """
        from litellm.proxy.utils import ProxyUpdateSpend

        ### UPDATE USER TABLE ###
        user_list_transactions: Final = db_spend_update_transactions["user_list_transactions"]
        verbose_proxy_logger.debug("User Spend transactions: %s", user_list_transactions)
        if user_list_transactions is not None and len(user_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, "LiteLLM_UserTable", outer_transaction=outer_transaction
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            # Sort by ID for consistent lock ordering across pods to prevent deadlocks.
                            # batch_() issues statements sequentially within the tx, so iteration
                            # order = lock acquisition order.
                            for user_id, response_cost in sorted(user_list_transactions.items()):
                                batcher.litellm_usertable.update_many(
                                    where={"user_id": user_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
        if on_table_committed is not None:
            on_table_committed("user_list_transactions")

        ### UPDATE END-USER TABLE ###
        end_user_list_transactions: Final = db_spend_update_transactions["end_user_list_transactions"]
        verbose_proxy_logger.debug("End-User Spend transactions: %s", end_user_list_transactions)
        if end_user_list_transactions is not None and len(end_user_list_transactions.keys()) > 0:
            await ProxyUpdateSpend.update_end_user_spend(
                n_retry_times=n_retry_times,
                prisma_client=prisma_client,
                proxy_logging_obj=proxy_logging_obj,
                end_user_list_transactions=end_user_list_transactions,
                outer_transaction=outer_transaction,
            )
        if on_table_committed is not None:
            on_table_committed("end_user_list_transactions")
        ### UPDATE KEY TABLE ###
        key_list_transactions: Final = db_spend_update_transactions["key_list_transactions"]
        verbose_proxy_logger.debug("KEY Spend transactions: %s", key_list_transactions)
        if key_list_transactions is not None and len(key_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, "LiteLLM_VerificationToken", outer_transaction=outer_transaction
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            # Sort by token for consistent lock ordering across pods to prevent deadlocks.
                            for token, response_cost in sorted(key_list_transactions.items()):
                                spend_increment: _SpendIncrement = {"increment": response_cost}
                                batcher.litellm_verificationtoken.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"token": token},
                                    data={
                                        "spend": spend_increment,
                                        "total_spend": spend_increment,
                                        "last_active": datetime.now(timezone.utc),
                                    },
                                )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
        if on_table_committed is not None:
            on_table_committed("key_list_transactions")

        ### UPDATE TEAM TABLE ###
        team_list_transactions: Final = db_spend_update_transactions["team_list_transactions"]
        verbose_proxy_logger.debug("Team Spend transactions: %s", team_list_transactions)
        if team_list_transactions is not None and len(team_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, "LiteLLM_TeamTable", outer_transaction=outer_transaction
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            # Sort by team_id for consistent lock ordering across pods to prevent deadlocks.
                            for team_id, response_cost in sorted(team_list_transactions.items()):
                                verbose_proxy_logger.debug(
                                    "Updating spend for team id=%s by %s", team_id, response_cost
                                )
                                batcher.litellm_teamtable.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"team_id": team_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
        if on_table_committed is not None:
            on_table_committed("team_list_transactions")

        ### UPDATE TEAM Membership TABLE with spend ###
        team_member_list_transactions: Final = db_spend_update_transactions["team_member_list_transactions"]
        verbose_proxy_logger.debug("Team Membership Spend transactions: %s", team_member_list_transactions)
        if team_member_list_transactions is not None and len(team_member_list_transactions.keys()) > 0:
            # Track which team memberships will be updated for cache invalidation
            team_memberships_to_invalidate: Final[list[tuple[str, str]]] = []
            for key in team_member_list_transactions:
                # key is "team_id::<value>::user_id::<value>"
                team_id = key.split("::")[1]
                user_id = key.split("::")[3]
                team_memberships_to_invalidate.append((user_id, team_id))

            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, "LiteLLM_TeamMembership", outer_transaction=outer_transaction
                    ) as transaction:
                        await _write_team_member_spend(transaction, team_member_list_transactions)
                    # Transaction succeeded, break out of retry loop
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
            if on_table_committed is not None:
                on_table_committed("team_member_list_transactions")

            # Invalidate cache for updated team memberships
            # This ensures budget checks read fresh spend data from the database
            if outer_transaction is None and team_memberships_to_invalidate and proxy_logging_obj is not None:
                user_api_key_cache: Final = proxy_logging_obj.call_details.get("user_api_key_cache")
                if user_api_key_cache is not None:
                    with service_target(AUTH_OBJECTS_TARGET):
                        for user_id, team_id in team_memberships_to_invalidate:
                            cache_key = f"team_membership:{user_id}:{team_id}"
                            await user_api_key_cache.async_delete_cache(key=cache_key)
                            verbose_proxy_logger.debug(
                                "Invalidated team membership cache for user_id=%s, team_id=%s", user_id, team_id
                            )
        elif on_table_committed is not None:
            on_table_committed("team_member_list_transactions")

        ### UPDATE ORG TABLE ###
        org_list_transactions: Final = db_spend_update_transactions["org_list_transactions"]
        verbose_proxy_logger.debug("Org Spend transactions: %s", org_list_transactions)
        if org_list_transactions is not None and len(org_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, "LiteLLM_OrganizationTable", outer_transaction=outer_transaction
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            # Sort by org_id for consistent lock ordering across pods to prevent deadlocks.
                            for org_id, response_cost in sorted(org_list_transactions.items()):
                                batcher.litellm_organizationtable.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"organization_id": org_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
        if on_table_committed is not None:
            on_table_committed("org_list_transactions")

        org_member_list_transactions: Final = db_spend_update_transactions.get("org_member_list_transactions")
        verbose_proxy_logger.debug("Org Membership Spend transactions: %s", org_member_list_transactions)
        if org_member_list_transactions is not None and len(org_member_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with (
                        _spend_update_tx(
                            prisma_client, "LiteLLM_OrganizationMembership", outer_transaction=outer_transaction
                        ) as transaction,
                        transaction.batch_() as batcher,
                    ):
                        for key, response_cost in sorted(org_member_list_transactions.items()):
                            _, quoted_org_id, _, quoted_user_id = key.split("::")
                            batcher.litellm_organizationmembership.update_many(
                                where={"organization_id": unquote(quoted_org_id), "user_id": unquote(quoted_user_id)},
                                data={"spend": {"increment": response_cost}},
                            )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
        if on_table_committed is not None:
            on_table_committed("org_member_list_transactions")

        ### UPDATE PROJECT TABLE ###
        project_list_transactions: Final = db_spend_update_transactions.get("project_list_transactions")
        await DBSpendUpdateWriter._update_entity_spend_in_db(
            entity_name="Project",
            transactions=project_list_transactions,
            outer_transaction=outer_transaction,
            table_accessor="litellm_projecttable",
            where_field="project_id",
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
        )
        if on_table_committed is not None:
            on_table_committed("project_list_transactions")
        if outer_transaction is None:
            await DBSpendUpdateWriter._invalidate_project_caches(
                project_ids=tuple(project_list_transactions or ()),
                proxy_logging_obj=proxy_logging_obj,
            )

        ### UPDATE TAG TABLE ###
        tag_list_transactions: Final = db_spend_update_transactions["tag_list_transactions"]
        await DBSpendUpdateWriter._update_entity_spend_in_db(
            entity_name="Tag",
            transactions=tag_list_transactions,
            outer_transaction=outer_transaction,
            table_accessor="litellm_tagtable",
            where_field="tag_name",
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
        )
        if on_table_committed is not None:
            on_table_committed("tag_list_transactions")

        ### UPDATE MODEL ACCESS GROUP TABLE ###
        model_access_group_list_transactions: Final = db_spend_update_transactions.get(
            "model_access_group_list_transactions"
        )
        await DBSpendUpdateWriter._update_entity_spend_in_db(
            entity_name="Model access group",
            transactions=model_access_group_list_transactions,
            outer_transaction=outer_transaction,
            table_accessor="litellm_modelaccessgroupbudgettable",
            where_field="access_group_name",
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
        )
        if on_table_committed is not None:
            on_table_committed("model_access_group_list_transactions")

        ### UPDATE AGENT TABLE ###
        agent_list_transactions: Final = db_spend_update_transactions["agent_list_transactions"]
        await DBSpendUpdateWriter._update_entity_spend_in_db(
            entity_name="Agent",
            transactions=agent_list_transactions,
            outer_transaction=outer_transaction,
            table_accessor="litellm_agentstable",
            where_field="agent_id",
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
        )
        if on_table_committed is not None:
            on_table_committed("agent_list_transactions")

    @staticmethod
    @with_service_target(AUTH_OBJECTS_TARGET)
    async def _invalidate_project_caches(project_ids: Sequence[str], proxy_logging_obj: ProxyLogging | None) -> None:
        if not project_ids or proxy_logging_obj is None:
            return
        user_api_key_cache: Final = proxy_logging_obj.call_details.get("user_api_key_cache")
        if user_api_key_cache is None:
            return
        for project_id in project_ids:
            await user_api_key_cache.async_delete_cache(key=project_cache_key(project_id))

    @staticmethod
    async def _update_entity_spend_in_db(
        entity_name: str,
        transactions: dict[str, float] | None,
        table_accessor: _EntitySpendTable,
        where_field: str,
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        outer_transaction: _SpendTransaction | None = None,
    ):
        """
        Helper function to update spend for any entity type (team, org, tag, etc).

        Args:
            entity_name: Name of entity for logging (e.g., "Team", "Org", "Tag")
            transactions: Dictionary of {entity_id: response_cost}
            table_accessor: Prisma table accessor (e.g., prisma_client.db.litellm_teamtable)
            where_field: Field name for where clause (e.g., "team_id", "organization_id", "tag_name")
            n_retry_times: Number of retries on failure
            prisma_client: Prisma client instance
            proxy_logging_obj: Proxy logging object
            outer_transaction: Existing ledger transaction owning the complete delivery update
        """
        verbose_proxy_logger.debug("%s Spend transactions: %s", entity_name, transactions)
        if transactions is not None and len(transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, _ENTITY_SPEND_MODELS[table_accessor], outer_transaction=outer_transaction
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            # Sort by entity_id for consistent lock ordering across pods to prevent deadlocks.
                            for entity_id, response_cost in sorted(transactions.items()):
                                verbose_proxy_logger.debug(
                                    "Updating spend for %s %s=%s by %s",
                                    entity_name,
                                    where_field,
                                    entity_id,
                                    response_cost,
                                )
                                _entity_spend_table(batcher, table_accessor).update_many(
                                    where={where_field: entity_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await DBSpendUpdateWriter.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )

    # fmt: off


    # fmt: on

    def _record_request_error_safely(self, *, payload: SpendLogsPayload, prisma_client: PrismaClient | None) -> None:
        try:
            self._record_request_error(payload=payload, prisma_client=prisma_client)
        except Exception as error:  # noqa: BLE001  # error rollup failure must not skip sibling writes
            _report_spend_update_step_failure("_record_request_error", payload=payload, exc=error)

    @staticmethod
    def _record_request_error(*, payload: SpendLogsPayload, prisma_client: PrismaClient | None) -> None:
        if prisma_client is None:
            return
        start_time: Final = payload["startTime"]
        date: Final = start_time.isoformat() if isinstance(start_time, datetime) else str(start_time or "")
        if not date:
            return
        metadata: Final = _parse_spend_log_metadata(payload)
        internal_origin: Final = bool(metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY))  # pyright: ignore[reportUnknownMemberType]  # TypedDict.get overloads carry Any defaults
        request_error_accumulator.record(
            payload=payload,
            request_status=prisma_client.get_request_status(payload),  # pyright: ignore[reportUnknownMemberType]  # legacy payload union; this caller supplies a typed spend payload
            date=date.split("T")[0],
            is_internal_call=internal_origin or _is_billed_retry_attempt(metadata),
        )
