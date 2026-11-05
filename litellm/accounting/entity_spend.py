import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import TypeAdapter
from typing_extensions import LiteralString, ReadOnly, TypedDict

from litellm._internal_context import service_target
from litellm.accounting.projection import Contribution
from litellm.proxy.common_utils.user_api_key_cache import AUTH_OBJECTS_TARGET, project_cache_key
from litellm.proxy.db.db_span import db_span
from litellm.repositories.prisma_protocols import BatchTable

if TYPE_CHECKING:
    from litellm.proxy.utils import ProxyLogging


class SpendUpsertTable(Protocol):
    def upsert(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None: ...


class SpendBatch(Protocol):
    litellm_usertable: BatchTable
    litellm_endusertable: SpendUpsertTable
    litellm_dailymodelusage: SpendUpsertTable
    litellm_verificationtoken: BatchTable
    litellm_teamtable: BatchTable
    litellm_organizationtable: BatchTable
    litellm_organizationmembership: BatchTable
    litellm_projecttable: BatchTable
    litellm_tagtable: BatchTable
    litellm_agentstable: BatchTable
    litellm_modelaccessgroupbudgettable: BatchTable

    def execute_raw(self, query: LiteralString, *args: object) -> None: ...


class SpendBatchManager(Protocol):
    async def __aenter__(self) -> SpendBatch: ...

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...


class EntitySpendTransaction(Protocol):
    def batch_(self) -> SpendBatchManager: ...

    async def execute_raw(self, query: LiteralString, *args: object) -> int: ...


_CALL_DETAILS: Final = TypeAdapter(Mapping[str, object])


class _MemberSpendRow(TypedDict):
    user_id: ReadOnly[str]
    team_id: ReadOnly[str]
    cost: ReadOnly[float]


# The per-team advisory lock the team endpoints hold while changing a roster (TEAM_ADVISORY_LOCK_SQL),
# so the roster check below cannot interleave with their writes. A row lock would deadlock with the
# access-group endpoints, which lock a team row after an access-group lock.
TEAM_ADVISORY_LOCK_SQL: Final = "SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS locked"

# A missing membership row is created only while the user is still on the team's roster, so a charge
# applied after a removal never recreates the member. Rows travel as one JSON document because Prisma
# types a raw array parameter from the first batch a connection sees, so an all-integer batch broke
# every later fractional one. total_spend is floored at spend: rows from before total_spend existed
# (migration 20260421135425) carry their history only in spend.
TEAM_MEMBER_SPEND_SQL: Final = """
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

_SpendRow = tuple[Mapping[str, str], float]
_Write = Callable[[SpendBatch, Mapping[str, str], float], None]


def _increment(table: Callable[[SpendBatch], BatchTable]) -> _Write:
    def write(batcher: SpendBatch, key: Mapping[str, str], spend: float) -> None:
        table(batcher).update_many(where=dict(key), data={"spend": {"increment": spend}})

    return write


def _key_spend(batcher: SpendBatch, key: Mapping[str, str], spend: float) -> None:
    batcher.litellm_verificationtoken.update_many(
        where=dict(key),
        data={
            "spend": {"increment": spend},
            "total_spend": {"increment": spend},
            "last_active": datetime.now(timezone.utc),
        },
    )


def _end_user_spend(batcher: SpendBatch, key: Mapping[str, str], spend: float) -> None:
    batcher.litellm_endusertable.upsert(
        where=dict(key),
        data={
            "create": {"user_id": key["user_id"], "spend": spend, "blocked": False},
            "update": {"spend": {"increment": spend}},
        },
    )


# Tables are written in this order and rows in key order inside each, so every writer of these rows
# takes its locks in the same sequence.
_BATCHED_WRITES: Final[tuple[tuple[str, _Write], ...]] = (
    ("LiteLLM_UserTable", _increment(lambda batcher: batcher.litellm_usertable)),
    ("LiteLLM_EndUserTable", _end_user_spend),
    ("LiteLLM_VerificationToken", _key_spend),
    ("LiteLLM_TeamTable", _increment(lambda batcher: batcher.litellm_teamtable)),
)
_BATCHED_WRITES_AFTER_MEMBERS: Final[tuple[tuple[str, _Write], ...]] = (
    ("LiteLLM_OrganizationTable", _increment(lambda batcher: batcher.litellm_organizationtable)),
    ("LiteLLM_OrganizationMembership", _increment(lambda batcher: batcher.litellm_organizationmembership)),
    ("LiteLLM_ProjectTable", _increment(lambda batcher: batcher.litellm_projecttable)),
    ("LiteLLM_TagTable", _increment(lambda batcher: batcher.litellm_tagtable)),
    ("LiteLLM_ModelAccessGroupBudgetTable", _increment(lambda batcher: batcher.litellm_modelaccessgroupbudgettable)),
    ("LiteLLM_AgentsTable", _increment(lambda batcher: batcher.litellm_agentstable)),
)


def _rows(contributions: Sequence[Contribution], table: str) -> tuple[_SpendRow, ...]:
    return tuple(
        sorted(
            (
                (MappingProxyType(dict(item.key)), float(dict(item.increments)["spend"]))
                for item in contributions
                if item.table == table
            ),
            key=lambda row: tuple(row[0].values()),
        )
    )


async def _write_batched(
    transaction: EntitySpendTransaction, contributions: Sequence[Contribution], writes: tuple[tuple[str, _Write], ...]
) -> None:
    for table, write in writes:
        rows = _rows(contributions, table)
        if not rows:
            continue
        async with db_span("commit_spend_updates", table), transaction.batch_() as batcher:
            for key, spend in rows:
                write(batcher, key, spend)


async def _write_team_members(transaction: EntitySpendTransaction, contributions: Sequence[Contribution]) -> None:
    members: Final = tuple(
        _MemberSpendRow(user_id=key["user_id"], team_id=key["team_id"], cost=spend)
        for key, spend in sorted(
            _rows(contributions, "LiteLLM_TeamMembership"), key=lambda row: (row[0]["team_id"], row[0]["user_id"])
        )
    )
    if not members:
        return
    async with db_span("commit_spend_updates", "LiteLLM_TeamMembership"):
        for team_id in dict.fromkeys(member["team_id"] for member in members):
            _ = await transaction.execute_raw(TEAM_ADVISORY_LOCK_SQL, team_id)
        _ = await transaction.execute_raw(TEAM_MEMBER_SPEND_SQL, json.dumps(members))


async def apply_entity_spend(transaction: EntitySpendTransaction, contributions: Sequence[Contribution]) -> None:
    await _write_batched(transaction, contributions, _BATCHED_WRITES)
    await _write_team_members(transaction, contributions)
    await _write_batched(transaction, contributions, _BATCHED_WRITES_AFTER_MEMBERS)


async def invalidate_entity_caches(contributions: Sequence[Contribution], proxy_logging_obj: "ProxyLogging") -> None:
    """Budget checks read project and team membership spend from the auth cache, so those entries go once the
    charge commits."""
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    details: Final = _CALL_DETAILS.validate_python(
        proxy_logging_obj.call_details  # pyright: ignore[reportUnknownMemberType]  # untyped proxy logging dict, validated here
    )
    cache: Final = details.get("user_api_key_cache")
    if not isinstance(cache, UserApiKeyCache):
        return
    project_keys: Final = (
        project_cache_key(dict(item.key)["project_id"])
        for item in contributions
        if item.table == "LiteLLM_ProjectTable"
    )
    membership_keys: Final = (
        f"team_membership:{dict(item.key)['user_id']}:{dict(item.key)['team_id']}"
        for item in contributions
        if item.table == "LiteLLM_TeamMembership"
    )
    keys: Final = tuple(dict.fromkeys((*project_keys, *membership_keys)))
    with service_target(AUTH_OBJECTS_TARGET):
        for key in keys:
            await cache.async_delete_cache(key=key)
