import json
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Final

import pytest

from litellm.accounting.entity_spend import (
    TEAM_ADVISORY_LOCK_SQL,
    TEAM_MEMBER_SPEND_SQL,
    apply_entity_spend,
    invalidate_entity_caches,
)
from litellm.accounting.projection import Contribution
from litellm.caching.dual_cache import DualCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, project_cache_key


class _Table:
    def __init__(self, name: str, log: list[tuple[str, str, dict, dict]]) -> None:
        self._name = name
        self._log = log

    def update_many(self, where: dict, data: dict) -> None:
        self._log.append((self._name, "update_many", where, data))

    def upsert(self, where: dict, data: dict) -> None:
        self._log.append((self._name, "upsert", where, data))


class _Batch:
    def __init__(self, log: list[tuple[str, str, dict, dict]]) -> None:
        self._log = log

    def __getattr__(self, name: str) -> _Table:
        return _Table(name, self._log)


class _Transaction:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str, dict, dict]] = []
        self.raw: list[tuple[str, tuple[object, ...]]] = []

    @asynccontextmanager
    async def _batch(self) -> AsyncIterator[_Batch]:
        yield _Batch(self.writes)

    def batch_(self) -> AbstractAsyncContextManager[_Batch]:
        return self._batch()

    async def execute_raw(self, query: str, *args: object) -> int:
        self.raw.append((query, args))
        return 1


def _spend(table: str, spend: float, **key: str) -> Contribution:
    return Contribution(table, tuple(key.items()), (("spend", spend),))


@pytest.mark.asyncio
async def test_every_entity_table_is_incremented_in_lock_order_by_key() -> None:
    transaction: Final = _Transaction()

    await apply_entity_spend(
        transaction,
        (
            _spend("LiteLLM_AgentsTable", 0.5, agent_id="agent"),
            _spend("LiteLLM_TeamTable", 0.5, team_id="team-b"),
            _spend("LiteLLM_TeamTable", 0.25, team_id="team-a"),
            _spend("LiteLLM_VerificationToken", 0.5, token="hashed"),
            _spend("LiteLLM_EndUserTable", 0.5, user_id="customer"),
            _spend("LiteLLM_UserTable", 0.5, user_id="user"),
            _spend("LiteLLM_OrganizationTable", 0.5, organization_id="org"),
            _spend("LiteLLM_OrganizationMembership", 0.5, organization_id="org::x", user_id="user"),
            _spend("LiteLLM_ProjectTable", 0.5, project_id="project"),
            _spend("LiteLLM_TagTable", 0.5, tag_name="tag"),
            _spend("LiteLLM_ModelAccessGroupBudgetTable", 0.5, access_group_name="pool"),
        ),
    )

    assert [(table, method, where) for table, method, where, _ in transaction.writes] == [
        ("litellm_usertable", "update_many", {"user_id": "user"}),
        ("litellm_endusertable", "upsert", {"user_id": "customer"}),
        ("litellm_verificationtoken", "update_many", {"token": "hashed"}),
        ("litellm_teamtable", "update_many", {"team_id": "team-a"}),
        ("litellm_teamtable", "update_many", {"team_id": "team-b"}),
        ("litellm_organizationtable", "update_many", {"organization_id": "org"}),
        ("litellm_organizationmembership", "update_many", {"organization_id": "org::x", "user_id": "user"}),
        ("litellm_projecttable", "update_many", {"project_id": "project"}),
        ("litellm_tagtable", "update_many", {"tag_name": "tag"}),
        ("litellm_modelaccessgroupbudgettable", "update_many", {"access_group_name": "pool"}),
        ("litellm_agentstable", "update_many", {"agent_id": "agent"}),
    ]
    data: Final = {(table, json.dumps(where)): payload for table, _, where, payload in transaction.writes}
    assert data[("litellm_teamtable", '{"team_id": "team-a"}')] == {"spend": {"increment": 0.25}}
    assert data[("litellm_endusertable", '{"user_id": "customer"}')] == {
        "create": {"user_id": "customer", "spend": 0.5, "blocked": False},
        "update": {"spend": {"increment": 0.5}},
    }
    key_data: Final = data[("litellm_verificationtoken", '{"token": "hashed"}')]
    assert set(key_data) == {"spend", "total_spend", "last_active"}
    assert key_data["spend"] == key_data["total_spend"] == {"increment": 0.5}


@pytest.mark.asyncio
async def test_team_members_are_charged_in_one_roster_checked_upsert_after_locking_each_team_in_order() -> None:
    transaction: Final = _Transaction()

    await apply_entity_spend(
        transaction,
        (
            _spend("LiteLLM_TeamMembership", 0.5, team_id="team-b", user_id="user-1"),
            _spend("LiteLLM_TeamMembership", 0.25, team_id="team-a", user_id="user-2"),
            _spend("LiteLLM_TeamMembership", 0.125, team_id="team-a", user_id="user-1"),
        ),
    )

    assert transaction.raw[:2] == [(TEAM_ADVISORY_LOCK_SQL, ("team-a",)), (TEAM_ADVISORY_LOCK_SQL, ("team-b",))]
    (query, (members,)) = transaction.raw[2]
    assert query == TEAM_MEMBER_SPEND_SQL
    assert json.loads(str(members)) == [
        {"user_id": "user-1", "team_id": "team-a", "cost": 0.125},
        {"user_id": "user-2", "team_id": "team-a", "cost": 0.25},
        {"user_id": "user-1", "team_id": "team-b", "cost": 0.5},
    ]
    assert transaction.writes == []


@pytest.mark.asyncio
async def test_a_charge_with_no_entities_writes_nothing() -> None:
    transaction: Final = _Transaction()

    await apply_entity_spend(transaction, (Contribution("LiteLLM_DailyUserSpend", (), (("spend", 1.0),)),))

    assert (transaction.writes, transaction.raw) == ([], [])


class _ProxyLogging:
    def __init__(self, cache: object) -> None:
        self.call_details = {"user_api_key_cache": cache}


@pytest.mark.asyncio
async def test_committed_project_and_membership_charges_drop_their_cached_budget_reads() -> None:
    cache: Final = UserApiKeyCache(DualCache())
    kept: Final = "team_membership:user-1:other-team"
    for key in (project_cache_key("project"), "team_membership:user-1:team-a", kept):
        await cache.async_set_cache(key=key, value={"spend": 0})

    await invalidate_entity_caches(
        (
            _spend("LiteLLM_ProjectTable", 0.5, project_id="project"),
            _spend("LiteLLM_TeamMembership", 0.5, team_id="team-a", user_id="user-1"),
            _spend("LiteLLM_TeamTable", 0.5, team_id="team-a"),
        ),
        _ProxyLogging(cache),
    )

    assert await cache.async_get_cache(key=project_cache_key("project")) is None
    assert await cache.async_get_cache(key="team_membership:user-1:team-a") is None
    assert await cache.async_get_cache(key=kept) == {"spend": 0}
