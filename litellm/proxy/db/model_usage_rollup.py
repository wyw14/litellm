from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from typing_extensions import LiteralString


class _UpsertTable(Protocol):
    def upsert(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None: ...


class ModelUsageBatch(Protocol):
    @property
    def litellm_dailymodelusage(self) -> _UpsertTable: ...

    def execute_raw(self, query: LiteralString, *args: object) -> None: ...


def date_from_start_time(start_time: datetime | str) -> str | None:
    if isinstance(start_time, datetime):
        return start_time.date().isoformat()
    return start_time[:10] if len(start_time) >= 10 else None


def queue_model_usage_upsert(
    batcher: ModelUsageBatch, key_fields: Mapping[str, str], increments: Mapping[str, int | float]
) -> None:
    batcher.litellm_dailymodelusage.upsert(
        where={"date_model_group_model_custom_llm_provider_task_type": key_fields},
        data={
            "create": {**key_fields, **increments},
            "update": {name: {"increment": value} for name, value in increments.items()},
        },
    )
