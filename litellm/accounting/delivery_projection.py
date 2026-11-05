from collections.abc import Iterable
from itertools import chain, groupby
from typing import Final

from litellm.accounting.payload import AccountingPayload
from litellm.accounting.projection import Contribution, project


def _destination(contribution: Contribution) -> tuple[str, tuple[tuple[str, str], ...], str]:
    return (
        contribution.table,
        contribution.key,
        (dict(contribution.values).get("window_start") or "")
        if contribution.table == "LiteLLM_BudgetWindowSpend"
        else "",
    )


def _sum(contributions: tuple[Contribution, ...]) -> Contribution:
    first: Final = contributions[0]
    increments: Final = tuple(
        (column, sum(dict(item.increments).get(column, 0) for item in contributions))
        for column in dict.fromkeys(
            column for column, _ in chain.from_iterable(item.increments for item in contributions)
        )
    )
    starts: Final = tuple(
        value
        for column, value in chain.from_iterable(item.values for item in contributions)
        if column == "started_at" and value
    )
    values: Final = tuple(
        (column, min(starts) if column == "started_at" and starts else value) for column, value in first.values
    )
    return Contribution(first.table, first.key, increments, values)


def project_deliveries(deliveries: Iterable[AccountingPayload]) -> tuple[Contribution, ...]:
    ordered: Final = sorted(chain.from_iterable(project(delivery) for delivery in deliveries), key=_destination)
    return tuple(_sum(tuple(group)) for _, group in groupby(ordered, key=_destination))
