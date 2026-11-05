import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

from litellm.accounting.delivery_projection import project_deliveries
from litellm.accounting.payload import AccountingPayload
from litellm.accounting.projection import Contribution

_NOW: Final = datetime(2026, 11, 2, tzinfo=timezone.utc)


def _payload(spend: float, started: datetime = _NOW, window_start: datetime = _NOW) -> Mapping[str, object]:
    return {
        "request_id": f"request-{spend}-{started.isoformat()}",
        "startTime": started,
        "api_key": "key",
        "user": "user/1",
        "team_id": "team",
        "organization_id": "org::1",
        "spend": spend,
        "metadata": json.dumps(
            {
                "accounting_entities": {
                    "api_key": "key",
                    "user": "user/1",
                    "team_id": "team",
                    "organization_id": "org::1",
                },
                "accounting_windows": [
                    {
                        "entity_type": "key",
                        "entity_id": "key",
                        "window_duration": "1d",
                        "window_start": window_start.isoformat(),
                    }
                ],
            }
        ),
    }


def _spend(contributions: tuple[Contribution, ...], table: str) -> tuple[float, ...]:
    return tuple(float(dict(item.increments)["spend"]) for item in contributions if item.table == table)


def test_deliveries_for_one_destination_sum_and_keep_the_earliest_request_start() -> None:
    later: Final = _NOW + timedelta(hours=1)

    contributions: Final = project_deliveries(
        (
            AccountingPayload.model_validate(_payload(0.5, started=later)),
            AccountingPayload.model_validate(_payload(0.25)),
        )
    )

    assert _spend(contributions, "LiteLLM_VerificationToken") == (0.75,)
    windows: Final = tuple(item for item in contributions if item.table == "LiteLLM_BudgetWindowSpend")
    assert len(windows) == 1
    assert dict(windows[0].increments) == {"spend": 0.75}
    assert dict(windows[0].values)["started_at"] == str(_NOW)


def test_the_same_window_in_a_new_period_is_charged_to_its_own_row() -> None:
    next_day: Final = _NOW + timedelta(days=1)

    contributions: Final = project_deliveries(
        (
            AccountingPayload.model_validate(_payload(0.25)),
            AccountingPayload.model_validate(_payload(0.5, window_start=next_day)),
        )
    )

    windows: Final = sorted(
        (dict(item.values)["window_start"], dict(item.increments)["spend"])
        for item in contributions
        if item.table == "LiteLLM_BudgetWindowSpend"
    )
    assert windows == [(_NOW.isoformat(), 0.25), (next_day.isoformat(), 0.5)]
