import json
from typing import Final

import pytest

from litellm.accounting.payload import AccountingPayload, charge_kind, spend_log_columns


def test_odd_caller_data_is_normalized_so_the_charge_still_validates() -> None:
    payload: Final = AccountingPayload.model_validate_json(
        json.dumps(
            {
                "request_id": "r",
                "spend": 0.25,
                "prompt_tokens": "10",
                "request_tags": '["tag", 7, null, "other"]',
                "request_duration_ms": "1234",
                "metadata": {
                    "usage_object": "not a mapping",
                    "guardrail_information": {"guardrail_id": "bare-dict"},
                    "billed_retry_attempt": "yes",
                    "accounting_model_access_groups": ["pool", 3],
                },
            }
        )
    )

    assert payload.spend == 0.25
    assert payload.prompt_tokens == 0
    assert payload.request_tags == ("tag", "other")
    assert payload.request_duration_ms is None
    assert payload.metadata.usage_object is None
    assert payload.metadata.guardrail_information == ({"guardrail_id": "bare-dict"},)
    assert payload.metadata.billed_retry_attempt is False
    assert payload.metadata.model_access_groups == ("pool",)


@pytest.mark.parametrize("metadata", ['{"status": "failure"}', "not json", ["a list"]])
def test_metadata_stored_as_text_or_junk_reads_as_a_mapping(metadata: object) -> None:
    payload: Final = AccountingPayload.model_validate({"request_id": "r", "metadata": metadata})

    assert payload.metadata.status == ("failure" if metadata == '{"status": "failure"}' else None)


def test_a_stored_payload_round_trips_under_the_keys_the_ledger_writes() -> None:
    stored: Final = {
        "request_id": "r",
        "startTime": "2026-11-02T00:00:00+00:00",
        "metadata": {
            "accounting_entities": {"api_key": "hashed", "team_id": "team"},
            "accounting_savings": {"compression": 0.5},
            "accounting_windows": [
                {"entity_type": "key", "entity_id": "hashed", "window_duration": "7d", "window_start": "2026-11-01"}
            ],
        },
    }

    payload: Final = AccountingPayload.model_validate(stored)

    assert AccountingPayload.model_validate(payload.model_dump(mode="json", by_alias=True)) == payload
    assert payload.start_time == "2026-11-02T00:00:00+00:00"
    assert (payload.metadata.entities.api_key, payload.metadata.entities.team_id) == ("hashed", "team")
    assert payload.metadata.savings.compression == 0.5
    assert payload.metadata.windows[0].window_duration == "7d"


@pytest.mark.parametrize(
    ("row", "kind"),
    [
        ({"request_id": "r", "accounting_payload": {}, "accounting_delivery_id": "d"}, "request"),
        ({"request_id": "batch_batch_cost", "accounting_payload": {}}, "job"),
        ({"request_id": "r"}, "log_only"),
    ],
)
def test_every_buffered_row_is_classified_by_what_it_is_owed(row: dict[str, object], kind: str) -> None:
    assert charge_kind(row) == kind


def test_spend_log_columns_leave_out_only_the_ledger_charge() -> None:
    row: Final = {
        "request_id": "r",
        "spend": 1.0,
        "accounting_payload": {},
        "accounting_delivery_id": "d",
        "accounting_log_disabled": False,
    }

    assert spend_log_columns(row) == {"request_id": "r", "spend": 1.0}
