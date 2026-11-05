import dataclasses

import pytest

from litellm.caching.cache_key_diff import CacheKeyFieldStatus
from litellm.caching.caching import Cache

PROMPT_MARKER = "prompt-marker-3a9f1c"
SECRET_MARKER = "sk-secret-marker-7e2b55"
USER_MARKER = "end-user-marker-9d10"


@pytest.fixture
def cache() -> Cache:
    return Cache()


def _statuses(result) -> dict[str, CacheKeyFieldStatus]:
    return {field.name: field.status for field in result.fields}


def _serialized(result) -> str:
    return repr(result) + repr(dataclasses.asdict(result)) + result.summary


def _request(model: str = "gpt-4o", **overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "same prompt"}],
    }
    request.update(overrides)
    return request


def test_identical_requests_match_with_every_field_equal(cache: Cache) -> None:
    result = cache.explain_cache_key_diff(_request(temperature=0.7), _request(temperature=0.7))

    assert result.keys_equal is True
    assert set(_statuses(result).values()) == {CacheKeyFieldStatus.EQUAL}
    assert "messages" in _statuses(result)


def test_model_alias_string_changes_cache_key_without_router_group(cache: Cache) -> None:
    result = cache.explain_cache_key_diff(_request(), _request(model="openai/gpt-4o"))

    assert result.keys_equal is False
    statuses = _statuses(result)
    assert statuses["model"] is CacheKeyFieldStatus.DIFFERENT
    assert statuses["messages"] is CacheKeyFieldStatus.EQUAL


def test_router_model_group_aliases_share_cache_key(cache: Cache) -> None:
    first = _request(model="deployment-a", metadata={"model_group": "gpt-4o"})
    second = _request(model="deployment-b", metadata={"model_group": "gpt-4o"})

    result = cache.explain_cache_key_diff(first, second)

    assert result.keys_equal is True
    assert _statuses(result)["model"] is CacheKeyFieldStatus.EQUAL


@pytest.mark.parametrize(
    "namespace_kwargs",
    [
        lambda namespace: {"cache": {"namespace": namespace}},
        lambda namespace: {"metadata": {"redis_namespace": namespace}},
    ],
)
def test_namespace_prefix_changes_final_key(cache: Cache, namespace_kwargs) -> None:
    first = _request(**namespace_kwargs("team-a"))
    second = _request(**namespace_kwargs("team-b"))

    different = cache.explain_cache_key_diff(first, second)
    assert different.keys_equal is False
    assert _statuses(different)["namespace"] is CacheKeyFieldStatus.DIFFERENT
    assert "team-a" not in _serialized(different)
    assert "team-b" not in _serialized(different)

    same_namespace = cache.explain_cache_key_diff(first, _request(**namespace_kwargs("team-a")))
    assert same_namespace.keys_equal is True
    assert _statuses(same_namespace)["namespace"] is CacheKeyFieldStatus.EQUAL


def test_keyed_optional_param_order_changes_final_key(cache: Cache) -> None:
    first: dict[str, object] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "same prompt"}],
        "temperature": 0.7,
    }
    second: dict[str, object] = {
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "same prompt"}],
        "model": "gpt-4o",
    }

    result = cache.explain_cache_key_diff(first, second)

    assert result.keys_equal is False
    assert result.order_dependent is True
    assert set(_statuses(result).values()) == {CacheKeyFieldStatus.EQUAL}


def test_extra_body_keyed_param_insertion_order_does_not_change_key(cache: Cache) -> None:
    first = _request(extra_body={"continuation_token": "token-abc", "return_token_ids": True})
    second = _request(extra_body={"return_token_ids": True, "continuation_token": "token-abc"})

    result = cache.explain_cache_key_diff(first, second)

    assert result.keys_equal is True
    statuses = _statuses(result)
    assert statuses["extra_body.continuation_token"] is CacheKeyFieldStatus.EQUAL
    assert statuses["extra_body.return_token_ids"] is CacheKeyFieldStatus.EQUAL


def test_missing_optional_field_is_flagged_only_in_one_request(cache: Cache) -> None:
    result = cache.explain_cache_key_diff(_request(temperature=0.7), _request())

    assert result.keys_equal is False
    statuses = _statuses(result)
    assert statuses["temperature"] is CacheKeyFieldStatus.ONLY_IN_FIRST
    assert result.order_dependent is False


def test_prompt_difference_is_reported_without_exposing_prompt(cache: Cache) -> None:
    first = _request(messages=[{"role": "user", "content": PROMPT_MARKER}])
    second = _request(messages=[{"role": "user", "content": "different prompt text"}])

    result = cache.explain_cache_key_diff(first, second)

    assert result.keys_equal is False
    assert _statuses(result)["messages"] is CacheKeyFieldStatus.DIFFERENT
    assert PROMPT_MARKER not in _serialized(result)


def test_credentials_never_enter_the_key_or_the_report(cache: Cache) -> None:
    first = _request(api_key=SECRET_MARKER)
    second = _request(api_key="sk-a-completely-different-key-value")

    result = cache.explain_cache_key_diff(first, second)

    assert result.keys_equal is True
    assert "api_key" not in _statuses(result)
    assert SECRET_MARKER not in _serialized(result)


def test_raw_keyed_value_never_appears_in_report(cache: Cache) -> None:
    result = cache.explain_cache_key_diff(_request(user=USER_MARKER), _request(user="other-user"))

    assert result.keys_equal is False
    assert _statuses(result)["user"] is CacheKeyFieldStatus.DIFFERENT
    assert USER_MARKER not in _serialized(result)


def test_input_requests_are_not_mutated(cache: Cache) -> None:
    first_litellm_params: dict[str, object] = {}
    first = _request(temperature=0.7, litellm_params=first_litellm_params)
    first_messages = first["messages"]
    second = _request(temperature=0.7)

    cache.explain_cache_key_diff(first, second)

    assert first_litellm_params == {}
    assert "preset_cache_key" not in first_litellm_params
    assert first["messages"] is first_messages
    assert set(first.keys()) == {"model", "messages", "temperature", "litellm_params"}


class _RaisingBackend:
    def get_cache(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("diff tool must not read the cache backend")

    def set_cache(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("diff tool must not write the cache backend")


def test_diff_is_computed_without_cache_backend_access(cache: Cache) -> None:
    cache.cache = _RaisingBackend()

    result = cache.explain_cache_key_diff(_request(temperature=0.7), _request(temperature=0.9))

    assert result.keys_equal is False
    assert _statuses(result)["temperature"] is CacheKeyFieldStatus.DIFFERENT
