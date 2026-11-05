"""
Local cache key diff tool.

Explains why two litellm.completion/embedding request kwargs produce the same
or different cache keys. Purely local: it builds keys with the existing cache
key logic but never reads or writes any cache backend. Reports contain field
names and equality flags only, never prompt content, credentials, or raw values.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from litellm.caching.caching import Cache

NAMESPACE_FIELD_NAME: Final = "namespace"
PRESET_CACHE_KEY_FIELD_NAME: Final = "preset_cache_key"


class CacheKeyFieldStatus(str, Enum):
    EQUAL = "equal"
    DIFFERENT = "different"
    ONLY_IN_FIRST = "only_in_first"
    ONLY_IN_SECOND = "only_in_second"


@dataclass(frozen=True, slots=True)
class CacheKeyFieldDiff:
    name: str
    status: CacheKeyFieldStatus


@dataclass(frozen=True, slots=True)
class CacheKeyDiff:
    keys_equal: bool
    fields: tuple[CacheKeyFieldDiff, ...]
    order_dependent: bool
    summary: str


def _is_preset_key(parts: tuple[tuple[str, str], ...]) -> bool:
    return len(parts) == 1 and parts[0][0] == PRESET_CACHE_KEY_FIELD_NAME


def _request_call_copy(request: Mapping[str, object]) -> dict[str, object]:
    litellm_params: Final = request.get("litellm_params")
    if isinstance(litellm_params, Mapping):
        typed_params: Final = cast(  # cast-ok: isinstance narrows to a Mapping but leaves key and value types unknown
            Mapping[str, object], litellm_params
        )
        return {**request, "litellm_params": dict(typed_params)}
    return dict(request)


def _ordered_field_names(
    first_parts: tuple[tuple[str, str], ...],
    second_parts: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for name, _ in (*first_parts, *second_parts)))


def _field_status(
    name: str,
    first_materials: Mapping[str, str],
    second_materials: Mapping[str, str],
) -> CacheKeyFieldStatus:
    in_first: Final = name in first_materials
    in_second: Final = name in second_materials
    if in_first and in_second:
        return (
            CacheKeyFieldStatus.EQUAL
            if first_materials[name] == second_materials[name]
            else CacheKeyFieldStatus.DIFFERENT
        )
    if in_first:
        return CacheKeyFieldStatus.ONLY_IN_FIRST
    return CacheKeyFieldStatus.ONLY_IN_SECOND


def _namespace_status(
    first_namespace: str | None,
    second_namespace: str | None,
) -> CacheKeyFieldStatus:
    if first_namespace == second_namespace:
        return CacheKeyFieldStatus.EQUAL
    if first_namespace is not None and second_namespace is not None:
        return CacheKeyFieldStatus.DIFFERENT
    if first_namespace is not None:
        return CacheKeyFieldStatus.ONLY_IN_FIRST
    return CacheKeyFieldStatus.ONLY_IN_SECOND


def _build_summary(
    keys_equal: bool,
    fields: tuple[CacheKeyFieldDiff, ...],
    order_dependent: bool,
) -> str:
    if keys_equal:
        return f"Cache keys match across {len(fields)} keyed field(s)."
    if order_dependent:
        return (
            "Cache keys differ only because the same keyed parameters appear in a different "
            "order. Key material is concatenated in request parameter order, so the hash changes."
        )

    different: Final = tuple(f.name for f in fields if f.status is CacheKeyFieldStatus.DIFFERENT)
    only_first: Final = tuple(f.name for f in fields if f.status is CacheKeyFieldStatus.ONLY_IN_FIRST)
    only_second: Final = tuple(f.name for f in fields if f.status is CacheKeyFieldStatus.ONLY_IN_SECOND)
    no_differences: Final = not different and not only_first and not only_second

    fragments: Final = tuple(
        fragment
        for fragment in (
            "Cache keys differ.",
            f"Fields with different values: {', '.join(different)}." if different else "",
            f"Fields present only in the first request: {', '.join(only_first)}." if only_first else "",
            f"Fields present only in the second request: {', '.join(only_second)}." if only_second else "",
            "All compared fields match." if no_differences else "",
        )
        if fragment
    )
    return " ".join(fragments)


def explain_cache_key_diff(
    cache: "Cache",
    first_request: Mapping[str, object],
    second_request: Mapping[str, object],
) -> CacheKeyDiff:
    """
    Compare the cache key produced by two local request kwargs.

    Returns the name of every field that feeds either key, whether that field
    is equal across the two requests, whether the final keys match, and a
    value-free summary. Final keys are computed locally with the same logic as
    ``Cache.get_cache_key`` without accessing the cache backend, and the input
    mappings are not mutated.
    """
    first_parts: Final = cache.get_cache_key_parts(first_request)
    second_parts: Final = cache.get_cache_key_parts(second_request)

    first_key: Final = cache.get_cache_key(**_request_call_copy(first_request))
    second_key: Final = cache.get_cache_key(**_request_call_copy(second_request))
    keys_equal: Final = first_key == second_key

    first_materials: Final = dict(first_parts)
    second_materials: Final = dict(second_parts)
    field_names: Final = _ordered_field_names(first_parts, second_parts)
    body_fields: Final = tuple(
        CacheKeyFieldDiff(name=name, status=_field_status(name, first_materials, second_materials))
        for name in field_names
    )

    includes_namespace: Final = not _is_preset_key(first_parts) and not _is_preset_key(second_parts)
    first_namespace: Final = cache.resolve_cache_namespace(first_request) if includes_namespace else None
    second_namespace: Final = cache.resolve_cache_namespace(second_request) if includes_namespace else None
    namespace_fields: Final = (
        (
            CacheKeyFieldDiff(
                name=NAMESPACE_FIELD_NAME,
                status=_namespace_status(first_namespace, second_namespace),
            ),
        )
        if includes_namespace and (first_namespace is not None or second_namespace is not None)
        else ()
    )

    fields: Final = (*body_fields, *namespace_fields)
    order_dependent: Final = (
        not keys_equal and first_parts != second_parts and sorted(first_parts) == sorted(second_parts)
    )
    return CacheKeyDiff(
        keys_equal=keys_equal,
        fields=fields,
        order_dependent=order_dependent,
        summary=_build_summary(keys_equal, fields, order_dependent),
    )
