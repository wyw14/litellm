import asyncio
import json
import random
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from unittest.mock import AsyncMock

from litellm.caching.in_memory_cache import InMemoryCache, LocalCacheEvictionStats


class _SlowInt(int):
    def __add__(self, value: int) -> "_SlowInt":
        time.sleep(0.05)
        return _SlowInt(int(self) + value)


def test_increment_cache_is_atomic_under_thread_concurrency():
    cache = InMemoryCache()
    seed = 1000
    cache.set_cache("counter", _SlowInt(seed))
    thread_count = 8
    barrier = threading.Barrier(thread_count)

    def increment(_: int) -> float:
        barrier.wait()
        return cache.increment_cache("counter", 1)

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        tuple(executor.map(increment, range(thread_count)))

    assert cache.get_cache("counter") == seed + thread_count


async def test_async_increment_delegates_to_locked_sync_path():
    cache = InMemoryCache()
    assert await cache.async_increment("counter", 2) == 2
    assert await cache.async_increment("counter", 3) == 5
    assert cache.get_cache("counter") == 5


class _AnotherReaderExpiresTheEntryFirstCache(InMemoryCache):
    other_reader_started = False
    other_reader_result: object = "not run"

    def evict_element_if_expired(self, key: str) -> bool:
        if not self.other_reader_started:
            self.other_reader_started = True
            self.other_reader_result = self.get_cache(key)
        return super().evict_element_if_expired(key)


def test_get_cache_reads_an_entry_another_reader_just_expired_as_a_miss() -> None:
    cache: Final = _AnotherReaderExpiresTheEntryFirstCache()
    cache.set_cache("tokenizer_failure", "HF unavailable", ttl=-1)

    assert cache.get_cache("tokenizer_failure") is None
    assert cache.other_reader_result is None


class _ClockThatLetsAnotherReaderRunFirst:
    cache: InMemoryCache | None = None
    other_reader_result: object = "not run"

    def __call__(self) -> float:
        cache: Final = self.cache
        if cache is not None:
            self.cache = None
            self.other_reader_result = cache.get_cache("tokenizer_failure")
        return 100.0


def test_get_cache_reads_an_entry_another_reader_expires_during_the_expiry_check_as_a_miss() -> None:
    clock: Final = _ClockThatLetsAnotherReaderRunFirst()
    cache: Final = InMemoryCache(clock=clock)
    cache.set_cache("tokenizer_failure", "HF unavailable", ttl=-1)
    clock.cache = cache

    assert cache.get_cache("tokenizer_failure") is None
    assert clock.other_reader_result is None


def test_in_memory_openai_obj_cache():
    from openai import OpenAI

    openai_obj = OpenAI(api_key="my-fake-key")

    in_memory_cache = InMemoryCache()

    in_memory_cache.set_cache(key="my-fake-key", value=openai_obj)

    cached_obj = in_memory_cache.get_cache(key="my-fake-key")

    assert cached_obj is not None

    assert cached_obj == openai_obj


def test_in_memory_cache_max_size_per_item():
    """
    Test that the cache will not store items larger than the max size per item
    """
    in_memory_cache = InMemoryCache(max_size_per_item=100)

    result = in_memory_cache.check_value_size("a" * 100000000)

    assert result is False


def test_in_memory_cache_ttl():
    """
    Check that
    - if ttl is not set, it will be set to default ttl
    - if object expires, the ttl is also removed
    """
    in_memory_cache = InMemoryCache()

    in_memory_cache.set_cache(key="my-fake-key", value="my-fake-value", ttl=10)
    initial_ttl_time = in_memory_cache.ttl_dict["my-fake-key"]
    assert initial_ttl_time is not None

    in_memory_cache.set_cache(key="my-fake-key", value="my-fake-value-2", ttl=10)
    new_ttl_time = in_memory_cache.ttl_dict["my-fake-key"]
    assert new_ttl_time == initial_ttl_time  # ttl should not be updated

    ## On object expiration, the ttl should be removed
    in_memory_cache.set_cache(key="new-fake-key", value="new-fake-value", ttl=1)
    new_ttl_time = in_memory_cache.ttl_dict["new-fake-key"]
    assert new_ttl_time is not None
    time.sleep(1)
    cached_obj = in_memory_cache.get_cache(key="new-fake-key")
    new_ttl_time = in_memory_cache.ttl_dict.get("new-fake-key")
    assert new_ttl_time is None


def test_in_memory_cache_ttl_allow_override():
    """
    Check that
    - if ttl is not set, it will be set to default ttl
    - if object expires, the ttl is also removed
    """
    in_memory_cache = InMemoryCache()
    ## On object expiration, but no get_cache, the override should be allowed
    in_memory_cache.set_cache(key="new-fake-key", value="new-fake-value", ttl=1)
    initial_ttl_time = in_memory_cache.ttl_dict["new-fake-key"]
    assert initial_ttl_time is not None
    time.sleep(1)

    in_memory_cache.set_cache(key="new-fake-key", value="new-fake-value-2", ttl=1)
    new_ttl_time = in_memory_cache.ttl_dict["new-fake-key"]
    assert new_ttl_time is not None
    assert new_ttl_time != initial_ttl_time


def test_in_memory_cache_max_size_with_ttl():
    """
    Test that max_size_in_memory is respected even when all items have long TTLs.
    This tests the fix for the unbounded growth issue.
    """
    in_memory_cache = InMemoryCache(max_size_in_memory=3)
    long_ttl = 86400  # 1 day

    # Fill the cache to max capacity
    for i in range(3):
        in_memory_cache.set_cache(key=f"key_{i}", value=f"value_{i}", ttl=long_ttl)
        time.sleep(0.01)  # Small delay to ensure different timestamps

    assert len(in_memory_cache.cache_dict) == 3
    assert len(in_memory_cache.ttl_dict) == 3

    # Add another item - should evict the earliest item
    in_memory_cache.set_cache(key="key_3", value="value_3", ttl=long_ttl)

    # Cache should still be at max size, not larger
    assert len(in_memory_cache.cache_dict) == 3
    assert len(in_memory_cache.ttl_dict) == 3

    # key_0 should have been evicted (it was added first)
    assert "key_0" not in in_memory_cache.cache_dict
    assert "key_0" not in in_memory_cache.ttl_dict

    # Other keys should still be present
    assert "key_1" in in_memory_cache.cache_dict
    assert "key_2" in in_memory_cache.cache_dict
    assert "key_3" in in_memory_cache.cache_dict


def test_in_memory_cache_expired_items_evicted_first():
    """
    Test that expired items are evicted before non-expired items when cache is full.
    """
    in_memory_cache = InMemoryCache(max_size_in_memory=3)

    # Add items with short TTL that will expire
    in_memory_cache.set_cache(key="expired_1", value="value_1", ttl=1)
    in_memory_cache.set_cache(key="expired_2", value="value_2", ttl=1)

    # Add item with long TTL
    in_memory_cache.set_cache(key="long_lived", value="value_long", ttl=86400)

    assert len(in_memory_cache.cache_dict) == 3

    # Wait for short TTL items to expire
    time.sleep(2)

    # Add new item - should evict expired items first, not the long-lived one
    in_memory_cache.set_cache(key="new_item", value="new_value", ttl=86400)

    # Long-lived item should still be present
    assert "long_lived" in in_memory_cache.cache_dict
    assert "new_item" in in_memory_cache.cache_dict

    # Expired items should be gone
    assert "expired_1" not in in_memory_cache.cache_dict
    assert "expired_2" not in in_memory_cache.cache_dict
    assert "expired_1" not in in_memory_cache.ttl_dict
    assert "expired_2" not in in_memory_cache.ttl_dict


def test_in_memory_cache_eviction_order():
    """
    Test that when non-expired items need to be evicted, those with earliest expiration times are evicted first.
    """
    in_memory_cache = InMemoryCache(max_size_in_memory=2)

    # Add items with different TTLs
    now = time.time()
    in_memory_cache.set_cache(key="early_expire", value="value_1", ttl=100)  # expires in 100 seconds
    time.sleep(0.01)
    in_memory_cache.set_cache(key="late_expire", value="value_2", ttl=200)  # expires in 200 seconds

    # Verify TTL order
    early_ttl = in_memory_cache.ttl_dict["early_expire"]
    late_ttl = in_memory_cache.ttl_dict["late_expire"]
    assert early_ttl < late_ttl, "early_expire should have earlier expiration time"

    assert len(in_memory_cache.cache_dict) == 2

    # Add third item - should evict the one with earliest expiration time
    in_memory_cache.set_cache(key="new_item", value="value_3", ttl=300)

    assert len(in_memory_cache.cache_dict) == 2

    # Item with earliest expiration should be evicted
    assert "early_expire" not in in_memory_cache.cache_dict
    assert "early_expire" not in in_memory_cache.ttl_dict

    # Items with later expiration should remain
    assert "late_expire" in in_memory_cache.cache_dict
    assert "new_item" in in_memory_cache.cache_dict


def test_in_memory_cache_heap_size_staus_bounded():
    """
    Test that the expiration_heap does not grow unbounded when the same key is updated repeaatedly.
    """
    in_memory_cache = InMemoryCache(max_size_in_memory=10)

    for i in range(1_000):
        in_memory_cache.set_cache(key="hot_key", value=f"value_{i}", ttl=60)

    # Expiration heap should only have 1 entry
    assert len(in_memory_cache.expiration_heap) == 1


def test_in_memory_cache_heap_stays_bounded_when_a_live_key_is_rewritten_with_a_new_ttl() -> None:
    in_memory_cache = InMemoryCache(max_size_in_memory=10)
    in_memory_cache.set_cache(key="expires_first", value="pin", ttl=300)

    for i in range(1_000):
        in_memory_cache.delete_cache("hot_key")
        in_memory_cache.set_cache(key="hot_key", value=f"value_{i}", ttl=3600)

    assert in_memory_cache.get_cache("hot_key") == "value_999"
    assert len(in_memory_cache.expiration_heap) <= 2 * in_memory_cache.max_size_in_memory + 1


@pytest.fixture
def fine_grained_thread_switching() -> Iterator[None]:
    previous_interval: Final = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(previous_interval)


def _keys_until(deadline: float, rng: random.Random) -> Iterator[str]:
    while time.monotonic() < deadline:
        yield f"hot_{rng.randrange(5)}" if rng.random() < 0.7 else f"cold_{rng.randrange(40)}"


def _rewrite_keys_until(cache: InMemoryCache, deadline: float, seed: int) -> None:
    rng: Final = random.Random(seed)
    for key in _keys_until(deadline, rng):
        cache.delete_cache(key)
        cache.set_cache(key, "pin", ttl=rng.choice((300, 3600)))


@pytest.mark.usefixtures("fine_grained_thread_switching")
def test_in_memory_cache_survives_threads_rewriting_keys_in_a_full_cache() -> None:
    cache: Final = InMemoryCache(max_size_in_memory=5)
    deadline: Final = time.monotonic() + 0.5
    thread_count: Final = 8

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures: Final = tuple(
            executor.submit(_rewrite_keys_until, cache, deadline, seed) for seed in range(thread_count)
        )

    assert [future.exception() for future in futures] == [None] * thread_count
    keys: Final = (*(f"hot_{i}" for i in range(5)), *(f"cold_{i}" for i in range(40)))
    assert sum(cache.get_cache(key) is not None for key in keys) <= cache.max_size_in_memory


def test_in_memory_cache_stays_bounded_when_entries_have_no_usable_expiry() -> None:
    cache: Final = InMemoryCache(max_size_in_memory=10)
    for i in range(30):
        cache.set_cache(key=f"nan_{i}", value="response", ttl="NaN")
    cache.set_cache(key="fresh", value="response", ttl=60)

    keys: Final = (*(f"nan_{i}" for i in range(30)), "fresh")
    retained: Final = tuple(key for key in keys if cache.get_cache(key) is not None)
    assert len(retained) <= cache.max_size_in_memory
    assert cache.get_cache("fresh") == "response"


def test_in_memory_cache_prunes_expired_heap_entries_below_capacity():
    """
    Re-inserting expired keys below capacity should not grow expiration_heap
    without bound.
    """
    in_memory_cache = InMemoryCache(max_size_in_memory=200, default_ttl=1)

    for cycle in range(3):
        for i in range(5):
            in_memory_cache.set_cache(key=f"key_{i}", value=f"value_{cycle}_{i}", ttl=1)
        time.sleep(1.1)

    for i in range(5):
        in_memory_cache.set_cache(key=f"key_{i}", value=f"value_final_{i}", ttl=1)

    assert len(in_memory_cache.cache_dict) == 5
    assert len(in_memory_cache.ttl_dict) == 5
    assert len(in_memory_cache.expiration_heap) == 5


def test_in_memory_cache_injected_clock_controls_expiry_and_eviction() -> None:
    class Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    cache = InMemoryCache(max_size_in_memory=2, default_ttl=60, clock=clock)
    cache.set_cache("first", "original", ttl=10)
    clock.now = 9.0
    cache.set_cache("second", "survivor")
    assert cache.get_cache("first") == "original"
    clock.now = 10.001
    assert cache.get_cache("first") is None
    cache.set_cache("third", "replacement")
    assert cache.get_cache("second") == "survivor"
    clock.now = 69.001
    cache.set_cache("fourth", "new")
    assert cache.get_cache("second") is None
    assert cache.get_cache("third") == "replacement"
    assert cache.get_cache("fourth") == "new"


@pytest.mark.asyncio
async def test_async_increment_pipeline_keeps_each_key_for_its_own_ttl() -> None:
    now: Final = {"t": 1_000.0}
    cache: Final = InMemoryCache(default_ttl=600, clock=lambda: now["t"])

    await cache.async_increment_pipeline(
        increment_list=[
            {"key": "minute", "increment_value": 1, "ttl": 60},
            {"key": "day", "increment_value": 1, "ttl": 86400},
        ]
    )
    now["t"] += 601

    assert cache.get_cache("minute") is None
    assert cache.get_cache("day") == 1


class _ControlledClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _stats_tracking_cache(max_size_in_memory: int = 2) -> InMemoryCache:
    return InMemoryCache(
        max_size_in_memory=max_size_in_memory,
        default_ttl=60,
        clock=_ControlledClock(),
        track_eviction_stats=True,
    )


def test_eviction_stats_are_opt_in_and_disabled_cache_still_evicts() -> None:
    cache = InMemoryCache(max_size_in_memory=1)
    cache.set_cache("a", "va", ttl=100)
    cache.set_cache("b", "vb", ttl=100)

    assert cache.get_eviction_stats() is None
    cache.reset_eviction_stats()
    assert "a" not in cache.cache_dict
    assert cache.get_cache("b") == "vb"


def test_eviction_stats_expose_only_integer_counts_never_keys_or_values() -> None:
    cache = _stats_tracking_cache()
    cache.set_cache("secret-key", {"response": "secret-value"}, ttl=100)
    cache.delete_cache("secret-key")

    stats = cache.get_eviction_stats()
    assert stats is not None
    assert {field.name for field in fields(stats)} == {
        "capacity_evictions",
        "ttl_expirations",
        "explicit_deletions",
        "cleared_entries",
    }
    assert all(isinstance(getattr(stats, field.name), int) for field in fields(stats))
    assert "secret" not in json.dumps(asdict(stats))


def test_eviction_stats_count_each_reason_once_per_removed_entry() -> None:
    clock: Final = _ControlledClock()
    cache: Final = InMemoryCache(max_size_in_memory=2, default_ttl=60, clock=clock, track_eviction_stats=True)

    cache.set_cache("a", "va", ttl=100)
    cache.set_cache("b", "vb", ttl=200)
    cache.set_cache("c", "vc", ttl=300)
    assert cache.get_eviction_stats() == LocalCacheEvictionStats(capacity_evictions=1)

    clock.now = 201.0
    assert cache.get_cache("b") is None
    assert cache.get_cache("b") is None
    assert cache.get_eviction_stats() == LocalCacheEvictionStats(capacity_evictions=1, ttl_expirations=1)

    assert cache.get_cache("c") == "vc"
    cache.delete_cache("c")
    cache.delete_cache("c")
    assert cache.get_eviction_stats() == LocalCacheEvictionStats(
        capacity_evictions=1, ttl_expirations=1, explicit_deletions=1
    )

    cache.set_cache("d", "vd", ttl=400)
    cache.set_cache("e", "ve", ttl=500)
    cache.flush_cache()
    assert cache.get_eviction_stats() == LocalCacheEvictionStats(
        capacity_evictions=1,
        ttl_expirations=1,
        explicit_deletions=1,
        cleared_entries=2,
    )

    cache.flush_cache()
    assert cache.get_eviction_stats().cleared_entries == 2


def test_expired_entries_swept_during_set_count_ttl_once() -> None:
    clock: Final = _ControlledClock()
    cache: Final = InMemoryCache(max_size_in_memory=10, clock=clock, track_eviction_stats=True)
    for name in ("a", "b", "c"):
        cache.set_cache(name, "v", ttl=10)

    clock.now = 11.0
    cache.set_cache("d", "v", ttl=100)

    assert cache.get_eviction_stats() == LocalCacheEvictionStats(ttl_expirations=3)
    for name in ("a", "b", "c"):
        assert cache.get_cache(name) is None
    assert cache.get_eviction_stats().ttl_expirations == 3


def test_deleting_an_expired_entry_before_a_lazy_read_counts_as_explicit_delete() -> None:
    clock: Final = _ControlledClock()
    cache: Final = InMemoryCache(max_size_in_memory=10, clock=clock, track_eviction_stats=True)
    cache.set_cache("a", "v", ttl=10)
    clock.now = 11.0

    cache.delete_cache("a")

    assert cache.get_eviction_stats() == LocalCacheEvictionStats(explicit_deletions=1)


def test_capacity_evictions_without_usable_expiry_heap_entries_are_counted() -> None:
    cache: Final = InMemoryCache(max_size_in_memory=10, track_eviction_stats=True)
    for i in range(30):
        cache.set_cache(f"nan_{i}", "response", ttl="NaN")

    assert len(cache.cache_dict) == 10
    assert cache.get_eviction_stats() == LocalCacheEvictionStats(capacity_evictions=20)


def test_zero_capacity_caches_nothing_and_counts_no_evictions() -> None:
    cache: Final = InMemoryCache(max_size_in_memory=0, track_eviction_stats=True)
    for i in range(5):
        cache.set_cache(f"k_{i}", "v")

    assert cache.cache_dict == {}
    assert cache.get_eviction_stats() == LocalCacheEvictionStats()
    cache.flush_cache()
    assert cache.get_eviction_stats() == LocalCacheEvictionStats()


def test_reset_eviction_stats_zeros_counters_and_counting_resumes() -> None:
    cache: Final = _stats_tracking_cache()
    cache.set_cache("a", "va", ttl=100)
    cache.set_cache("b", "vb", ttl=200)
    cache.set_cache("c", "vc", ttl=300)
    assert cache.get_eviction_stats().capacity_evictions == 1

    cache.reset_eviction_stats()
    assert cache.get_eviction_stats() == LocalCacheEvictionStats()

    cache.delete_cache("b")
    assert cache.get_eviction_stats() == LocalCacheEvictionStats(explicit_deletions=1)


def test_expired_entry_removed_by_a_nested_reader_counts_ttl_once() -> None:
    cache: Final = _AnotherReaderExpiresTheEntryFirstCache(track_eviction_stats=True)
    cache.set_cache("tokenizer_failure", "HF unavailable", ttl=-1)

    assert cache.get_cache("tokenizer_failure") is None
    assert cache.other_reader_result is None
    assert cache.get_eviction_stats() == LocalCacheEvictionStats(ttl_expirations=1)


def test_tracking_stats_does_not_change_eviction_order_or_contents() -> None:
    clock_plain: Final = _ControlledClock()
    clock_tracked: Final = _ControlledClock()
    plain: Final = InMemoryCache(max_size_in_memory=3, default_ttl=60, clock=clock_plain)
    tracked: Final = InMemoryCache(max_size_in_memory=3, default_ttl=60, clock=clock_tracked, track_eviction_stats=True)

    def assert_same_state() -> None:
        assert sorted(tracked.cache_dict.items()) == sorted(plain.cache_dict.items())
        assert sorted(tracked.ttl_dict) == sorted(plain.ttl_dict)

    for i, ttl in enumerate((10, 10, 10, 100)):
        plain.set_cache(f"k_{i}", f"v_{i}", ttl=ttl)
        tracked.set_cache(f"k_{i}", f"v_{i}", ttl=ttl)
        assert_same_state()

    clock_plain.now = 11.0
    clock_tracked.now = 11.0
    plain.set_cache("k_4", "v_4", ttl=200)
    tracked.set_cache("k_4", "v_4", ttl=200)
    assert_same_state()

    for key in ("k_0", "k_1", "k_2", "k_3", "k_4"):
        assert plain.get_cache(key) == tracked.get_cache(key)
    assert_same_state()

    plain.delete_cache("k_4")
    tracked.delete_cache("k_4")
    assert_same_state()

    plain.flush_cache()
    tracked.flush_cache()
    assert_same_state()


@pytest.mark.usefixtures("fine_grained_thread_switching")
def test_eviction_stats_concurrent_reads_and_deletes_count_each_entry_once() -> None:
    clock: Final = _ControlledClock()
    cache: Final = InMemoryCache(max_size_in_memory=200, default_ttl=10, clock=clock, track_eviction_stats=True)
    for i in range(50):
        cache.set_cache(f"exp_{i}", "v", ttl=10)
    for i in range(20):
        cache.set_cache(f"del_{i}", "v", ttl=1_000)

    clock.now = 11.0
    reader_count: Final = 8
    deleter_count: Final = 4
    barrier: Final = threading.Barrier(reader_count + deleter_count)

    def read_expired(_: int) -> None:
        barrier.wait()
        for _ in range(2):
            for i in range(50):
                assert cache.get_cache(f"exp_{i}") is None

    def delete_repeatedly(_: int) -> None:
        barrier.wait()
        for _ in range(3):
            for i in range(20):
                cache.delete_cache(f"del_{i}")

    with ThreadPoolExecutor(max_workers=reader_count + deleter_count) as executor:
        read_results = executor.map(read_expired, range(reader_count))
        delete_results = executor.map(delete_repeatedly, range(deleter_count))
        read_futures: Final = tuple(read_results)
        delete_futures: Final = tuple(delete_results)

    assert read_futures == (None,) * reader_count
    assert delete_futures == (None,) * deleter_count
    assert cache.get_eviction_stats() == LocalCacheEvictionStats(ttl_expirations=50, explicit_deletions=20)
