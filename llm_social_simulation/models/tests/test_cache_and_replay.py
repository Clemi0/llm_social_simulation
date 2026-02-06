from pathlib import Path

from llm_social_simulation.models.mock_client import MockClient
from llm_social_simulation.models.toolkit import CachedClient, DiskCache, ReplayClient
from llm_social_simulation.models.types import LLMRequest


def test_cache_key_stability_across_instances(tmp_path: Path) -> None:
    req1 = LLMRequest(model="m", messages=({"role": "user", "content": "x"},))
    req2 = LLMRequest(model="m", messages=({"role": "user", "content": "x"},))
    assert req1.stable_hash() == req2.stable_hash()

    cache = DiskCache(tmp_path)
    wrapped = CachedClient(MockClient(fixed_latency_ms=0), cache)
    first = wrapped.generate(req1)
    second = wrapped.generate(req2)
    assert first.content == second.content


def test_replay_client_reads_cached_response(tmp_path: Path) -> None:
    req = LLMRequest(model="m", messages=({"role": "user", "content": "x"},))
    cache = DiskCache(tmp_path)
    cached = CachedClient(MockClient(fixed_latency_ms=0), cache)
    generated = cached.generate(req)

    replay = ReplayClient(cache)
    replayed = replay.generate(req)
    assert generated.content == replayed.content
