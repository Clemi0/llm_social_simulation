from llm_social_simulation.models.mock_client import MockClient
from llm_social_simulation.models.types import LLMRequest


def _request() -> LLMRequest:
    return LLMRequest(
        model="unit-model",
        messages=(
            {"role": "system", "content": "You are strict."},
            {"role": "user", "content": "Return JSON."},
        ),
        temperature=0.0,
    )


def test_request_hash_is_stable() -> None:
    req1 = _request()
    req2 = _request()
    assert req1.stable_hash() == req2.stable_hash()


def test_mock_client_is_deterministic() -> None:
    client = MockClient(fixed_latency_ms=0.0)
    req = _request()
    assert client.generate(req).content == client.generate(req).content
