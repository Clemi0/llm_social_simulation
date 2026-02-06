import os

import pytest

from llm_social_simulation.models.openai_client import OpenAIClient
from llm_social_simulation.models.types import LLMRequest


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY configured")
def test_openai_provider_contract_live() -> None:
    client = OpenAIClient(timeout_s=30)
    req = LLMRequest(
        model="gpt-4o-mini",
        messages=(
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": '{"ok": true}'},
        ),
        temperature=0.0,
    )
    resp = client.generate(req)
    assert isinstance(resp.content, str)
    assert resp.request_hash == req.stable_hash()
