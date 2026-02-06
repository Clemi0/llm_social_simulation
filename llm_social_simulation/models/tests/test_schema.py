from pydantic import BaseModel
import pytest

from llm_social_simulation.models.schema import strict_json_parse
from llm_social_simulation.models.types import LLMParseError


class Decision(BaseModel):
    action: str
    confidence: float


def test_strict_json_parse_success() -> None:
    parsed = strict_json_parse('{"action":"C","confidence":0.5}', Decision)
    assert parsed.action == "C"


def test_strict_json_parse_rejects_non_json() -> None:
    with pytest.raises(LLMParseError):
        strict_json_parse("action=C", Decision)


def test_strict_json_parse_rejects_schema_mismatch() -> None:
    with pytest.raises(LLMParseError):
        strict_json_parse('{"action":"C","confidence":"0.5"}', Decision)
