from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .types import LLMParseError

TModel = TypeVar("TModel", bound=BaseModel)


def strict_json_parse(content: str, schema: type[TModel]) -> TModel:
    """Parse JSON text into a pydantic schema with strict validation."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"Invalid JSON response: {exc}") from exc

    try:
        return schema.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise LLMParseError(f"Schema validation failed: {exc}") from exc


def response_format_for_schema(schema: type[BaseModel]) -> dict[str, object]:
    """Provider-ready json-schema response format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        },
    }
