from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping

# this is a contract layer, it defines how we talk to models


@dataclass(frozen=True)
# once a request is generated, it can't be changed

class LLMRequest:
    """Canonical request envelope for all providers."""

    model: str  # model's name
    messages: tuple[Mapping[str, str], ...]
    response_format: Mapping[str, Any] | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # putting metadata in stable_hash may cause some bugs here

    # If two requests are exactly the same, they will produce the same hash.
    def stable_hash(self) -> str:
        """Stable hash used for caching/replay keys."""
        payload = {
            "model": self.model,
            "messages": [dict(m) for m in self.messages],
            "response_format": self.response_format,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "metadata": dict(self.metadata),
        }
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256(canon.encode("utf-8")).hexdigest()


# types.py


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "LLMUsage | None":
        if d is None:
            return None
        return LLMUsage(
            prompt_tokens=d.get("prompt_tokens"),
            completion_tokens=d.get("completion_tokens"),
            total_tokens=d.get("total_tokens"),
        )


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    request_hash: str
    latency_ms: float
    usage: LLMUsage | None = None
    raw: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "request_hash": self.request_hash,
            "latency_ms": self.latency_ms,
            "usage": self.usage.to_dict() if self.usage else None,
            "raw": dict(self.raw) if self.raw else None,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LLMResponse":
        return LLMResponse(
            content=d["content"],
            model=d["model"],
            request_hash=d["request_hash"],
            latency_ms=d["latency_ms"],
            usage=LLMUsage.from_dict(d.get("usage")),
            raw=d.get("raw"),
        )


# classify possible errors


class LLMClientError(Exception):
    """Base typed error for client-level failures."""


class LLMTimeoutError(LLMClientError):
    """Request timeout."""


class LLMRetryableError(LLMClientError):
    """Transient/retryable provider failure."""


class LLMRateLimitError(LLMRetryableError):
    """Rate limit exceeded (local or provider)."""


class LLMParseError(LLMClientError):
    """Strict JSON/schema parsing failure."""


class LLMProviderError(LLMClientError):
    """Non-retryable provider error."""
