from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class LLMRequest:
    """Canonical request envelope for all providers."""

    model: str
    messages: tuple[Mapping[str, str], ...]
    response_format: Mapping[str, Any] | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

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


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    request_hash: str
    latency_ms: float
    usage: LLMUsage | None = None
    raw: Mapping[str, Any] | None = None


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
