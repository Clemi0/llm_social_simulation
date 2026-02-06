from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import threading
import time
from typing import Any

from .client import LLMClient
from .types import (
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMRetryableError,
)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_s: float = 0.25
    max_backoff_s: float = 2.0


class TokenBucketRateLimiter:
    """Simple thread-safe local rate limiter."""

    def __init__(self, rate_per_second: float, burst: int):
        self.rate = rate_per_second
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(wait)


class DiskCache:
    """JSON file cache keyed by request hash."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(**payload)

    def put(self, value: LLMResponse) -> None:
        path = self._path(value.request_hash)
        path.write_text(json.dumps(value.__dict__, sort_keys=True), encoding="utf-8")


class CachedClient(LLMClient):
    def __init__(self, inner: LLMClient, cache: DiskCache):
        self.inner = inner
        self.cache = cache

    def generate(self, request: LLMRequest) -> LLMResponse:
        key = request.stable_hash()
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        out = self.inner.generate(request)
        self.cache.put(out)
        return out


class ReplayClient(LLMClient):
    """Reads deterministic responses from disk cache only."""

    def __init__(self, cache: DiskCache):
        self.cache = cache

    def generate(self, request: LLMRequest) -> LLMResponse:
        key = request.stable_hash()
        hit = self.cache.get(key)
        if hit is None:
            raise KeyError(f"Replay miss for {key}")
        return hit


class ResilientClient(LLMClient):
    """Adds local rate limiting and retry w/ exponential backoff."""

    def __init__(
        self,
        inner: LLMClient,
        retry_policy: RetryPolicy | None = None,
        limiter: TokenBucketRateLimiter | None = None,
    ):
        self.inner = inner
        self.retry_policy = retry_policy or RetryPolicy()
        self.limiter = limiter

    def generate(self, request: LLMRequest) -> LLMResponse:
        backoff = self.retry_policy.initial_backoff_s
        attempts = 0
        while True:
            attempts += 1
            if self.limiter:
                self.limiter.acquire()
            try:
                return self.inner.generate(request)
            except (LLMRetryableError, LLMRateLimitError):
                if attempts >= self.retry_policy.max_attempts:
                    raise
                time.sleep(min(backoff, self.retry_policy.max_backoff_s))
                backoff *= 2


class JsonLoggerClient(LLMClient):
    """Structured logs with prompt redaction by default."""

    def __init__(self, inner: LLMClient, *, log_prompts: bool = False):
        self.inner = inner
        self.log_prompts = log_prompts

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        resp = self.inner.generate(request)
        event: dict[str, Any] = {
            "event": "llm.generate",
            "request_hash": request.stable_hash(),
            "model": request.model,
            "latency_ms": resp.latency_ms,
            "wall_clock_ms": (time.perf_counter() - started) * 1000,
            "usage": resp.usage.__dict__ if resp.usage else None,
            "messages": "[REDACTED]" if not self.log_prompts else [dict(m) for m in request.messages],
        }
        print(json.dumps(event, sort_keys=True))
        return resp
