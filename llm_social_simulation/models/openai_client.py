from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import error
from urllib import request as urlrequest

from .client import LLMClient
from .types import (
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMRetryableError,
    LLMTimeoutError,
    LLMUsage,
)


class OpenAIClient(LLMClient):
    """Minimal real OpenAI client using the Responses API over HTTPS."""

    def __init__(self, api_key: str | None = None, timeout_s: float = 30.0):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMProviderError("OPENAI_API_KEY is required for OpenAIClient")
        self.timeout_s = timeout_s
        self.url = "https://api.openai.com/v1/responses"

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": request.model,
            "input": [dict(m) for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        if request.response_format is not None:
            payload["text"] = {"format": request.response_format}

        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except TimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except error.HTTPError as exc:
            status = exc.code
            data = exc.read().decode("utf-8", errors="ignore")
            if status in (408, 409, 429, 500, 502, 503, 504):
                if status == 429:
                    raise LLMRateLimitError(data) from exc
                raise LLMRetryableError(data) from exc
            raise LLMProviderError(f"OpenAI HTTP {status}: {data}") from exc
        except error.URLError as exc:
            raise LLMRetryableError(str(exc)) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        usage_obj = raw.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=usage_obj.get("input_tokens"),
            completion_tokens=usage_obj.get("output_tokens"),
            total_tokens=usage_obj.get("total_tokens"),
        )

        text = self._extract_text(raw)
        return LLMResponse(
            content=text,
            model=str(raw.get("model", request.model)),
            request_hash=request.stable_hash(),
            latency_ms=latency_ms,
            usage=usage,
            raw={"id": raw.get("id")},
        )

    @staticmethod
    def _extract_text(raw: dict[str, Any]) -> str:
        output = raw.get("output") or []
        fragments: list[str] = []
        for item in output:
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    fragments.append(c.get("text", ""))
        if fragments:
            return "".join(fragments)
        fallback = raw.get("output_text")
        if isinstance(fallback, str):
            return fallback
        raise LLMProviderError("OpenAI response did not contain text output")
