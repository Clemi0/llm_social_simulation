from __future__ import annotations

from abc import ABC, abstractmethod

from .types import LLMRequest, LLMResponse


class LLMClient(ABC):
    """Stable protocol for model providers and wrappers."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a model response for a canonical request."""
