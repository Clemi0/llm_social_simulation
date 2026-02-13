from __future__ import annotations

from abc import ABC, abstractmethod

from .types import LLMRequest, LLMResponse


class LLMClient(ABC):
    """Stable protocol for model providers and wrappers."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a model response for a canonical request."""


# agents calls their clients (openai, gemini....) to generate response
# class OpaiAgent:
#     def __init__(self, client: LLMClient):
#         self.client = oepnai_client.py

#     def act(self, state):
#         request = build_request(state)
#         response = self.client.generate(request)
#         return parse_decision(response)
