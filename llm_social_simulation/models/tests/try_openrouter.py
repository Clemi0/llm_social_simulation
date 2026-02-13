from llm_social_simulation.models.openrouter_client import OpenRouterClient
from llm_social_simulation.models.types import LLMRequest


def main():
    client = OpenRouterClient(timeout_s=60)

    req = LLMRequest(
        model="openai/gpt-4o-mini",
        messages=({"role": "user", "content": "Return exactly: OK"},),
        temperature=0,
        max_tokens=10,
    )

    resp = client.generate(req)

    print("=== content ===")
    print(resp.content)
    print("=== meta ===")
    print("model:", resp.model)
    print("latency_ms:", resp.latency_ms)
    print("usage:", resp.usage)


if __name__ == "__main__":
    main()
