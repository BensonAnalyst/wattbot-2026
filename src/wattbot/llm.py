"""Chat model client for a Databricks serving endpoint (OpenAI-style chat completions over HTTPS).

Credentials come from environment variables or Kaggle Secrets, never from the code:
    DATABRICKS_HOST   e.g. https://dbc-xxxx.cloud.databricks.com
    DATABRICKS_TOKEN  a personal access token
"""

import os
import re
import time


def chat_text(response: dict) -> str:
    """Reasoning models (gpt-oss) return content as a list of blocks; keep only the text blocks."""
    content = response["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        content = "".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")
    # Some reasoning models (Qwen3.5) put their reasoning inline as <think>...</think>; keep only the answer.
    return re.sub(r"<think>.*?(</think>|$)", "", content, flags=re.S).strip()


def endpoint(host: str, model: str):
    """(url, extra body fields) for a model.

    Unity Gateway models are named system.ai.<name> and share one OpenAI-style URL with the model in the body;
    older foundation models (databricks-gpt-oss-120b) keep a per-endpoint URL."""
    if model.startswith("system.ai."):
        return f"{host}/ai-gateway/mlflow/v1/chat/completions", {"model": model}
    return f"{host}/serving-endpoints/{model}/invocations", {}


def make_databricks_llm(model: str = "databricks-gpt-oss-120b", host: str = None, token: str = None,
                        max_tokens: int = 8000, retries: int = 8):
    import requests

    host = (host or os.environ["DATABRICKS_HOST"]).rstrip("/")
    token = token or os.environ["DATABRICKS_TOKEN"]
    url, extra = endpoint(host, model)

    def llm(messages, temperature: float = 0.0):
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, **extra}
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=300)
            except requests.RequestException:
                resp = None
            if resp is not None and resp.ok:
                return chat_text(resp.json())
            code = resp.status_code if resp is not None else 0
            if code and code < 500 and code != 429:  # a request error (e.g. deprecated model): retrying won't help
                raise RuntimeError(f"HTTP {code}: {resp.text[:300]}")
            if attempt == retries - 1:
                raise RuntimeError(f"HTTP {code}: gave up after {retries} attempts")
            # Rate limited (429) or a server error: wait, longer each time, or as long as the server asks.
            wait = 15 * 2 ** min(attempt, 4)
            if resp is not None and resp.headers.get("Retry-After", "").isdigit():
                wait = max(wait, int(resp.headers["Retry-After"]))
            time.sleep(wait)

    return llm
