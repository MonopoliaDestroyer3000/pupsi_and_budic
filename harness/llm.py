"""OpenAI-compatible proxy transport. Credentials never enter logs or prompts."""
from __future__ import annotations

import json
import html
import re
import urllib.error
import urllib.request

from .budget import Budget

ALLOWED_MODELS = ("qwen3.8-27b-noreason", "qwen3.6-35b-a3b-noreason",
                  "qwen3.8-27b", "qwen3.6-35b-a3b")


class ProxyError(RuntimeError):
    def __init__(self, message: str, cost: float = 0, detail: str = ""):
        super().__init__(message)
        self.cost = cost
        self.detail = detail


def error_detail(error, key: str) -> str:
    """Expose bounded error text, stripping markup and credentials."""
    raw = b""
    try:
        raw = error.read(16384)
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return "Server returned an unrecognized error format."
        value = payload.get("error", payload.get("detail", ""))
        if isinstance(value, dict):
            value = value.get("message", value.get("code", ""))
        text = str(value) if value else "No error description returned."
    except Exception:
        text = raw.decode('utf-8', errors='replace')
        text = re.sub(r'(?is)<(script|style)\b[^>]*>.*?</\1>', ' ', text)
        text = html.unescape(re.sub(r'<[^>]+>', ' ', text))
        text = 'Non-JSON response: ' + (text.strip() or '[empty body]')
    if key:
        text = text.replace(key, "[REDACTED]")
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?:sk-|NGU26-)[A-Za-z0-9_.-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)(Key Hash\s*\(Token\)\s*=\s*)[a-f0-9]+", r"\1[REDACTED]", text)
    return " ".join(text.split())[:1000]


class Proxy:
    def __init__(self, base_url: str, api_key: str, budget: Budget, timeout: float = 80):
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            self.url = base_url
        else:
            self.url = base_url + "/chat/completions"
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("LLM_BASE_URL must be an HTTP(S) proxy URL, normally ending in /v1")
        self.key = api_key
        self.budget = budget
        self.timeout = timeout

    def complete(self, messages: list, model: str, max_tokens: int, task_remaining: float) -> tuple[str, float]:
        if model not in ALLOWED_MODELS:
            raise ValueError("Model is not permitted by the competition")
        serialized = json.dumps(messages, ensure_ascii=False)
        # Conservative byte heuristic; actual uncached input is settled from usage.
        estimate = len(serialized.encode("utf-8")) + 100
        reservation = self.budget.reserve(estimate, max_tokens, task_remaining)
        body = json.dumps({"model": model, "messages": messages,
                           "max_tokens": reservation.output_limit, "temperature": 0.1}).encode()
        request = urllib.request.Request(self.url, data=body, headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        try:
            # Explicitly ignore ambient HTTP proxies: only the organizer endpoint is used.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=self.timeout) as response:
                raw = response.read(2_000_000)
            result = json.loads(raw)
        except urllib.error.HTTPError as exc:
            cost = self.budget.settle(reservation, None)
            raise ProxyError(f"Proxy HTTP {exc.code}", cost, error_detail(exc, self.key)) from None
        except Exception as exc:
            cost = self.budget.settle(reservation, None)
            raise ProxyError(f"Proxy request failed ({type(exc).__name__})", cost) from None
        cost = self.budget.settle(reservation, result.get("usage") if isinstance(result, dict) else None)
        try:
            content = result["choices"][0]["message"].get("content") or ""
            if isinstance(content, list):
                content = "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
            return str(content), cost
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProxyError("Proxy response has no assistant completion", cost) from None
