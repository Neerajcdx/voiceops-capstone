"""Provider-agnostic LLM client with a built-in tool-calling loop.

Two adapters cover almost every hosted or local model:
  * anthropic  -> Claude models (LLM_PROVIDER=anthropic)
  * openai     -> any OpenAI-compatible Chat Completions endpoint: OpenAI, Gemini, Groq, OpenRouter, Ollama, vLLM...

Both expose the same call:  llm.complete(system, messages, tools=..., tool_handler=...) -> LLMResult
`messages` is a plain list of {"role": "user"|"assistant", "content": str}; tool-call blocks never leak out of
the adapter, so conversation history stays portable between providers.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import settings


def _timeout() -> float:
    """Seconds to wait for the LLM. Raise it in .env (LLM_TIMEOUT=180) if your model is slow."""
    import os
    try:
        return float(os.getenv("LLM_TIMEOUT", "60"))
    except ValueError:
        return 60.0


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema


@dataclass
class LLMResult:
    text: str
    tool_trace: list[dict] = field(default_factory=list)   # [{"name", "args", "result"}]
    usage: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


def _retry(fn: Callable[[], Any], attempts: int = 5):
    """Exponential backoff for rate limits / transient errors (free tiers are strict)."""
    delay = 1.5
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            transient = status in (408, 409, 429, 500, 502, 503, 504, 529) or "timeout" in str(exc).lower() \
                or "overloaded" in str(exc).lower() or "rate" in type(exc).__name__.lower()
            if not transient or i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


class BaseLLM:
    def complete(self, system: str, messages: list[dict], tools: list[Tool] | None = None,
                 tool_handler: Callable[[str, dict], Any] | None = None, max_steps: int = 6,
                 max_tokens: int = 700, temperature: float = 0.0) -> LLMResult:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------------------- Anthropic
class AnthropicLLM(BaseLLM):
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        import anthropic

        kw = {"api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        kw["timeout"] = _timeout()
        self.client = anthropic.Anthropic(**kw)
        self.model = model

    def complete(self, system, messages, tools=None, tool_handler=None, max_steps=6, max_tokens=700, temperature=0.0):
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        a_tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in (tools or [])]
        trace: list[dict] = []
        usage = {"input": 0, "output": 0}
        for _ in range(max_steps):
            kw = dict(model=self.model, max_tokens=max_tokens, system=system, messages=msgs, temperature=temperature)
            if a_tools:
                kw["tools"] = a_tools
            resp = _retry(lambda: self.client.messages.create(**kw))
            usage["input"] += resp.usage.input_tokens
            usage["output"] += resp.usage.output_tokens
            calls = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not calls:
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                return LLMResult(text, trace, usage)
            msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]})
            results = []
            for c in calls:
                out = tool_handler(c.name, dict(c.input)) if tool_handler else {"error": "no tool handler"}
                trace.append({"name": c.name, "args": dict(c.input), "result": out})
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(out, ensure_ascii=False, default=str)})
            msgs.append({"role": "user", "content": results})
        return LLMResult("", trace, usage)


def _echo_tool_call(c) -> dict:
    """Rebuild a tool call for the next request. Gemini 3 returns a `thought_signature` inside
    `extra_content` and rejects the follow-up request (HTTP 400) unless it is sent back unchanged."""
    d = {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
    extra = getattr(c, "extra_content", None)
    if extra is None:
        extra = (getattr(c, "model_extra", None) or {}).get("extra_content")
    if extra:
        d["extra_content"] = extra
    return d


# --------------------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatLLM(BaseLLM):
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or "none", base_url=base_url, timeout=_timeout(), max_retries=0)
        self.model = model

    def complete(self, system, messages, tools=None, tool_handler=None, max_steps=6, max_tokens=700, temperature=0.0):
        msgs: list[dict] = [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in messages]
        o_tools = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                   for t in (tools or [])]
        trace: list[dict] = []
        usage = {"input": 0, "output": 0}
        for _ in range(max_steps):
            kw = dict(model=self.model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
            if o_tools:
                kw["tools"] = o_tools
            resp = _retry(lambda: self.client.chat.completions.create(**kw))
            if getattr(resp, "usage", None):
                usage["input"] += resp.usage.prompt_tokens or 0
                usage["output"] += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return LLMResult((msg.content or "").strip(), trace, usage)
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [_echo_tool_call(c) for c in msg.tool_calls]})
            for c in msg.tool_calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = tool_handler(c.function.name, args) if tool_handler else {"error": "no tool handler"}
                trace.append({"name": c.function.name, "args": args, "result": out})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out, ensure_ascii=False, default=str)})
        return LLMResult("", trace, usage)


def get_llm() -> BaseLLM:
    prov, model, key, base = settings.llm_provider, settings.llm_model, settings.llm_api_key, settings.llm_base_url
    if prov == "anthropic":
        if not key:
            raise LLMError("LLM_API_KEY is not set (LLM_PROVIDER=anthropic).")
        return AnthropicLLM(model, key, base)
    if prov in ("openai", "openai_compat", "gemini", "groq", "ollama", "openrouter"):
        if not key and "localhost" not in (base or "") and "127.0.0.1" not in (base or ""):
            raise LLMError("LLM_API_KEY is not set. See .env.example for provider presets.")
        return OpenAICompatLLM(model, key, base)
    raise LLMError(f"Unknown LLM_PROVIDER '{prov}'. Use 'anthropic' or 'openai'.")
