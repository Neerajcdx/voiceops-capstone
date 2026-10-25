"""Scripted LLM for plumbing tests (no network)."""
import json, re
from voiceops.llm import BaseLLM, LLMResult
from voiceops.prompts import REWRITE_SYSTEM


class FakeLLM(BaseLLM):
    def __init__(self, replies=None):
        self.calls = []          # (system, messages)
        self.replies = list(replies or [])

    def complete(self, system, messages, tools=None, tool_handler=None, max_steps=6, max_tokens=700, temperature=0.0):
        self.calls.append((system, messages))
        if system.startswith(REWRITE_SYSTEM[:40]):
            return LLMResult(json.dumps({"query": messages[-1]["content"].split("Customer message:")[-1].strip(), "language": "en"}))
        user = messages[-1]["content"].lower()
        trace = []
        if self.replies:
            text = self.replies.pop(0)
        elif re.search(r"human|insaan|इंसान|approve", user) and tool_handler:
            out = tool_handler("create_ticket", {"summary": "Customer asked for a human; verified order discussed; escalating per ESC-1 (customer request). last4 4533", "reason": "human_requested", "intent": "human_requested"})
            trace.append({"name": "create_ticket", "args": {}, "result": out})
            text = "A colleague will contact you. Ticket " + out.get("ticket_id", "?") if out.get("created") else "sorry"
        else:
            text = "OK."
        return LLMResult(text, trace, {"input": 1, "output": 1})
