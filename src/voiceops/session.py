"""Per-conversation state and the code-enforced identity-verification flow (AGT-IDV).

Why in code and not only in the prompt: the brief says order details must not be revealed before the caller proves
identity, allows exactly one retry, and forbids repeating verification secrets. The LLM therefore never receives
order data until `lookup_order(order_id, verification_value)` has returned verified=True.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from .config import ensure_dataset_on_path
from .textnorm import extract_order_ids, extract_verification_candidates

MAX_ATTEMPTS_PER_ORDER = 2      # first try + ONE retry (IDV-3)
MAX_FAILURES_PER_SESSION = 3    # stop order-ID guessing across several orders

# fields that must never reach the LLM (or logs / ticket summaries)
_HIDDEN_FIELDS = {"delivery_pincode", "customer_id", "found", "verified", "verification_required"}


def sanitize_order(order: dict) -> dict:
    """Drop empty fields and verification secrets; keep everything the agent may legitimately use."""
    out = {}
    for k, v in order.items():
        if k in _HIDDEN_FIELDS or v is None:
            continue
        if k == "items":
            out[k] = [{kk: vv for kk, vv in it.items() if kk not in ("order_id",) and vv is not None} for it in v]
        else:
            out[k] = v
    return out


@dataclass
class Session:
    session_id: str = "default"
    channel: str = "text"                       # "text" | "voice"
    history: list[dict] = field(default_factory=list)   # [{"role","content"}]
    verified: dict[str, dict] = field(default_factory=dict)  # order_id -> sanitized order
    attempts: dict[str, int] = field(default_factory=dict)
    pending_oid: str | None = None
    active_oid: str | None = None
    total_failures: int = 0
    secrets: set[str] = field(default_factory=set)     # values the customer supplied as proof (never repeat)
    tickets: list[dict] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)  # every tool call across the conversation
    turn: int = 0

    # -------------------------------------------------------------- verification state
    def locked(self, oid: str | None = None) -> bool:
        if self.total_failures >= MAX_FAILURES_PER_SESSION:
            return True
        oid = oid or self.pending_oid
        return bool(oid) and oid not in self.verified and self.attempts.get(oid, 0) >= MAX_ATTEMPTS_PER_ORDER

    def state_summary(self) -> str:
        if self.active_oid and self.active_oid in self.verified:
            return f"VERIFIED for order {self.active_oid}. Order record is below; you may discuss it."
        if self.pending_oid:
            if self.locked():
                return (f"LOCKED: verification for order {self.pending_oid} failed {MAX_ATTEMPTS_PER_ORDER} times. "
                        "Share NO order details. Offer general policy help and suggest contacting support from the registered phone number.")
            n = self.attempts.get(self.pending_oid, 0)
            if n == 0:
                return (f"ORDER ID RECEIVED ({self.pending_oid}) but NOT VERIFIED. Ask for ONE of: last 4 digits of the registered phone "
                        "number OR the delivery pincode. Share nothing about the order yet.")
            return (f"NOT VERIFIED: the value given for order {self.pending_oid} did not match (attempt {n} of {MAX_ATTEMPTS_PER_ORDER}). "
                    "Tell the customer it did not match and invite ONE retry (last 4 phone digits or delivery pincode). Do not reveal or hint at the correct value.")
        return "NO ORDER IN CONTEXT. For order-specific help ask for the order ID first (one question). General policy questions need no verification."


def run_verification(session: Session, user_text: str, lookup_fn=None) -> list[str]:
    """Update the session from what the customer just said; returns tool-event notes for the LLM.

    Executes real `lookup_order` calls (recorded in session.tool_log, matching the practice expected_tool_calls):
      * order ID seen            -> lookup_order(order_id)                      (verification prompt only)
      * 4-digit / 6-digit value  -> lookup_order(order_id, verification_value)  (only while an order is pending)
    """
    if lookup_fn is None:
        ensure_dataset_on_path()
        from tools import lookup_order as lookup_fn  # type: ignore

    notes: list[str] = []
    ids = extract_order_ids(user_text)
    for oid in ids:
        if oid in session.verified:
            session.active_oid = oid
            notes.append(f"Customer mentioned order {oid}: already verified in this conversation.")
            continue
        if session.pending_oid != oid:
            session.pending_oid = oid
            session.attempts.setdefault(oid, 0)
            res = lookup_fn(oid)
            session.tool_log.append({"name": "lookup_order", "args": {"order_id": oid}, "result": {"verification_required": bool(res.get("verification_required"))}})
            notes.append(f"lookup_order({oid}) -> verification_required (no details disclosed).")

    pending = session.pending_oid
    if pending and pending not in session.verified:
        cands = extract_verification_candidates(user_text, exclude=ids)
        if cands and session.locked(pending):
            notes.append(f"Customer sent another verification value but order {pending} is LOCKED; no further attempts allowed.")
        elif cands:
            value = cands[0]  # one value at a time (IDV-2); extra numbers are ignored
            res = lookup_fn(pending, value)
            ok = bool(res.get("found") and res.get("verified"))
            session.tool_log.append({"name": "lookup_order", "args": {"order_id": pending, "verification_value": value},
                                     "result": {"verified": ok}})
            session.secrets.add(value)
            if ok:
                session.verified[pending] = sanitize_order(res)
                session.active_oid = pending
                notes.append(f"lookup_order({pending}, verification_value=<given>) -> VERIFIED. Order record now available.")
            else:
                session.attempts[pending] = session.attempts.get(pending, 0) + 1
                session.total_failures += 1
                left = MAX_ATTEMPTS_PER_ORDER - session.attempts[pending]
                notes.append(f"lookup_order({pending}, verification_value=<given>) -> NOT VERIFIED ({'one retry left' if left > 0 else 'no retries left: LOCKED'}).")
    return notes


def scrub_secrets(text: str, session: Session) -> str:
    """Remove any verification value the customer supplied from outgoing text (tickets, logs)."""
    for s in session.secrets:
        text = re.sub(rf"(?<!\d){re.escape(s)}(?!\d)", "[redacted]", text)
    return text


def order_context(session: Session) -> str:
    """Compact JSON of verified orders for the prompt."""
    import json

    if not session.verified:
        return "none"
    blocks = []
    for oid, o in session.verified.items():
        tag = " (ACTIVE)" if oid == session.active_oid else ""
        blocks.append(f"order {oid}{tag}: " + json.dumps(copy.deepcopy(o), ensure_ascii=False, separators=(",", ":")))
    return "\n".join(blocks)
