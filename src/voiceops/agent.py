"""The ZentroMart support agent.

SupportAgent  = Step 2 + Step 3: retrieval-grounded answers, memory, tools (verification, tickets, date maths).
PromptOnlyAgent = Step 1 baseline: the same persona prompt, no documents, no tools (used only for the evaluation).

One turn:
  1. normalise ASR digits -> harness runs identity verification (real lookup_order calls, in code)
  2. (optional) LLM rewrites the message into a standalone English search query (Hindi / follow-ups)
  3. hybrid retrieval -> top chunks (+ companion sections) injected as POLICY CONTEXT
  4. LLM answers with tools: search_kb, lookup_product, check_pincode, date_calc, create_ticket
  5. guardrails: no fake actions / fake tickets, no leaked verification values
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date

from . import dates as dates_mod
from .config import TODAY, ensure_dataset_on_path, settings
from .kb import check_pincode, find_products, products, sku_prefixes
from .llm import BaseLLM, LLMError, Tool, get_llm
from .prompts import BASELINE_SYSTEM, CONTEXT_TEMPLATE, REWRITE_SYSTEM, STYLE_TEXT, STYLE_VOICE, SYSTEM_TEMPLATE
from .retriever import Retriever
from .session import Session, order_context, run_verification, scrub_secrets
from .textnorm import extract_order_ids, extract_skus, extract_verification_candidates

TOOLS = [
    Tool("search_kb", "Search the ZentroMart policy knowledge base. Use an ENGLISH query in policy vocabulary. Returns relevant policy sections with their [SECTION-ID].",
         {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
    Tool("lookup_product", "Look up catalog products by SKU (e.g. ELE-0012) or by product name. Returns return_category, price, seller, assured, warranty_months.",
         {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
    Tool("check_pincode", "Check whether ZentroMart delivers to a 6-digit pincode and whether COD / Express are available there.",
         {"type": "object", "properties": {"pincode": {"type": "string"}}, "required": ["pincode"]}),
    Tool("date_calc", "Calendar maths. ALWAYS use this instead of computing dates yourself. ops: add_calendar_days(date1, days), add_business_days(date1, days), "
         "business_days_between(date1, date2), calendar_days_between(date1, date2), weekday(date1), is_business_day(date1). Dates are YYYY-MM-DD or 'today'. "
         "Business days = Mon-Sat excluding holidays; date1 itself is day 0.",
         {"type": "object", "properties": {"op": {"type": "string", "enum": ["add_calendar_days", "add_business_days", "business_days_between", "calendar_days_between", "weekday", "is_business_day"]},
                                           "date1": {"type": "string"}, "date2": {"type": "string"}, "days": {"type": "integer"}}, "required": ["op", "date1"]}),
    Tool("create_ticket", "Create a human-handover ticket. Call this BEFORE telling the customer a colleague will contact them. Only claim a handover if the result has created=true.",
         {"type": "object", "properties": {"summary": {"type": "string", "description": "Issue, detected intent, order ID, what the customer told you, what you checked, reason for escalation. Min 20 chars. No verification values."},
                                           "reason": {"type": "string", "description": "short code e.g. high_value_damage, human_requested, repeat_contact, safety_risk, fraud_report, legal_threat, refund_overdue, customer_insists"},
                                           "intent": {"type": "string"}}, "required": ["summary", "reason"]}),
]

# --- guardrail patterns (first person claims of actions that no tool of ours can perform / unproven handover)
_FAKE_ACTION = re.compile(
    r"\bI(?:'ve| have)\s+(?:just\s+)?(?:cancell?ed|approved|refunded|credited|processed|issued|waived|replaced|arranged)\b|"
    r"\b(?:kar diya|kar di hai|कर दिया|कर दी है)\b.*\b(?:cancel|approve|refund|credit)|\b(?:cancel|approve|refund|credit)\w*\s+kar\s+di(?:ya|ye)\b",
    re.I)
_FAKE_TICKET = re.compile(
    r"\b(?:ticket\s+(?:has\s+been|is|was)\s+(?:created|raised|logged|opened)|I(?:'ve| have)\s+(?:created|raised|logged|opened)\s+a\s+ticket|ticket\s+(?:id|number))\b|"
    r"टिकट\s+(?:बना|बनाया|बन गया|दर्ज)|ticket\s+(?:bana|raise|create)\s+(?:kar\s+)?di(?:ya|ye)",
    re.I)


@dataclass
class AgentResult:
    reply: str
    sources: list[str] = field(default_factory=list)          # chunk IDs the answer was grounded on
    chunks: list[dict] = field(default_factory=list)          # [{id, title, score}]
    tool_calls: list[dict] = field(default_factory=list)      # every tool call made this turn
    ticket_id: str | None = None
    events: list[str] = field(default_factory=list)
    rewritten_query: str = ""
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    guard: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def clean_reply(text: str) -> str:
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


class SupportAgent:
    def __init__(self, llm: BaseLLM | None = None, retriever: Retriever | None = None):
        ensure_dataset_on_path()
        self.llm = llm or get_llm()
        self.retriever = retriever or Retriever()
        pin_ids = [c for c in self.retriever.chunks if c.doc_type == "agent_guide" or c.id in ("KB-00.1", "KB-00.2", "KB-00.3")]
        self._pinned = "\n\n".join(c.render() for c in pin_ids)
        self._prefixes = sku_prefixes()
        self._known_skus = set(products())

    # ------------------------------------------------------------------ public API
    def new_session(self, session_id: str = "default", channel: str = "text") -> Session:
        return Session(session_id=session_id, channel=channel)

    def respond(self, session: Session, text: str, channel: str | None = None,
                alternatives: list[str] | None = None) -> AgentResult:
        """`alternatives`: other-language speech-recognition hypotheses of the same utterance (voice only)."""
        t0 = time.time()
        channel = channel or session.channel
        session.channel = channel
        session.turn += 1
        first = not session.history
        res = AgentResult(reply="")
        text = (text or "").strip()

        # 1. verification harness (real tool calls, in code)
        log0 = len(session.tool_log)
        alternatives = [a for a in (alternatives or []) if a and a.strip()]
        harness_text = text
        if alternatives and not (extract_order_ids(text) or extract_verification_candidates(text)):
            harness_text = next((a for a in alternatives if extract_order_ids(a) or extract_verification_candidates(a)), text)
        events = run_verification(session, harness_text)
        harness_calls = list(session.tool_log[log0:])
        res.events = events
        ids_found = extract_order_ids(text)
        skus = extract_skus(text, self._prefixes, self._known_skus)
        note = ""
        if skus:
            note += f"\n[detected SKU(s): {', '.join(skus)}]"
        if alternatives:
            note += ("\n[Other speech-recognition hypothesis of the same utterance (a different language model; one of the two may be garbled): "
                     + " | ".join(alternatives) + "]")
        user_msg = text + note

        # 2. retrieval query (LLM rewrite when the message is non-English / a short follow-up)
        rq = self._rewrite(session, text)
        res.rewritten_query = rq
        hits = self.retriever.search(f"{rq} {text}" if rq and rq != text else text, k=settings.top_k)
        chunk_ids = [h.chunk.id for h in hits]
        res.chunks = [{"id": h.chunk.id, "title": h.chunk.title, "score": round(h.score, 4)} for h in hits]
        seen = list(chunk_ids)

        # 3. prompt
        system = self._system(channel) + "\n" + CONTEXT_TEMPLATE.format(
            state=session.state_summary(), first="YES - introduce yourself as the AI assistant" if first else "no",
            events="\n".join(f"- {e}" for e in events) or "- none",
            orders=order_context(session),
            chunks="\n\n".join(h.chunk.render() for h in hits) or "(nothing retrieved)")
        messages = session.history + [{"role": "user", "content": user_msg}]

        # 4. answer with tools
        ticket_before = len(session.tickets)
        tool_trace: list[dict] = []

        def handler(name: str, args: dict):
            out = self._run_tool(session, name, args, seen)
            return out

        try:
            llm_res = self.llm.complete(system, messages, tools=TOOLS, tool_handler=handler, max_tokens=500)
            reply = clean_reply(llm_res.text)
            tool_trace = llm_res.tool_trace
            res.usage = llm_res.usage

            # 5. guardrails (one corrective regeneration at most)
            problem = self._guard(reply, session, ticket_before)
            if problem:
                res.guard = problem
                fix = system + f"\n\n## CORRECTION\nYour previous draft was rejected: {problem}\nWrite a corrected reply now."
                llm_res2 = self.llm.complete(fix, messages, tools=TOOLS, tool_handler=handler, max_tokens=500)
                tool_trace += llm_res2.tool_trace
                reply = clean_reply(llm_res2.text) or reply
                if self._guard(reply, session, ticket_before):
                    reply = self._safe_fallback(first)
            if not reply:
                reply = self._safe_fallback(first)
        except LLMError as exc:
            res.error = str(exc)
            reply = "Sorry, I'm having trouble right now. Please try again in a moment."
        except Exception as exc:  # noqa: BLE001  never crash a call
            res.error = f"{type(exc).__name__}: {exc}"
            reply = "Sorry, I'm having trouble right now. Please try again in a moment."

        reply = scrub_secrets(reply, session)

        # 6. book-keeping
        session.history.append({"role": "user", "content": text})
        session.history.append({"role": "assistant", "content": reply})
        res.reply = reply
        res.tool_calls = [{"name": e["name"], "args": e["args"], "result": e["result"]} for e in
                          (harness_calls + tool_trace)]
        session.tool_log += tool_trace
        if len(session.tickets) > ticket_before:
            res.ticket_id = session.tickets[-1]["ticket_id"]
        res.sources = self._sources(seen, first, events, res.ticket_id, session)
        res.latency_s = round(time.time() - t0, 2)
        return res

    # ------------------------------------------------------------------ internals
    def _system(self, channel: str) -> str:
        d = date.fromisoformat(TODAY)
        return SYSTEM_TEMPLATE.format(
            channel_desc="a live phone call (speech-to-text in, text-to-speech out)" if channel == "voice" else "text chat",
            today=TODAY, weekday=d.strftime("%A"),
            style_channel=STYLE_VOICE if channel == "voice" else STYLE_TEXT, pinned=self._pinned)

    def _needs_rewrite(self, session: Session, text: str) -> bool:
        mode = settings.query_rewrite
        if mode == "never":
            return False
        if mode == "always":
            return True
        non_latin = any(ord(c) > 127 and c.isalpha() for c in text)
        from .textnorm import detect_script_language

        hinglish = detect_script_language(text) == "hi-en"
        short_followup = bool(session.history) and len(text.split()) <= 9
        return non_latin or hinglish or short_followup

    def _rewrite(self, session: Session, text: str) -> str:
        if not text or not self._needs_rewrite(session, text):
            return text
        facts = ""
        if session.active_oid in session.verified:
            o = session.verified[session.active_oid]
            facts = json.dumps({k: o.get(k) for k in ("status", "payment_method", "delivery_type", "delivered_on", "return_status", "damage_status", "is_plus")
                                if o.get(k) is not None} | {"items": [f"{i.get('title')} ({i.get('return_category')})" for i in o.get("items", [])]}, ensure_ascii=False)
        recent = "\n".join(f"{m['role']}: {m['content'][:160]}" for m in session.history[-4:])
        prompt = f"Conversation so far:\n{recent or '(none)'}\nVerified order facts: {facts or 'none'}\nCustomer message: {text}"
        try:
            out = self.llm.complete(REWRITE_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=120).text
            m = re.search(r"\{.*\}", out, re.S)
            q = json.loads(m.group(0)).get("query", "") if m else ""
            return q.strip() or text
        except Exception:  # noqa: BLE001
            return text

    def _run_tool(self, session: Session, name: str, args: dict, seen: list[str]):
        if name == "search_kb":
            hits = self.retriever.search(str(args.get("query", "")), k=4)
            for h in hits:
                if h.chunk.id not in seen:
                    seen.append(h.chunk.id)
            return [{"id": h.chunk.id, "text": h.chunk.render()} for h in hits]
        if name == "lookup_product":
            rows = find_products(str(args.get("query", "")))
            for r in rows:
                cid = f"products.csv:{r['sku']}"
                if cid not in seen:
                    seen.append(cid)
            return rows or {"found": False}
        if name == "check_pincode":
            return check_pincode(str(args.get("pincode", "")))
        if name == "date_calc":
            return dates_mod.date_calc(args.get("op", ""), args.get("date1", "today"), args.get("date2"), args.get("days"))
        if name == "create_ticket":
            return self._create_ticket(session, args)
        return {"error": f"unknown tool {name}"}

    def _create_ticket(self, session: Session, args: dict) -> dict:
        from tools import create_ticket  # type: ignore  (dataset's mock tool)

        reason = str(args.get("reason") or "")
        prev = next((t for t in session.tickets if t["reason"] == reason), None)
        if prev:  # one ticket per reason per conversation
            return {"created": True, "ticket_id": prev["ticket_id"], "note": "ticket for this reason already exists in this conversation"}
        summary = scrub_secrets(str(args.get("summary", "")), session)
        oid = session.active_oid if session.active_oid in session.verified else None
        r = create_ticket(summary=summary, order_id=oid, reason=reason or None, intent=args.get("intent") or None)
        if r.get("created"):
            session.tickets.append({"ticket_id": r["ticket_id"], "reason": reason, "order_id": oid})
        return r

    def _guard(self, reply: str, session: Session, tickets_before: int) -> str:
        if _FAKE_ACTION.search(reply):
            return "it claims you performed an action (cancel/approve/refund/credit). You cannot do these; explain what the customer can do or hand over via create_ticket."
        if _FAKE_TICKET.search(reply) and not session.tickets:
            return "it claims a ticket/handover exists but create_ticket has not returned created=true. Call create_ticket first, or do not claim it."
        return ""

    @staticmethod
    def _safe_fallback(first: bool) -> str:
        pre = "I'm ZentroMart's AI assistant. " if first else ""
        return pre + "Sorry, I couldn't complete that safely. I can connect you to a human colleague who can help."

    @staticmethod
    def _sources(seen: list[str], first: bool, events: list[str], ticket_id: str | None, session: Session) -> list[str]:
        extra = []
        if first:
            extra.append("STY-1")
        ev = " ".join(events)
        if "verification_required" in ev or "VERIFIED" in ev or "NOT VERIFIED" in ev or "LOCKED" in ev:
            extra.append("IDV-2")
        if "NOT VERIFIED" in ev or "LOCKED" in ev:
            extra.append("IDV-3")
        if ticket_id:
            extra += ["ESC-1", "ESC-2", "ESC-3"]
        out: list[str] = []
        for x in seen + extra:
            if x not in out:
                out.append(x)
        return out


class PromptOnlyAgent:
    """Step 1 baseline: persona prompt only. No documents, no tools, no verification harness."""

    def __init__(self, llm: BaseLLM | None = None):
        self.llm = llm or get_llm()

    def new_session(self, session_id: str = "default", channel: str = "text") -> Session:
        return Session(session_id=session_id, channel=channel)

    def respond(self, session: Session, text: str, channel: str | None = None) -> AgentResult:
        t0 = time.time()
        messages = session.history + [{"role": "user", "content": text}]
        try:
            out = self.llm.complete(BASELINE_SYSTEM, messages, max_tokens=300)
            reply = clean_reply(out.text)
        except Exception as exc:  # noqa: BLE001
            return AgentResult(reply="", error=str(exc))
        session.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return AgentResult(reply=reply, usage=out.usage, latency_s=round(time.time() - t0, 2))
