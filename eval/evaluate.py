"""Evaluation on the practice set.

  python eval/evaluate.py --suite questions      --agents baseline,rag     # Step 1 vs Step 2 (the number the report needs)
  python eval/evaluate.py --suite conversations  --agents rag              # Step 3: multi-turn, tools, handovers
  python eval/evaluate.py --suite all --field asr_text                     # simulate speech-to-text input

Scoring is deterministic (src/voiceops/scoring.py): must_include facts, must_not_include phrases, retrieval recall vs
gold_evidence, expected tool calls, and "ticket created iff escalation expected". Results are written to eval/results/<run>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceops.config import ensure_dataset_on_path  # noqa: E402
from voiceops.scoring import score_reply  # noqa: E402

TOOL_RE = re.compile(r"(\w+)\((.*)\)")


def parse_expected_tools(s: str) -> list[dict]:
    out = []
    for part in re.findall(r"\w+\([^)]*\)", s or ""):
        name, args = TOOL_RE.match(part).groups()
        kv = dict(a.strip().split("=", 1) for a in args.split(",") if "=" in a)
        out.append({"name": name, "args": kv})
    return out


def tools_ok(expected: list[dict], made: list[dict]) -> bool:
    for e in expected:
        hit = False
        for m in made:
            if m["name"] != e["name"]:
                continue
            if all(str(m["args"].get(k)) == v for k, v in e["args"].items()):
                hit = True
                break
        if e["name"] == "create_ticket":
            hit = any(m["name"] == "create_ticket" and (m.get("result") or {}).get("created") for m in made)
        if not hit:
            return False
    return True


def make_agents(names: list[str]):
    from voiceops.agent import PromptOnlyAgent, SupportAgent
    from voiceops.llm import get_llm

    llm = get_llm()
    agents = {}
    if "baseline" in names:
        agents["baseline"] = PromptOnlyAgent(llm)
    if "rag" in names:
        agents["rag"] = SupportAgent(llm)
    return agents


def sandbox_tickets():
    """Write tickets created during evaluation to a temp file, not into the dataset folder."""
    import tools.tools as t  # type: ignore

    t._TICKETS = str(Path(tempfile.mkdtemp()) / "tickets_eval.csv")


def eval_questions(agents, rows, field, out_dir):
    summary = {}
    for name, agent in agents.items():
        recs, t0 = [], time.time()
        for row in rows:
            s = agent.new_session(row["id"], "text")
            r = agent.respond(s, row[field])
            sc = score_reply(row, r.reply)
            gold = [g.strip() for g in row["gold_evidence"].split(";") if g.strip()]
            rec = {"id": row["id"], "language": row["language"], "intent": row["intent"], "expected": row["expected_decision"],
                   "answerable": row["answerable"], "question": row[field], "reply": r.reply, **sc,
                   "sources": r.sources, "gold": gold, "ticket": r.ticket_id, "error": r.error, "latency_s": r.latency_s}
            if gold and name == "rag":
                rec["recall"] = sum(g in r.sources for g in gold) / len(gold)
            if row["expected_decision"] == "escalate":
                rec["escalation_ok"] = bool(r.ticket_id)
                rec["pass"] = rec["pass"] and rec["escalation_ok"]
            recs.append(rec)
        (out_dir / f"questions_{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in recs), encoding="utf-8")
        summary[name] = aggregate_questions(recs, time.time() - t0)
    return summary


def aggregate_questions(recs, secs):
    n = len(recs)
    by = lambda key: {k: f"{sum(r['pass'] for r in v)}/{len(v)}" for k, v in _group(recs, key).items()}  # noqa: E731
    out = {"n": n, "pass": sum(r["pass"] for r in recs), "pass_rate": round(sum(r["pass"] for r in recs) / n, 3),
           "by_language": by("language"), "by_intent": by("intent"), "errors": sum(bool(r["error"]) for r in recs),
           "unanswerable_pass": f"{sum(r['pass'] for r in recs if r['answerable'] == 'False')}/{sum(r['answerable'] == 'False' for r in recs)}",
           "avg_latency_s": round(sum(r["latency_s"] for r in recs) / n, 2), "wall_s": round(secs, 1)}
    rr = [r["recall"] for r in recs if "recall" in r]
    if rr:
        out["retrieval_recall"] = round(sum(rr) / len(rr), 3)
    return out


def _group(recs, key):
    g = defaultdict(list)
    for r in recs:
        g[r[key]].append(r)
    return g


def eval_conversations(agents, rows, field, out_dir):
    convs = defaultdict(list)
    for r in rows:
        convs[r["conversation_id"]].append(r)
    summary = {}
    for name, agent in agents.items():
        recs = []
        for cid, turns in convs.items():
            turns.sort(key=lambda r: int(r["turn"]))
            s = agent.new_session(cid, "text")
            for row in turns:
                r = agent.respond(s, row["customer_message"])
                sc = score_reply(row, r.reply)
                exp_tools = parse_expected_tools(row["expected_tool_calls"])
                t_ok = tools_ok(exp_tools, r.tool_calls) if name == "rag" else None
                esc_ok = (bool(r.ticket_id) if row["expected_decision"] == "escalate" else None) if name == "rag" else None
                gold = [g.strip() for g in row["gold_evidence"].split(";") if g.strip()]
                rec = {"conversation_id": cid, "turn": int(row["turn"]), "scenario": row["scenario"], "language": row["language"],
                       "message": row["customer_message"], "reply": r.reply, **sc, "tools_ok": t_ok, "escalation_ok": esc_ok,
                       "expected": row["expected_decision"], "tool_calls": [c["name"] for c in r.tool_calls],
                       "sources": r.sources, "gold": gold, "error": r.error}
                rec["turn_pass"] = sc["pass"] and (t_ok in (True, None)) and (esc_ok in (True, None))
                if gold:
                    rec["recall"] = sum(g in r.sources for g in gold) / len(gold)
                recs.append(rec)
        (out_dir / f"conversations_{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in recs), encoding="utf-8")
        n = len(recs)
        conv_pass = {cid: all(r["turn_pass"] for r in recs if r["conversation_id"] == cid) for cid in convs}
        rr = [r["recall"] for r in recs if "recall" in r]
        summary[name] = {
            "turns": n, "turn_pass": sum(r["turn_pass"] for r in recs), "turn_pass_rate": round(sum(r["turn_pass"] for r in recs) / n, 3),
            "conversations": len(convs), "conversation_pass": sum(conv_pass.values()),
            "tool_call_ok": f"{sum(r['tools_ok'] is True for r in recs)}/{sum(r['tools_ok'] is not None for r in recs)}",
            "escalations_ok": f"{sum(r['escalation_ok'] is True for r in recs)}/{sum(r['escalation_ok'] is not None for r in recs)}",
            "by_scenario": {k: f"{sum(r['turn_pass'] for r in v)}/{len(v)}" for k, v in _group(recs, 'scenario').items()},
            "retrieval_recall": round(sum(rr) / len(rr), 3) if rr else None,
            "failed_turns": [f"{r['conversation_id']}#{r['turn']}" for r in recs if not r["turn_pass"]],
        }
    return summary


def markdown(summary: dict, args) -> str:
    L = [f"# Evaluation summary  (model: {__import__('os').getenv('LLM_MODEL', '?')}, input field: `{args.field}`)\n"]
    q = summary.get("questions")
    if q:
        L += ["## Single questions (practice/questions.csv)\n", "| Agent | Passed | Rate | Retrieval recall | Unanswerable handled | Avg latency |", "|---|---|---|---|---|---|"]
        for a, s in q.items():
            L.append(f"| {'Step 1 prompt-only' if a == 'baseline' else 'Step 2 RAG agent'} | {s['pass']}/{s['n']} | {s['pass_rate']:.0%} | "
                     f"{s.get('retrieval_recall', '-')} | {s['unanswerable_pass']} | {s['avg_latency_s']}s |")
        for a, s in q.items():
            L += [f"\n**{a} by language:** {s['by_language']}", f"\n**{a} by intent:** {s['by_intent']}"]
    c = summary.get("conversations")
    if c:
        L += ["\n## Multi-turn conversations (practice/conversations.csv)\n"]
        for a, s in c.items():
            L += [f"- **{a}**: turns {s['turn_pass']}/{s['turns']} ({s['turn_pass_rate']:.0%}); conversations fully correct {s['conversation_pass']}/{s['conversations']}; "
                  f"tool calls {s['tool_call_ok']}; escalations {s['escalations_ok']}; retrieval recall {s['retrieval_recall']}",
                  f"  - by scenario: {s['by_scenario']}", f"  - failed turns: {s['failed_turns']}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["questions", "conversations", "all"])
    ap.add_argument("--agents", default="baseline,rag")
    ap.add_argument("--field", default="question", choices=["question", "asr_text"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions / conversations (quick smoke test)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = ensure_dataset_on_path()
    sandbox_tickets()
    out_dir = Path(args.out or Path(__file__).parent / "results" / time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    agents = make_agents([a.strip() for a in args.agents.split(",")])
    summary = {}
    if args.suite in ("questions", "all"):
        rows = list(csv.DictReader(open(d / "practice" / "questions.csv", encoding="utf-8", newline="")))
        rows = rows[: args.limit] if args.limit else rows
        summary["questions"] = eval_questions(agents, rows, args.field, out_dir)
    if args.suite in ("conversations", "all"):
        rows = list(csv.DictReader(open(d / "practice" / "conversations.csv", encoding="utf-8", newline="")))
        if args.limit:
            keep = list(dict.fromkeys(r["conversation_id"] for r in rows))[: args.limit]
            rows = [r for r in rows if r["conversation_id"] in keep]
        summary["conversations"] = eval_conversations({k: v for k, v in agents.items() if k == "rag"} or agents, rows, args.field, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(summary, args)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Details: {out_dir}")


if __name__ == "__main__":
    main()
