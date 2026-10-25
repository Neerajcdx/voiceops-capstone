"""Offline retrieval evaluation (no LLM needed): does the retriever surface the gold evidence chunks?

Usage: python eval/retrieval_eval.py [--k 6] [--field question|asr_text] [--verbose]
Metrics: recall@k of gold section IDs (macro-averaged), full-hit rate (all gold IDs found), MRR of first gold hit.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceops.config import ensure_dataset_on_path  # noqa: E402
from voiceops.retriever import Retriever  # noqa: E402


def run(k: int, field: str, verbose: bool) -> dict:
    d = ensure_dataset_on_path()
    r = Retriever()
    rows = list(csv.DictReader(open(d / "practice" / "questions.csv", encoding="utf-8", newline="")))
    per_lang: dict[str, list[float]] = {}
    recalls, full, mrr, n = [], 0, [], 0
    for row in rows:
        gold = [g.strip() for g in row["gold_evidence"].split(";") if g.strip()]
        if not gold:
            continue
        n += 1
        hits = r.search(row[field], k=k)
        ids = [h.chunk.id for h in hits]
        found = [g for g in gold if g in ids]
        rec = len(found) / len(gold)
        recalls.append(rec)
        per_lang.setdefault(row["language"], []).append(rec)
        full += rec == 1.0
        rr = next((1 / (i + 1) for i, x in enumerate(ids) if x in gold), 0.0)
        mrr.append(rr)
        if verbose and rec < 1.0:
            print(f"{row['id']} [{row['language']}] recall={rec:.2f} gold={gold} got={ids}\n   Q: {row[field][:110]}")
    res = {
        "k": k, "field": field, "n": n,
        "recall@k": round(sum(recalls) / n, 3), "full_hit_rate": round(full / n, 3), "mrr": round(sum(mrr) / n, 3),
        "by_language": {l: round(sum(v) / len(v), 3) for l, v in per_lang.items()},
    }
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--field", default="question", choices=["question", "asr_text"])
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    print(run(a.k, a.field, a.verbose))
