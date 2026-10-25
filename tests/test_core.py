"""Offline tests: no LLM or network needed. Run: pytest -q  (needs DATASET_DIR)."""
import csv, os
import pytest
from voiceops import dates
from voiceops.textnorm import (collapse_digit_runs, extract_order_ids, extract_skus, extract_verification_candidates, detect_script_language)
from voiceops.kb import sku_prefixes, products
from voiceops.retriever import Retriever


def test_business_days_matches_practice_example():
    # PQ0008: card refund initiated Sat 17 Oct 2026, 5-7 business days -> 24..27 Oct (Sundays + 19 Oct closure skipped)
    start = dates.parse("2026-10-17")
    assert dates.add_business_days(start, 5).isoformat() == "2026-10-24"
    assert dates.add_business_days(start, 7).isoformat() == "2026-10-27"
    assert dates.business_days_between(dates.parse("2026-10-17"), dates.parse("2026-10-27")) == 7


def test_asr_numbers():
    assert extract_order_ids("the order number is eight nine seven eight two eight three seven") == ["89782837"]
    assert extract_order_ids("Order number hai 8978 2837.") == ["89782837"]
    assert extract_order_ids("ऑर्डर नंबर है ३७११ ३६७६।") == ["37113676"]
    assert extract_verification_candidates("last four digits are four five three three") == ["4533"]
    assert extract_verification_candidates("Pincode 440010 hai") == ["440010"]
    assert extract_verification_candidates("order 8978 2837 and last four 4533") == ["4533"]
    assert "five to seven" in collapse_digit_runs("refund in five to seven days")  # short runs untouched


def test_sku_from_speech():
    assert extract_skus("so sku ele-zero zero one to pe assured", sku_prefixes(), set(products())) == ["ELE-0012"]


def test_language_detection():
    assert detect_script_language("आपका रिफ़ंड जल्दी आएगा") == "hi-IN"
    assert detect_script_language("Aapka refund 2 din mein aa jayega, kya aur kuch chahiye") == "hi-en"
    assert detect_script_language("Your refund will arrive soon") == "en-IN"


def test_retrieval_gold_on_practice_questions():
    d = os.environ["DATASET_DIR"]
    rows = list(csv.DictReader(open(f"{d}/practice/questions.csv", encoding="utf-8")))
    r = Retriever()
    ok = n = 0
    for row in rows:
        gold = [g.strip() for g in row["gold_evidence"].split(";") if g.strip()]
        if not gold:
            continue
        n += 1
        ids = [h.chunk.id for h in r.search(row["question"], k=5)]
        ok += all(g in ids for g in gold)
    assert ok / n >= 0.85
