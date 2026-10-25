"""Deterministic scorer that follows DATA_DICTIONARY.md for `must_include` / `must_not_include`:

  * facts are separated by " ; " (ALL required); alternatives inside a fact by " / " (ANY one is enough)
  * case-insensitive; equivalent dates (2026-10-27 == 27 October 2026 == 27 अक्टूबर 2026) and spoken numbers are accepted
  * must_not_include phrases are forbidden claims (literal check; a paraphrase would also be wrong, which a literal check cannot see)
"""
from __future__ import annotations

import re
from datetime import date

from .textnorm import ascii_digits

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    "जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9,
    "अक्टूबर": 10, "अक्तूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
}
_MONTH_RE = "|".join(sorted(map(re.escape, _MONTHS), key=len, reverse=True))

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def words_to_numbers(text: str) -> str:
    """'forty-nine rupees' -> '49 rupees', 'two hours' -> '2 hours', 'twenty seven' -> '27' (up to 99, plus 'hundred'/'thousand')."""
    toks = re.split(r"(\s+|-)", text)
    out, i = [], 0
    words = [t for t in toks]
    while i < len(words):
        w = words[i].lower()
        if w in _TENS:
            val, j = _TENS[w], i + 1
            k = j
            while k < len(words) and words[k] in ("-", " ", "  "):
                k += 1
            if k < len(words) and words[k].lower() in _ONES and 0 < _ONES[words[k].lower()] < 10:
                val += _ONES[words[k].lower()]
                j = k + 1
            out.append(str(val))
            i = j
        elif w in _ONES:
            out.append(str(_ONES[w]))
            i += 1
        else:
            out.append(words[i])
            i += 1
    s = "".join(out)
    s = re.sub(r"(\d+)\s*thousand", lambda m: str(int(m.group(1)) * 1000), s, flags=re.I)
    return s


def extract_dates(text: str, default_year: int = 2026) -> set[str]:
    t = ascii_digits(text.lower())
    found: set[str] = set()

    def add(y, m, d):
        try:
            found.add(date(int(y), int(m), int(d)).isoformat())
        except ValueError:
            pass

    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", t):
        add(*m.groups())
    for m in re.finditer(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({_MONTH_RE})\.?,?\s*(\d{{4}})?", t):
        add(m.group(3) or default_year, _MONTHS[m.group(2)], m.group(1))
    for m in re.finditer(rf"({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?", t):
        add(m.group(3) or default_year, _MONTHS[m.group(1)], m.group(2))
    for m in re.finditer(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", t):
        add(m.group(3), m.group(2), m.group(1))
    return found


def _norm(s: str) -> str:
    s = ascii_digits(s.lower())
    s = words_to_numbers(s)
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)          # 10,000 -> 10000
    s = s.replace("₹", " ₹").replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def fact_present(alternatives: str, reply: str) -> bool:
    nrep = _norm(reply)
    rep_dates = extract_dates(reply)
    for alt in (a.strip() for a in alternatives.split(" / ")):
        if not alt:
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", alt):
            if alt in rep_dates:
                return True
            continue
        a = _norm(alt)
        if re.fullmatch(r"\d+", a):  # bare numbers: whole-token match only ("7" must not hit "17")
            if re.search(rf"(?<![\d]){a}(?![\d])", nrep):
                return True
        elif a in nrep:
            return True
    return False


def check_must_include(must_include: str, reply: str) -> tuple[bool, list[str]]:
    facts = [f.strip() for f in (must_include or "").split(" ; ") if f.strip()]
    missing = [f for f in facts if not fact_present(f, reply)]
    return not missing, missing


def check_must_not(must_not: str, reply: str) -> tuple[bool, list[str]]:
    bad = []
    for ph in (p.strip() for p in (must_not or "").split(" ; ") if p.strip()):
        if _norm(ph) in _norm(reply):
            bad.append(ph)
    return not bad, bad


def score_reply(row: dict, reply: str) -> dict:
    ok_inc, missing = check_must_include(row.get("must_include", ""), reply)
    ok_not, bad = check_must_not(row.get("must_not_include", ""), reply)
    return {"include_ok": ok_inc, "missing": missing, "forbidden_ok": ok_not, "forbidden_hit": bad, "pass": ok_inc and ok_not}
