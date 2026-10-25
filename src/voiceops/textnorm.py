"""Text normalisation for noisy speech-to-text and Indian-language input.

ASR text has spelled-out numbers ("four five three three"), homophone errors
("to" for "two") and no punctuation. The agent needs order IDs, phone suffixes,
pincodes and SKUs as digits, so this module recovers them deterministically.
"""
from __future__ import annotations

import re
import unicodedata

_DEVANAGARI_DIGITS = {ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")}
# other Indic digit blocks -> ASCII
for _start in (0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66):
    for _i in range(10):
        _DEVANAGARI_DIGITS[_start + _i] = str(_i)

# unambiguous digit words (English + Hindi, romanised + Devanagari)
STRONG = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "shunya": 0, "ek": 1, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "chhah": 6,
    "chheh": 6, "saat": 7, "aath": 8, "nau": 9,
    "शून्य": 0, "ज़ीरो": 0, "जीरो": 0, "एक": 1, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6,
    "सात": 7, "आठ": 8, "नौ": 9,
}
# ambiguous words that are only digits when sitting inside a run of digit tokens
WEAK = {"oh": 0, "o": 0, "to": 2, "too": 2, "tu": 2, "for": 4, "fore": 4, "ate": 8, "won": 1, "do": 2, "दो": 2, "nine": 9}
MULTIPLIERS = {"double": 2, "triple": 3, "dubble": 2}


def ascii_digits(text: str) -> str:
    """Convert native-script digits (Devanagari, Tamil, ...) to ASCII."""
    return text.translate(_DEVANAGARI_DIGITS)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w₹.']+|[^\w\s]", text, flags=re.UNICODE)


def _tok_digits(tok: str) -> str | None:
    t = tok.lower().strip(".,;:!?।")
    if t.isdigit():
        return t
    if t in STRONG:
        return str(STRONG[t])
    return None


def _tok_weak(tok: str) -> str | None:
    t = tok.lower().strip(".,;:!?।")
    if t in WEAK:
        return str(WEAK[t])
    return None


def collapse_digit_runs(text: str, lengths=(4, 6, 8)) -> str:
    """Collapse runs of spoken/spaced digits into one number when the total digit count looks like
    an ID: 4 (phone suffix), 6 (pincode) or 8 (order ID). Shorter runs ("five to seven days") stay as is."""
    text = ascii_digits(text)
    toks = _tokenize(text)
    out: list[str] = []
    i = 0
    while i < len(toks):
        j = i
        digits: list[str] = []
        has_strong = False
        k = i
        pending_mult = 1
        while k < len(toks):
            tok = toks[k]
            low = tok.lower()
            if low in MULTIPLIERS:
                pending_mult = MULTIPLIERS[low]
                k += 1
                continue
            d = _tok_digits(tok)
            if d is not None:
                digits.append(d * pending_mult)
                pending_mult = 1
                has_strong = True
                k += 1
                j = k
                continue
            w = _tok_weak(tok)
            if w is not None and digits:  # weak tokens only continue an existing run
                digits.append(w * pending_mult)
                pending_mult = 1
                k += 1
                j = k
                continue
            break
        joined = "".join(digits)
        # trim trailing weak token (e.g. "... four five to" where "to" is a real word)
        if has_strong and len(joined) in lengths and j > i:
            out.append(joined)
            i = j
        else:
            out.append(toks[i])
            i += 1
    # re-join, spacing tokens sensibly
    s = ""
    for t in out:
        if not s:
            s = t
        elif re.match(r"^[.,;:!?।)\]]$", t):
            s += t
        else:
            s += " " + t
    return s


ORDER_RE = re.compile(r"(?<!\d)(\d{4})[\s\-]?(\d{4})(?!\d)")
SIX_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
FOUR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def extract_order_ids(text: str) -> list[str]:
    """8-digit order IDs, written as 89782837 / 8978 2837 / 8978-2837 or spoken."""
    norm = collapse_digit_runs(text)
    ids = []
    for m in re.finditer(r"(?<!\d)(\d{8})(?!\d)", norm):
        ids.append(m.group(1))
    for m in ORDER_RE.finditer(norm):
        oid = m.group(1) + m.group(2)
        if oid not in ids:
            ids.append(oid)
    return ids


def extract_verification_candidates(text: str, exclude: list[str] | None = None) -> list[str]:
    """Return 6-digit pincode / 4-digit phone-suffix candidates, excluding digits that belong to an order ID."""
    norm = collapse_digit_runs(text)
    for oid in (exclude or []) + extract_order_ids(text):
        norm = norm.replace(oid, " ").replace(oid[:4] + " " + oid[4:], " ")
    norm = ORDER_RE.sub(" ", norm)
    cands = SIX_RE.findall(norm)
    rest = SIX_RE.sub(" ", norm)
    cands += FOUR_RE.findall(rest)
    return cands


def extract_skus(text: str, prefixes: list[str], known: set[str] | None = None) -> list[str]:
    """Find catalog SKUs (e.g. ELE-0012) in typed or spoken text ("ele zero zero one to")."""
    t = ascii_digits(text).lower()
    found: list[str] = []
    for pre in prefixes:
        for m in re.finditer(rf"(?<![a-z]){pre.lower()}(?![a-z])[\s\-_]*((?:[a-z0-9]+[\s\-]*){{1,6}})", t):
            digs = ""
            for tok in re.findall(r"[a-z0-9]+", m.group(1)):
                d = _tok_digits(tok) or _tok_weak(tok)
                if d is None:
                    if tok.isdigit():
                        d = tok
                    else:
                        break
                digs += d
                if len(digs) >= 4:
                    break
            if len(digs) >= 4:
                sku = f"{pre}-{digs[:4]}"
                if (known is None or sku in known) and sku not in found:
                    found.append(sku)
    return found


# ------------------------------------------------------------------ language helpers
_SCRIPT_RANGES = [
    ("hi-IN", 0x0900, 0x097F),  # Devanagari (Hindi / Marathi)
    ("bn-IN", 0x0980, 0x09FF),
    ("pa-IN", 0x0A00, 0x0A7F),
    ("gu-IN", 0x0A80, 0x0AFF),
    ("ta-IN", 0x0B80, 0x0BFF),
    ("te-IN", 0x0C00, 0x0C7F),
    ("kn-IN", 0x0C80, 0x0CFF),
    ("ml-IN", 0x0D00, 0x0D7F),
]

_HINGLISH_WORDS = {
    "hai", "hain", "kya", "aap", "mujhe", "mera", "meri", "mere", "nahi", "nahin", "kar", "karna", "karein", "kijiye",
    "ka", "ke", "ki", "ko", "se", "mein", "me", "aur", "kab", "kitne", "kitna", "din", "paisa", "paise", "wapas", "aaya",
    "aayega", "chahiye", "sakta", "sakti", "sakte", "bhi", "toh", "tha", "thi", "ho", "hoon", "raha", "rahi", "abhi",
    "kaise", "kyun", "kahan", "pe", "par", "wala", "wali", "lena", "dena", "batao", "bataiye", "haan", "theek",
}


def detect_script_language(text: str) -> str:
    """Rough TTS language code for a reply: en-IN, hi-IN, hi-en (Hinglish) or another Indic code."""
    counts: dict[str, int] = {}
    for ch in text:
        o = ord(ch)
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= o <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if counts:
        return max(counts, key=counts.get)
    words = re.findall(r"[a-z]+", text.lower())
    if sum(1 for w in words if w in _HINGLISH_WORDS) >= 2:
        return "hi-en"
    return "en-IN"


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
