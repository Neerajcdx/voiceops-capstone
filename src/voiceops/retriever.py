"""Hybrid retriever over KB chunks: BM25 (words) + TF-IDF char n-grams (+ optional dense embeddings),
fused with reciprocal-rank fusion, then re-weighted by document precedence (KB-00.1).

Customers don't use the documents' words ("send it back" vs "reverse pickup"), and many write Hindi/Hinglish.
Three things bridge that gap: (1) a small domain glossary below, (2) char n-grams, (3) an LLM query rewrite
done by the agent (see agent.py) that produces an English standalone query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import settings
from .kb import Chunk, load_kb_chunks, products, sku_prefixes
from .textnorm import ascii_digits, extract_skus

STOP = set("""a an the is are was were be been am i me my we our you your it its of to for on in at by with from and or but if
so do does did can could would should will shall may might have has had this that these those there here what which who whom
how when where why not no yes please hi hello hey just also about into than then them they he she him her his
uh um ok okay tell want need get got""".split())

# Domain glossary: customer wording / Hindi / Hinglish -> the documents' vocabulary. Terms only, never answers.
GLOSSARY = {
    r"send (it |this |them )?back|give back|take back|wapas|वापस|लौटा|रिटर्न|return": "return returns policy",
    r"how (many|long).*(days|return|send back)|window|deadline|last date|kitne din|कितने दिन|कब तक|din tak|how long (do|can|to)": "return window days category delivery date calendar days",
    r"already on (its|the) way|on the way|stop (it|the order)|shipped|out for delivery|रास्ते में|शिप हो": "cancel shipped order cannot be cancelled refuse parcel",
    r"nobody came|no one came|didn.?t come|pickup|collect|कलेक्ट|पिकअप": "reverse pickup missed pickup",
    r"money back|paise|पैसे|पैसा|रिफ़ंड|रिफंड|refund|वापसी|reimburse": "refund refunds timeline initiated business days",
    r"cancel|कैंसिल|रद्द|radd": "cancel cancellation shipped",
    r"late|delay|देर|late aaya|not arrived|hasn.?t arrived|nahi aaya|नहीं आया|where is my order|track": "delivery late delayed promised date tracking status",
    r"broken|cracked|crack|damage|damaged|defect|faulty|kharab|खराब|टूट|टूटा|wrong item|missing|galat|गलत": "damaged defective wrong missing item report 48 hours",
    r"warranty|guarantee|वारंटी|गारंटी|assured|claim": "warranty assured brand warranty claim",
    r"cod|cash on delivery|कैश ऑन": "cash on delivery cod fee refund",
    r"coupon|promo|voucher|कूपन|discount|code": "coupon coupons offers",
    r"plus|member|membership|प्लस": "zentromart plus membership benefits",
    r"human|agent|person|colleague|manager|इंसान|इंसान से|insaan|senior|escalat": "when to hand over to a human agent escalation colleague ticket",
    r"deliver.*(to|in)|pincode|pin code|area|serviceable|पिनकोड|abroad|outside india|international": "pincode serviceable delivery area where we deliver",
    r"express|same day|next day|quick": "express delivery",
    r"exchange|swap|size|colour|color|doesn.?t fit|not fit|fit nahi|फिट": "exchange size return clothing footwear",
    r"verify|verification|otp|identity|authenticate": "verifying the customer last 4 digits phone delivery pincode",
    r"fraud|not me|never (placed|ordered)|didn.?t (order|place)|unauthori[sz]ed|hacked|scam|धोखा|मैंने नहीं": "when to hand over to a human customer reports an order payment or account activity they did not make",
    r"court|legal|lawyer|police|consumer forum|sue|case file": "when to hand over to a human legal action consumer court police complaint",
    r"smoke|burning|swelling|swollen|spark|overheat|fire|shock|injur": "product safety risk swelling battery burning smell sparks injury hand over to a human",
    r"earbud|earphone|airpod|watch|speaker|charging|bluetooth|not working|not turning on|doesn.?t work": "troubleshooting electronics earbuds smartwatch charging bluetooth speaker",
    r"wallet|वॉलेट": "zentromart wallet instant refund",
    r"bank|neft|transfer|बैंक": "bank transfer neft refund",
    r"upi|card|credit|debit|netbanking|net banking": "refund payment method upi card netbanking timeline",
    r"fee|charge|pickup fee|shipping|delivery fee|free delivery|फीस": "fee charges reverse pickup fee delivery fee free delivery",
    r"holiday|sunday|sundays|working day|business day|कामकाजी": "business days holidays sundays",
    r"slot|grocery|vegetable|milk|fresh": "grocery delivery slot perishable not returnable",
    r"seller|marketplace|third party|sold by|fulfilled": "marketplace seller fulfilled by",
    r"sealed|opened|unopened|seal|cream|shampoo|makeup|lipstick|beauty": "beauty personal care unopened seal intact",
    r"innerwear|underwear|brief|bra|lingerie|swimsuit|swimwear|trunks": "innerwear lingerie swimwear not returnable",
    r"festive|sale|diwali|festival|30 days": "festive sale returns clothing footwear bags 30 days",
    r"refuse|rto|returned to origin|not at home|attempt|nobody home": "delivery attempts rto refuse parcel",
    r"change (the )?address|wrong address|update address|new address": "changing the address before shipping",
    r"phone number|email|address of|who (ordered|bought)|someone else|another customer|other customer|neighbou?r": "never read out phone number email address never share another customer's details verifying the customer",
    r"compensat|voucher|gift|goodwill|sorry|apolog": "late deliveries compensation not allowed credit express",
    r"gift card|digital code|e-?gift": "gift cards digital codes not returnable",
    r"store|shop|offline|showroom|pick ?up from|outlet": "where we deliver",
}
_GLOSS = [(re.compile(p, re.I), v) for p, v in GLOSSARY.items()]


def expand_query(q: str) -> str:
    extra = [v for rx, v in _GLOSS if rx.search(q)]
    return q + " " + " ".join(extra) if extra else q


def tokenize(text: str) -> list[str]:
    text = ascii_digits(text.lower())
    toks = re.findall(r"[\wऀ-ॿঀ-ൿ]+", text)
    out = []
    for t in toks:
        if t in STOP:
            continue
        # light stemming
        for suf in ("ing", "ed", "es", "s"):
            if len(t) > 4 and t.endswith(suf) and t.isascii():
                t = t[: -len(suf)]
                break
        out.append(t)
    return out


# Sections that jointly decide one answer (from the KB's own cross-references and KB-00). When the trigger
# section is among the top hits, its companions are appended so the LLM sees every rule it must combine.
COMPANIONS: list[tuple[set[str], list[str]]] = [
    ({"RET-2", "RET-3", "RET-8", "RETv3-2"}, ["RET-2", "RET-3", "RET-8"]),          # return window = base table + electronics change + festive
    ({"RET-3"}, ["RETv3-2"]),                                                           # older 10-day rule still applies to pre-Aug electronics
    ({"REF-2", "REF-1", "RET-7", "DEL-3"}, ["KB-00.3"]),                                # timelines are in business days
    ({"REF-2"}, ["KB-00.1", "REF-3"]),                                                  # FAQ-1 conflicts with REF-2 -> precedence
    ({"REF-3"}, ["REF-2"]),
    ({"DWM-1", "DWM-3", "DWM-4", "DWM-5"}, ["DWM-3", "ESC-1"]),                         # high-value / safety -> human
    ({"WAR-1", "WAR-2", "MKS-3"}, ["WAR-1", "WAR-2"]),
    ({"CAN-1", "CAN-2", "FAQH-1"}, ["CAN-1", "CAN-2"]),
    ({"DEL-5"}, ["FAQ-5"]),
    ({"RET-6"}, ["PLS-2", "DWM-2"]),
    ({"ESC-4", "IDV-4"}, ["ESC-4", "IDV-4"]),
    ({"DEL-4"}, ["ESC-4"]),
]


# Hindi FAQ twins: only pulled in when the customer writes in Devanagari
HINDI_TWINS = {"CAN-1": "FAQH-1", "CAN-2": "FAQH-1", "REF-3": "FAQH-2", "REF-2": "FAQH-2", "DWM-1": "FAQH-3", "DWM-2": "FAQH-3",
               "PLS-2": "FAQH-4"}


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or load_kb_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        docs = [tokenize(c.search_text) for c in self.chunks]
        self.bm25 = BM25Okapi(docs)
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.mat = self.tfidf.fit_transform([c.search_text.lower() for c in self.chunks])
        self.dense = None
        if settings.dense_model:
            self._init_dense(settings.dense_model)

    # ---------------------------------------------------------------- optional dense
    def _init_dense(self, name: str):
        try:
            from sentence_transformers import SentenceTransformer

            self.dense = SentenceTransformer(name)
            pre = "passage: " if "e5" in name else ""
            self.dense_mat = self.dense.encode([pre + c.search_text for c in self.chunks], normalize_embeddings=True)
            self._qpre = "query: " if "e5" in name else ""
        except Exception as exc:  # pragma: no cover
            print(f"[retriever] dense model unavailable ({exc}); using BM25+char n-grams only")
            self.dense = None

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int | None = None, context: str = "") -> list[Hit]:
        k = k or settings.top_k
        q_exp = expand_query(f"{query} {context}".strip())
        n = len(self.chunks)

        bm = np.asarray(self.bm25.get_scores(tokenize(q_exp)), dtype=float)
        tf = (self.mat @ self.tfidf.transform([q_exp.lower()]).T).toarray().ravel()
        rankings = [np.argsort(-bm), np.argsort(-tf)]
        if self.dense is not None:
            dn = self.dense_mat @ self.dense.encode([self._qpre + q_exp], normalize_embeddings=True)[0]
            rankings.append(np.argsort(-dn))

        rrf = np.zeros(n)
        for order in rankings:
            for r, idx in enumerate(order[:40]):
                rrf[idx] += 1.0 / (60 + r)
        # direct section-ID / SKU mentions are strong signals
        for m in re.findall(r"\b([A-Z]{2,5}v?\d?-\d(?:\.\d)?)\b", query):
            if m in self.by_id:
                rrf[self._idx(m)] += 0.05
        for sku in extract_skus(query, sku_prefixes(), set(products())):
            rrf[self._idx(f"products.csv:{sku}")] += 0.08

        weights = np.array([c.weight for c in self.chunks])
        final = rrf * weights
        order = [int(i) for i in np.argsort(-final)[:k] if final[i] > 0]
        top_ids = [self.chunks[i].id for i in order]
        # companions of the top-3 hits, appended after the ranked list (bounded)
        extra: list[int] = []
        for trig, adds in COMPANIONS:
            if trig & set(top_ids[:5]):
                for cid in adds:
                    if cid in self.by_id and cid not in top_ids:
                        i = self._idx(cid)
                        if i not in extra:
                            extra.append(i)
        if re.search(r"[\u0900-\u097F]", query):
            for tid in top_ids[:5]:
                twin = HINDI_TWINS.get(tid)
                if twin and twin in self.by_id and twin not in top_ids and self._idx(twin) not in extra:
                    extra.append(self._idx(twin))
        order = order + extra[: settings.max_companions]
        top = float(final[order[0]]) if order else 0.0
        return [Hit(self.chunks[i], float(final[i]) if final[i] > 0 else top * 0.5, r) for r, i in enumerate(order)]

    def _idx(self, cid: str) -> int:
        return self.chunks.index(self.by_id[cid])
