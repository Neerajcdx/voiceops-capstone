"""Knowledge base: Markdown policy chunks (keyed by [SECTION-ID]) + product catalog rows + pincode lookups."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache

from .config import ensure_dataset_on_path

# precedence from KB-00.1: policy > agent guide > help > faq > marketing; superseded docs rank last
DOC_TYPE_WEIGHT = {"policy": 1.0, "agent_guide": 0.97, "meta": 0.92, "help": 0.92, "catalog": 0.95, "faq": 0.80, "marketing": 0.65}
SUPERSEDED_WEIGHT = 0.55


@dataclass
class Chunk:
    id: str                      # e.g. "RET-2" or "products.csv:ELE-0012"
    title: str
    text: str                    # text shown to the LLM (with header metadata)
    doc_id: str = ""
    doc_type: str = ""
    status: str = "current"
    effective_from: str = ""
    language: str = "en"
    source: str = ""
    search_text: str = field(default="", repr=False)

    @property
    def weight(self) -> float:
        w = DOC_TYPE_WEIGHT.get(self.doc_type, 0.9)
        return w * (SUPERSEDED_WEIGHT if self.status == "superseded" else 1.0)

    def render(self) -> str:
        return self.text


_FM = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_SEC = re.compile(r"^##\s+\[([^\]]+)\]\s*(.*)$", re.M)


def _front_matter(md: str) -> tuple[dict, str]:
    m = _FM.match(md)
    meta = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        md = md[m.end():]
    return meta, md


def load_kb_chunks() -> list[Chunk]:
    d = ensure_dataset_on_path()
    chunks: list[Chunk] = []
    for path in sorted((d / "kb").glob("*.md")):
        meta, body = _front_matter(path.read_text(encoding="utf-8"))
        secs = list(_SEC.finditer(body))
        if not secs:
            continue
        preamble = body[: secs[0].start()]
        preamble = re.sub(r"^#\s+.*$", "", preamble, flags=re.M).strip()  # drop H1, keep notes (e.g. v3 banner)
        for i, m in enumerate(secs):
            end = secs[i + 1].start() if i + 1 < len(secs) else len(body)
            sec_id, title = m.group(1).strip(), m.group(2).strip()
            content = body[m.end():end].strip()
            status = meta.get("status", "current")
            header = (f"[{sec_id}] {title}\n"
                      f"(doc: {meta.get('title', path.stem)} | doc_id: {meta.get('doc_id', '')} | type: {meta.get('doc_type', '')} | "
                      f"status: {status} | effective_from: {meta.get('effective_from', '')})")
            if status == "superseded":
                header += "\nWARNING: SUPERSEDED document, see note below for the only cases where it still applies."
            note = f"\nNote: {preamble}" if preamble else ""
            text = f"{header}{note}\n{content}"
            chunks.append(Chunk(
                id=sec_id, title=title, text=text, doc_id=meta.get("doc_id", ""), doc_type=meta.get("doc_type", ""),
                status=status, effective_from=meta.get("effective_from", ""), language=meta.get("language", "en"),
                source=path.name, search_text=f"{sec_id} {title} {meta.get('title', '')} {content} {preamble}",
            ))
    chunks += _product_chunks()
    return chunks


@lru_cache(maxsize=1)
def products() -> dict[str, dict]:
    d = ensure_dataset_on_path()
    with open(d / "data" / "products.csv", encoding="utf-8", newline="") as fh:
        return {r["sku"]: r for r in csv.DictReader(fh)}


@lru_cache(maxsize=1)
def pincodes() -> dict[str, dict]:
    d = ensure_dataset_on_path()
    with open(d / "data" / "pincodes.csv", encoding="utf-8", newline="") as fh:
        return {r["pincode"]: r for r in csv.DictReader(fh)}


def sku_prefixes() -> list[str]:
    return sorted({s.split("-")[0] for s in products()})


def _product_chunks() -> list[Chunk]:
    out = []
    for sku, r in products().items():
        txt = (f"[products.csv:{sku}] {r['title']}\n"
               f"(catalog row | sku: {sku} | brand: {r['brand']} | return_category: {r['return_category']} | price_inr: {r['price_inr']} | "
               f"fulfilled_by: {r['fulfilled_by']} | assured: {r['assured']} | warranty_months: {r['warranty_months']})\n"
               f"{r['description']}")
        out.append(Chunk(id=f"products.csv:{sku}", title=r["title"], text=txt, doc_id="products.csv", doc_type="catalog",
                         source="products.csv",
                         search_text=f"{sku} {r['title']} {r['brand']} {r['return_category']} {r['fulfilled_by']} {r['description']}"))
    return out


def find_products(query: str, limit: int = 5) -> list[dict]:
    """Catalog lookup by SKU or fuzzy title/brand match (used by the lookup_product tool)."""
    q = query.strip().lower()
    cat = products()
    if q.upper() in cat:
        return [cat[q.upper()]]
    qt = set(re.findall(r"[a-z0-9]+", q))
    scored = []
    for r in cat.values():
        tt = set(re.findall(r"[a-z0-9]+", f"{r['title']} {r['brand']}".lower()))
        s = len(qt & tt) / (len(qt) or 1)
        if q in r["title"].lower():
            s += 1
        if s > 0.3:
            scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:limit]]


def check_pincode(pin: str) -> dict:
    r = pincodes().get(str(pin).strip())
    if not r:
        return {"pincode": pin, "listed": False,
                "note": "Pincode is not in the serviceable pincode table, so ZentroMart does not deliver there (DEL-5)."}
    return {"listed": True, **r}
