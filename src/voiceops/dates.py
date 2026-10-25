"""Date arithmetic for support timelines (KB-00.3): business days = Mon-Sat minus holidays.csv.

Exposed to the LLM as the `date_calc` tool because language models are unreliable at calendar maths.
Return windows use calendar days, refund/logistics timelines use business days.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from functools import lru_cache

from .config import TODAY, ensure_dataset_on_path


@lru_cache(maxsize=1)
def holidays() -> frozenset[date]:
    d = ensure_dataset_on_path()
    with open(d / "data" / "holidays.csv", encoding="utf-8", newline="") as fh:
        return frozenset(date.fromisoformat(r["date"]) for r in csv.DictReader(fh))


def parse(s: str) -> date:
    s = str(s).strip()
    if s.lower() in ("today", "now"):
        return date.fromisoformat(TODAY)
    return datetime.fromisoformat(s[:10]).date()


def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d not in holidays()  # Sunday = 6


def add_business_days(start: date, n: int) -> date:
    """The date that is `n` business days after `start` (start itself is day 0)."""
    cur, step = start, 1 if n >= 0 else -1
    left = abs(n)
    while left:
        cur += timedelta(days=step)
        if is_business_day(cur):
            left -= 1
    return cur


def business_days_between(a: date, b: date) -> int:
    """Number of business days in the half-open range (a, b]: how many business days after `a` have passed by `b`."""
    if b < a:
        return -business_days_between(b, a)
    n, cur = 0, a
    while cur < b:
        cur += timedelta(days=1)
        if is_business_day(cur):
            n += 1
    return n


def fmt(d: date) -> str:
    return f"{d.isoformat()} ({d.strftime('%A')}, {d.day} {d.strftime('%B %Y')})"


def date_calc(op: str, date1: str = "today", date2: str | None = None, days: int | None = None) -> dict:
    """Tool entry point. ops: add_calendar_days, add_business_days, business_days_between,
    calendar_days_between, weekday, is_business_day."""
    try:
        d1 = parse(date1)
        if op == "add_calendar_days":
            r = d1 + timedelta(days=int(days))
            return {"result": fmt(r)}
        if op == "add_business_days":
            r = add_business_days(d1, int(days))
            return {"result": fmt(r), "note": "start date counted as day 0; Sundays and holidays skipped"}
        if op == "business_days_between":
            return {"business_days": business_days_between(d1, parse(date2)),
                    "note": "business days after date1 up to and including date2"}
        if op == "calendar_days_between":
            return {"calendar_days": (parse(date2) - d1).days}
        if op == "weekday":
            return {"result": fmt(d1)}
        if op == "is_business_day":
            return {"is_business_day": is_business_day(d1), "date": fmt(d1)}
        return {"error": f"unknown op {op}"}
    except Exception as exc:  # bad input from the model
        return {"error": f"{type(exc).__name__}: {exc}"}
