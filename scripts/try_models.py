"""Tries a tiny question on several models and shows which ones answer (25 s limit each).
   python scripts/try_models.py            (uses LLM_API_KEY / LLM_BASE_URL from .env; never prints the key)"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceops.config import settings  # noqa: E402
from openai import OpenAI  # noqa: E402

CANDIDATES = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest",
              "gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
c = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=25, max_retries=0)
good = []
for m in CANDIDATES:
    t = time.time()
    try:
        r = c.chat.completions.create(model=m, messages=[{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=200)
        print(f"[ OK ] {m:28s} {time.time() - t:4.1f}s  -> {(r.choices[0].message.content or '').strip()[:20]!r}")
        good.append((time.time() - t, m))
    except Exception as e:
        print(f"[FAIL] {m:28s} {type(e).__name__}: {str(e)[:90]}")
if good:
    print("\nFastest working model:  LLM_MODEL=" + min(good)[1])
