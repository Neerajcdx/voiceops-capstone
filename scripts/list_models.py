"""Shows which model names your LLM key can use.   python scripts/list_models.py
Reads LLM_API_KEY / LLM_BASE_URL from .env. Never prints the key."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceops.config import settings  # noqa: E402  (loads .env)
import requests  # noqa: E402

base = (settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")
print("Asking", base + "/models", "...")
t = time.time()
try:
    r = requests.get(base + "/models", headers={"Authorization": f"Bearer {settings.llm_api_key}"}, timeout=30)
except Exception as e:
    sys.exit(f"Could not reach the server in 30 s: {type(e).__name__}. Check your internet / VPN / firewall and try again.")
print(f"HTTP {r.status_code} in {time.time() - t:.1f}s")
if r.status_code != 200:
    sys.exit("Server said: " + r.text[:300] + "\n(401/403 = wrong or disabled key)")
ids = sorted(m.get("id", "").replace("models/", "") for m in r.json().get("data", []))
print("\nModel names you can put after LLM_MODEL=  (flash ones are fastest):")
for i in ids:
    if "gemini" in i or "gpt" in i or "claude" in i or "llama" in i:
        print("  ", i)
