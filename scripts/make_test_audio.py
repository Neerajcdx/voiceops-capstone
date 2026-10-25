"""Make a folder of test audio WITHOUT recording: synthesises the practice questions / conversations with Gnani TTS.
(Gnani-only, so it respects the speech-service rule. Also record a few with your own voice for realism!)

  python scripts/make_test_audio.py --out test_audio --field asr_text --limit 10      # PQ####.wav (single questions)
  python scripts/make_test_audio.py --out test_audio --conversations                   # PC002_t1.wav ... (multi-turn)
Then:  python run.py test_audio --out out
"""
import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceops.config import ensure_dataset_on_path  # noqa: E402
from voiceops.voice import GnaniVoice  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="test_audio")
ap.add_argument("--field", default="asr_text", choices=["question", "asr_text"])
ap.add_argument("--limit", type=int, default=10)
ap.add_argument("--conversations", action="store_true")
a = ap.parse_args()
d = ensure_dataset_on_path()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
v = GnaniVoice()
if a.conversations:
    rows = list(csv.DictReader(open(d / "practice" / "conversations.csv", encoding="utf-8")))
    keep = list(dict.fromkeys(r["conversation_id"] for r in rows))[: a.limit]
    items = [(f"{r['conversation_id']}_t{r['turn']}", r["customer_message"]) for r in rows if r["conversation_id"] in keep]
else:
    rows = list(csv.DictReader(open(d / "practice" / "questions.csv", encoding="utf-8")))[: a.limit]
    items = [(r["id"], r[a.field]) for r in rows]
for name, text in items:
    r = v.synthesize(text, out / f"{name}.wav")
    print(("ok  " if r.path else "FAIL"), name, r.error or "")
