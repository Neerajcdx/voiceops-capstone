import json, wave
from pathlib import Path
import pytest
from fake_llm import FakeLLM
from voiceops.agent import SupportAgent
from voiceops.batch import discover, run_batch
from voiceops.config import ensure_dataset_on_path
from voiceops.voice import SttResult, TtsResult, prepare_for_speech, silent_wav, split_wav, wav_duration


class FakeVoice:
    """Stands in for GnaniVoice: transcript comes from a sidecar .txt, TTS writes a silent wav."""
    def transcribe(self, path):
        side = Path(path).with_suffix(".txt")
        if not side.exists():
            return SttResult(error="boom")
        txt = side.read_text(encoding="utf-8")
        if txt.startswith("ALT|"):
            return SttResult(text="garbled", alternatives=[txt[4:]])
        return SttResult(text=txt, seconds=0.1)

    def synthesize(self, text, out, language=None):
        silent_wav(out)
        return TtsResult(path=str(out), spoken_text=prepare_for_speech(text), language="en-IN", voice="Kaveri")


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    ensure_dataset_on_path()
    import tools.tools as t
    monkeypatch.setattr(t, "_TICKETS", str(tmp_path / "tickets.csv"))
    return SupportAgent(llm=FakeLLM())


def mk(d: Path, name, text=None):
    silent_wav(d / f"{name}.wav")
    if text is not None:
        (d / f"{name}.txt").write_text(text, encoding="utf-8")


def test_grouping(tmp_path):
    for n in ["PC001_t1", "PC001_t2", "q_07", "q_08"]:
        mk(tmp_path, n)
    (tmp_path / "convA").mkdir(); mk(tmp_path / "convA", "1"); mk(tmp_path / "convA", "2")
    g = {k: [s for _, s in v] for k, v in discover(tmp_path)}
    assert g["PC001"] == ["PC001_t1", "PC001_t2"] and g["convA"] == ["convA__1", "convA__2"]
    assert "q_07" in g and "q_08" in g           # no turn marker -> independent files


def test_batch_end_to_end_with_failures(tmp_path, agent):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    mk(inp, "PC002_t1", "Hello, I need help with an order")
    mk(inp, "PC002_t2", "the order number is eight nine seven eight two eight three seven")
    mk(inp, "PC002_t3", "ALT|last four digits four five three three")   # primary STT garbled, alt hypothesis has the digits
    mk(inp, "silent_one")                                                # no transcript -> STT error -> polite re-prompt
    recs = run_batch(inp, out, agent=agent, voice=FakeVoice())
    by = {r["file"].split("/")[-1]: r for r in recs}
    assert len(recs) == 4 and all(r["reply_text"] for r in recs)
    assert (out / "PC002_t3.reply.wav").exists() and (out / "results.csv").exists() and (out / "results.jsonl").exists()
    assert "STY-1" in by["PC002_t1.wav"]["sources"]
    assert by["PC002_t2.wav"]["tool_calls"][0]["name"] == "lookup_order"
    assert by["PC002_t3.wav"]["tool_calls"][0]["args"]["verification_value"] == "4533"   # recovered from alternative hypothesis
    assert "catch that" in by["silent_one.wav"]["reply_text"] and by["silent_one.wav"]["stt_error"]
    # resume: second run reuses results
    recs2 = run_batch(inp, out, agent=agent, voice=FakeVoice())
    assert len(recs2) == 4


def test_speech_normalisation():
    s = prepare_for_speech("**Refund** by 2026-10-27. Fee is ₹49 for order 89782837, total Rs. 10,000!")
    assert "27 October 2026" in s and "49 rupees" in s and "8978 2837" in s and "10,000 rupees" in s and "**" not in s
    assert prepare_for_speech("आपका ₹49 लगेगा").startswith("आपका 49 रुपये")


def test_split_long_wav(tmp_path):
    p = tmp_path / "long.wav"
    silent_wav(p, seconds=130, rate=8000)
    parts = split_wav(p)
    assert len(parts) == 3 and all(wav_duration(x) <= 55.5 for x in parts)
