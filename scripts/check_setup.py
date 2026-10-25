"""First thing to run after filling .env:  python scripts/check_setup.py
Checks: dataset found + valid, LLM reachable (tool calling works), Gnani TTS -> STT round trip."""
import subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceops.config import find_dataset_dir, settings  # noqa: E402


def step(name, fn):
    try:
        out = fn()
        print(f"[ OK ] {name}" + (f": {out}" if out else ""))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return False


def dataset():
    d = find_dataset_dir()
    r = subprocess.run([sys.executable, "-I", "validate_dataset.py"], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    return str(d)


def llm():
    from voiceops.llm import Tool, get_llm

    l = get_llm()
    calls = []

    def handler(name, args):
        calls.append((name, args))
        return {"result": "ok"}

    r = l.complete("Reply briefly. You MUST call the ping tool once first.", [{"role": "user", "content": "ping please"}],
                   tools=[Tool("ping", "connectivity test", {"type": "object", "properties": {"x": {"type": "string"}}})], tool_handler=handler, max_tokens=600)
    if not calls:
        raise RuntimeError("model replied but did not call the tool: choose a model with tool/function-calling support")
    return f"{settings.llm_provider}/{settings.llm_model} replied {r.text[:40]!r}, tool calling works"


def gnani():
    from voiceops.voice import GnaniVoice

    v = GnaniVoice()
    wav = Path(tempfile.mkdtemp()) / "check.wav"
    t = v.synthesize("Hello, your order number is 8978 2837.", wav)
    if not t.path:
        raise RuntimeError(f"TTS failed: {t.error}")
    s = v.transcribe(wav, ["en-IN"])
    if s.error or not s.text:
        raise RuntimeError(f"STT failed: {s.error or 'empty transcript'}")
    return f"TTS voice={t.voice}, STT heard {s.text!r}"


if __name__ == "__main__":
    ok = [step("dataset", dataset), step("LLM", llm), step("Gnani TTS->STT", gnani)]
    sys.exit(0 if all(ok) else 1)
