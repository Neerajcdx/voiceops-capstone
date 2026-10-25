"""Voice layer: Gnani STT (ears) and Gnani TTS (voice) around the text agent.

Only Gnani is used for speech (competition rule). Design goals:
  * never crash on bad audio / API hiccups: every failure turns into a polite spoken re-prompt
  * the customer's language is unknown up front, so STT runs in every language of STT_LANGUAGES (default en-IN + hi-IN,
    in parallel) and the agent sees the extra hypothesis when it differs
  * long clips (> 60 s REST limit) are split into chunks
  * replies are rewritten "for the ear" (dates with year, amounts in rupees, order IDs in groups of four)
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .config import settings
from .textnorm import detect_script_language

SUPPORTED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
MAX_CLIP_S = 55.0

REPROMPT = {
    "en-IN": "Sorry, I didn't catch that. Could you say it again?",
    "hi-en": "Sorry, mujhe sunai nahi diya. Kya aap dobara bol sakte hain?",
    "hi-IN": "माफ़ कीजिए, मैं सुन नहीं पाया। क्या आप दोबारा बोल सकते हैं?",
}

# per-language default voices for timbre-v2.5 (first = preferred)
_VOICES = {
    "en-IN": lambda: [settings.tts_voice_en, "Kaveri", "Pranav"],
    "hi-IN": lambda: [settings.tts_voice_hi, "Nalini", "Deepak"],
    "hi-en": lambda: [settings.tts_voice_hinglish, "Poorvi", "Nalini"],
    "ta-IN": lambda: ["Asmita"], "te-IN": lambda: ["Suhana"], "kn-IN": lambda: ["Saanvi"], "ml-IN": lambda: ["Reshma"],
    "mr-IN": lambda: ["Zahira"], "bn-IN": lambda: ["Kirra"], "gu-IN": lambda: ["Falak"], "pa-IN": lambda: ["Mehuli"],
}


@dataclass
class SttResult:
    text: str = ""                                   # primary transcript passed to the agent
    alternatives: list[str] = field(default_factory=list)   # other-language hypotheses (only if different)
    hypotheses: dict[str, str] = field(default_factory=dict)
    error: str = ""
    seconds: float = 0.0


@dataclass
class TtsResult:
    path: str = ""
    spoken_text: str = ""
    language: str = ""
    voice: str = ""
    error: str = ""


# ------------------------------------------------------------------------------------ audio helpers
def _real_frames(w: "wave.Wave_read", path: Path) -> int:
    """Frames actually present. Streamed TTS files carry a bogus 0xFFFFFFFF length in the header, so
    never trust getnframes() alone: cap it by what the file size can hold."""
    framesize = max(1, w.getnchannels() * w.getsampwidth())
    by_size = max(0, (path.stat().st_size - 44) // framesize)
    return min(w.getnframes(), by_size)


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            return _real_frames(w, path) / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return None


def split_wav(path: Path, max_s: float = MAX_CLIP_S) -> list[Path]:
    """Split a long PCM wav into <= max_s chunks (STT REST accepts up to 60 s)."""
    dur = wav_duration(path)
    if dur is None or dur <= max_s:
        return [path]
    out = []
    with wave.open(str(path), "rb") as w:
        nch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        step = int(max_s * rate)
        i = 0
        while True:
            frames = w.readframes(step)
            if not frames:
                break
            p = Path(tempfile.mkdtemp()) / f"{path.stem}_part{i}.wav"
            with wave.open(str(p), "wb") as o:
                o.setnchannels(nch)       # do not copy nframes from the (possibly bogus) header
                o.setsampwidth(width)
                o.setframerate(rate)
                o.writeframes(frames)
            out.append(p)
            i += 1
    return out


def ensure_supported(path: Path) -> Path:
    """Browsers record webm/opus etc.: convert to 16 kHz mono wav with ffmpeg when needed."""
    if path.suffix.lower() in SUPPORTED_EXT:
        return path
    if not shutil.which("ffmpeg"):
        raise ValueError(f"Unsupported audio format {path.suffix} and ffmpeg is not installed")
    dst = Path(tempfile.mkdtemp()) / (path.stem + ".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ar", "16000", "-ac", "1", str(dst)], check=True)
    return dst


# ------------------------------------------------------------------------------------ speech text
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def prepare_for_speech(text: str) -> str:
    """Rewrite a chat reply for the ear (Gnani text-normalisation guide + STY-3)."""
    t = text
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\*\*|__|`|#+\s*", "", t)
    t = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", t)
    hindi = bool(re.search(r"[ऀ-ॿ]", t))
    rupee_word = "रुपये" if hindi else "rupees"

    # ISO dates -> "27 October 2026"
    def iso(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            date(y, mo, d)
        except ValueError:
            return m.group(0)
        return f"{d} {_MONTHS[mo - 1]} {y}"

    t = re.sub(r"\b(20\d{2})-(\d{2})-(\d{2})\b(?:T\d{2}:\d{2})?", iso, t)
    # amounts
    t = re.sub(r"(?:₹|Rs\.?|INR)\s?([\d,]+(?:\.\d+)?)", lambda m: f"{m.group(1)} {rupee_word}", t)
    t = re.sub(r"([\d,]+(?:\.\d+)?)\s?(?:₹|/-)", lambda m: f"{m.group(1)} {rupee_word}", t)
    # 8-digit order IDs -> groups of four ("8978 2837")
    t = re.sub(r"(?<![\d.,-])(\d{8})(?!\d)(?![.,]\d)", lambda m: f"{m.group(1)[:4]} {m.group(1)[4:]}", t)
    t = re.sub(r"\s+([,.!?।])", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    if t and t[-1] not in ".!?।":
        t += "."
    return t


# ------------------------------------------------------------------------------------ Gnani clients
class GnaniVoice:
    def __init__(self):
        if not settings.gnani_api_key:
            raise RuntimeError("GNANI_API_KEY is not set. Create a key at https://app.gnani.ai/voice and put it in .env")
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient

        self.stt = GnaniSTTClient(api_key=settings.gnani_api_key)
        self.tts = GnaniTTSClient(api_key=settings.gnani_api_key)

    # ------------------------------------------------------------- STT
    def _transcribe_one(self, path: Path, lang: str) -> str:
        parts = []
        for part in split_wav(path):
            last = None
            for attempt in range(2):
                try:
                    r = self.stt.transcribe(str(part), language_code=lang)
                    parts.append((r.get("transcript") or "").strip())
                    last = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last = exc
            if last is not None:
                raise last
        return " ".join(p for p in parts if p).strip()

    def transcribe(self, audio_path: str | Path, languages: list[str] | None = None) -> SttResult:
        import time

        t0 = time.time()
        res = SttResult()
        languages = languages or settings.stt_languages
        try:
            path = ensure_supported(Path(audio_path))
        except Exception as exc:  # noqa: BLE001
            res.error = f"audio: {exc}"
            return res

        def run(lang):
            try:
                return lang, self._transcribe_one(path, lang), ""
            except Exception as exc:  # noqa: BLE001
                return lang, "", f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=len(languages)) as ex:
            outs = list(ex.map(run, languages))
        errors = [e for _, t, e in outs if e]
        for lang, txt, _ in outs:
            if txt:
                res.hypotheses[lang] = txt
        if res.hypotheses:
            primary_lang = next(l for l in languages if l in res.hypotheses)
            res.text = res.hypotheses[primary_lang]
            norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
            for l, t in res.hypotheses.items():
                if l != primary_lang and norm(t) != norm(res.text):
                    res.alternatives.append(t)
        elif errors:
            res.error = "; ".join(errors)
        res.seconds = round(time.time() - t0, 2)
        return res

    # ------------------------------------------------------------- TTS
    def synthesize(self, text: str, out_path: str | Path, language: str | None = None) -> TtsResult:
        spoken = prepare_for_speech(text)
        lang = language or detect_script_language(spoken)
        res = TtsResult(spoken_text=spoken, language=lang)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        voices = _VOICES.get(lang, _VOICES["en-IN"])()
        tts_langs = [lang] + (["hi-IN"] if lang == "hi-en" else []) + (["auto"] if lang not in ("en-IN",) else [])
        last_err = ""
        for model in dict.fromkeys([settings.tts_model, "timbre-v2.5"]):
            for tl in tts_langs:
                for voice in dict.fromkeys(voices):
                    try:
                        self.tts.synthesize(spoken, voice=voice, model=model, language=tl, output_file=str(out_path))
                        if out_path.exists() and out_path.stat().st_size > 0:
                            res.path, res.voice = str(out_path), voice
                            return res
                    except Exception as exc:  # noqa: BLE001
                        last_err = f"{model}/{tl}/{voice}: {type(exc).__name__}: {exc}"
        res.error = last_err or "TTS produced no audio"
        return res


def silent_wav(path: str | Path, seconds: float = 0.5, rate: int = 16000) -> str:
    """Placeholder audio so a batch run always has an output file even if TTS is unreachable."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return str(path)
