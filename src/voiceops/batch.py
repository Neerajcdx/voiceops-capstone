"""Batch runner: a folder of customer audio files -> agent replies (text + audio) + the document chunks used.

Grouping into conversations (so follow-ups work across files):
  * sub-folder          : every sub-folder is one conversation, files in natural-sort order
  * name pattern        : PC002_t1.wav, PC002_t2.wav ... (marker `t` / `turn` + number) -> conversation PC002
  * everything else     : each file is an independent single-turn conversation
Override with --mode single|conversation.
"""
from __future__ import annotations

import csv
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import SupportAgent
from .voice import REPROMPT, SUPPORTED_EXT, GnaniVoice, SttResult, silent_wav
from .textnorm import detect_script_language

_TURN_RE = re.compile(r"^(?P<conv>.+?)[_\-\s](?:t|turn)[_\-]?(?P<turn>\d+)$", re.I)


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def discover(input_dir: Path, mode: str = "auto", include_text: bool = False) -> list[tuple[str, list[tuple[Path, str]]]]:
    """-> [(conversation_id, [(file, output_stem), ...]), ...]"""
    exts = set(SUPPORTED_EXT) | ({".txt"} if include_text else set())
    files = sorted((p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts), key=natural_key)
    groups: dict[str, list[tuple[Path, str]]] = {}
    order: list[str] = []
    for f in files:
        rel = f.relative_to(input_dir)
        m = _TURN_RE.match(f.stem)
        if mode == "single":
            key = str(rel.with_suffix(""))
        elif len(rel.parts) > 1:
            key = rel.parts[0] if mode in ("auto", "conversation") else str(rel.with_suffix(""))
        elif m and mode in ("auto", "conversation"):
            key = m.group("conv")
        else:
            key = f.stem if mode != "conversation" else input_dir.name
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((f, re.sub(r"[^\w\-.]+", "_", "__".join(rel.with_suffix("").parts))))
    return [(k, groups[k]) for k in order]


def _write_result(out: Path, stem: str, rec: dict) -> None:
    (out / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{stem}.reply.txt").write_text(rec["reply_text"], encoding="utf-8")


def process_conversation(agent: SupportAgent, voice: GnaniVoice | None, conv_id: str, files: list[tuple[Path, str]], out: Path,
                         channel: str = "voice", tts: bool = True, force: bool = False) -> list[dict]:
    session = agent.new_session(conv_id, channel)
    records = []
    last_lang = "en-IN"
    for turn, (f, stem) in enumerate(files, 1):
        done = out / f"{stem}.json"
        if done.exists() and not force:
            records.append(json.loads(done.read_text(encoding="utf-8")))
            continue
        rec = {"file": str(f), "conversation_id": conv_id, "turn": turn, "heard": "", "alternatives": [], "stt_error": "",
               "reply_text": "", "reply_audio": "", "tts_error": "", "sources": [], "chunks": [], "tool_calls": [], "ticket_id": None,
               "decision_hint": "", "latency_s": {}, "error": ""}
        t0 = time.time()
        try:
            # ---- ears
            if f.suffix.lower() == ".txt":
                stt = SttResult(text=f.read_text(encoding="utf-8").strip())
            else:
                stt = voice.transcribe(f) if voice else SttResult(error="voice disabled")
            rec.update(heard=stt.text, alternatives=stt.alternatives, stt_error=stt.error)
            rec["latency_s"]["stt"] = stt.seconds
            # ---- brain
            if not stt.text.strip() or len(re.sub(r"\W", "", stt.text)) < 2:
                reply = REPROMPT.get(last_lang if last_lang in REPROMPT else "en-IN")  # recover from unclear audio
                rec["decision_hint"] = "reprompt_unclear_audio"
            else:
                r = agent.respond(session, stt.text, channel=channel, alternatives=stt.alternatives)
                reply = r.reply
                rec.update(sources=r.sources, chunks=r.chunks, tool_calls=r.tool_calls, ticket_id=r.ticket_id,
                           error=r.error)
                rec["rewritten_query"] = r.rewritten_query
                rec["latency_s"]["agent"] = r.latency_s
                rec["guard"] = r.guard
                last_lang = detect_script_language(reply)
            rec["reply_text"] = reply
            # ---- voice
            wav = out / f"{stem}.reply.wav"
            if tts and voice:
                tr = voice.synthesize(reply, wav, language=None)
                rec["tts_error"] = tr.error
                rec["spoken_text"] = tr.spoken_text
                rec["tts_voice"], rec["tts_language"] = tr.voice, tr.language
                rec["reply_audio"] = tr.path or silent_wav(wav)
            elif tts:
                rec["reply_audio"] = silent_wav(wav)
        except Exception as exc:  # noqa: BLE001 - one bad file must never stop the batch
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["trace"] = traceback.format_exc(limit=3)
            if not rec["reply_text"]:
                rec["reply_text"] = REPROMPT["en-IN"]
        rec["latency_s"]["total"] = round(time.time() - t0, 2)
        _write_result(out, stem, rec)
        records.append(rec)
    return records


def run_batch(input_dir: str | Path, out_dir: str | Path, mode: str = "auto", workers: int = 1, tts: bool = True,
              use_voice: bool = True, force: bool = False, channel: str = "voice", agent=None, voice=None) -> list[dict]:
    input_dir, out = Path(input_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agent = agent or SupportAgent()
    voice = voice if voice is not None else (GnaniVoice() if use_voice else None)
    groups = discover(input_dir, mode, include_text=not use_voice)
    if not groups:
        print(f"No audio files ({', '.join(sorted(SUPPORTED_EXT))}) found in {input_dir}")
    all_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(process_conversation, agent, voice, cid, files, out, channel, tts, force) for cid, files in groups]
        for fu in futs:
            all_records += fu.result()
    # aggregated outputs
    with open(out / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "conversation_id", "turn", "heard", "reply_text", "reply_audio", "sources", "ticket_id", "error"])
        for r in all_records:
            w.writerow([r["file"], r["conversation_id"], r["turn"], r["heard"], r["reply_text"], r["reply_audio"],
                        " ; ".join(r["sources"]), r["ticket_id"] or "", r["error"] or r["stt_error"] or r["tts_error"]])
    print(f"Processed {len(all_records)} file(s) in {len(groups)} conversation(s) -> {out}")
    return all_records
