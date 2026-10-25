#!/usr/bin/env python3
"""ZentroMart VoiceOps agent: command-line entry point.

  python run.py <audio_folder> [--out out]      batch: audio in -> reply text + reply audio + retrieved chunks (the graded command)
  python run.py chat                            terminal chat with the agent (text)
  python run.py serve [--port 8000]             web demo with microphone (record, hear the reply)
  python run.py eval [eval args]                Step 1 vs Step 2 evaluation on the practice set
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def cmd_batch(a):
    from voiceops.batch import run_batch

    run_batch(a.input_dir, a.out, mode=a.mode, workers=a.workers, tts=not a.no_tts, use_voice=not a.text_only, force=a.force)


def cmd_chat(_a):
    from voiceops.agent import SupportAgent

    agent = SupportAgent()
    s = agent.new_session("cli", "text")
    print("ZentroMart support (type 'quit' to exit, '/sources' to see the last chunks)\n")
    last = None
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in ("quit", "exit"):
            break
        if msg == "/sources" and last:
            print("   sources:", last.sources, "\n   tools:", [(c["name"], c["args"]) for c in last.tool_calls])
            continue
        last = agent.respond(s, msg)
        print(f"bot> {last.reply}\n")


def cmd_serve(a):
    from voiceops.webapp import create_app

    create_app().run(host=a.host, port=a.port, debug=False, threaded=True)


def cmd_eval(rest):
    sys.exit(subprocess.call([sys.executable, str(ROOT / "eval" / "evaluate.py"), *rest]))


def batch_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description="Batch: folder of customer audio -> reply text + audio + retrieved chunks")
    p.add_argument("input_dir", help="folder with customer audio files (wav/mp3/flac/ogg/m4a/aac)")
    p.add_argument("--out", default="out", help="output folder (default: out)")
    p.add_argument("--mode", default="auto", choices=["auto", "single", "conversation"],
                   help="how files form conversations: auto = sub-folders or NAME_t1/NAME_t2 patterns, else each file alone")
    p.add_argument("--workers", type=int, default=1, help="conversations processed in parallel (mind LLM rate limits)")
    p.add_argument("--no-tts", action="store_true", help="skip speech synthesis")
    p.add_argument("--text-only", action="store_true", help="dev mode: no Gnani calls; .txt files are used as transcripts")
    p.add_argument("--force", action="store_true", help="re-run files that already have results in --out")
    return p


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    head, rest = argv[0], argv[1:]
    if head == "eval":
        return cmd_eval(rest)
    if head == "chat":
        return cmd_chat(None)
    if head == "serve":
        p = argparse.ArgumentParser(prog="run.py serve")
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000)
        return cmd_serve(p.parse_args(rest))
    if head == "batch":
        argv = rest
    cmd_batch(batch_parser().parse_args(argv))


if __name__ == "__main__":
    main()
