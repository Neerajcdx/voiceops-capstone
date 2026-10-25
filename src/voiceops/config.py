"""Central configuration: environment variables, dataset discovery, constants."""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:  # .env is optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]

# The dataset's fixed "today" (README.md / KB-00.2). Never use the machine clock.
TODAY = "2026-10-20"

DATASET_NAME = "Capstone_Project_Dataset_FTTT_2026"


def find_dataset_dir() -> Path:
    """Locate the extracted dataset (folder containing kb/, data/, tools/)."""
    candidates = []
    if os.getenv("DATASET_DIR"):
        candidates.append(Path(os.environ["DATASET_DIR"]).expanduser())
    cwd = Path.cwd()
    for base in (cwd, REPO_ROOT, REPO_ROOT.parent):
        candidates += [base / DATASET_NAME, base / "dataset", base / "Capstone_Project_Dataset"]
    for c in candidates:
        if (c / "kb").is_dir() and (c / "data").is_dir() and (c / "tools").is_dir():
            return c.resolve()
        # tolerate one extra nesting level (zip extracted into a same-named folder)
        if (c / DATASET_NAME / "kb").is_dir():
            return (c / DATASET_NAME).resolve()
    raise FileNotFoundError(
        "Could not find the ZentroMart dataset. Extract Capstone_Project_Dataset_FTTT_2026.zip next to "
        "this repo (or anywhere) and set DATASET_DIR=/path/to/Capstone_Project_Dataset_FTTT_2026 "
        "in your environment / .env file."
    )


class Settings:
    """Lazy settings object so importing the package never fails without keys."""

    @property
    def dataset_dir(self) -> Path:
        return find_dataset_dir()

    # ---- LLM
    llm_provider = property(lambda s: os.getenv("LLM_PROVIDER", "openai").lower())
    llm_model = property(lambda s: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_api_key = property(lambda s: os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    llm_base_url = property(lambda s: os.getenv("LLM_BASE_URL") or None)

    # ---- Retrieval
    top_k = property(lambda s: int(os.getenv("RETRIEVAL_TOP_K", "6")))
    max_companions = property(lambda s: int(os.getenv("MAX_COMPANIONS", "4")))
    dense_model = property(lambda s: os.getenv("DENSE_MODEL", ""))  # e.g. intfloat/multilingual-e5-small
    query_rewrite = property(lambda s: os.getenv("QUERY_REWRITE", "auto").lower())  # auto | always | never

    # ---- Voice
    gnani_api_key = property(lambda s: os.getenv("GNANI_API_KEY", ""))
    stt_languages = property(lambda s: [x.strip() for x in os.getenv("STT_LANGUAGES", "en-IN,hi-IN").split(",") if x.strip()])
    tts_model = property(lambda s: os.getenv("TTS_MODEL", "timbre-v2.5"))
    tts_voice_en = property(lambda s: os.getenv("TTS_VOICE_EN", "Kaveri"))
    tts_voice_hi = property(lambda s: os.getenv("TTS_VOICE_HI", "Nalini"))
    tts_voice_hinglish = property(lambda s: os.getenv("TTS_VOICE_HINGLISH", "Poorvi"))


settings = Settings()


def ensure_dataset_on_path() -> Path:
    """Put the dataset folder on sys.path so `from tools import lookup_order, create_ticket` works."""
    d = find_dataset_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return d
