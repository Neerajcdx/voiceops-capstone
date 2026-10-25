# ZentroMart VoiceOps: a support agent you can phone

LLM + RAG + tools + Gnani speech for the fictional store **ZentroMart** (VoiceOps 2026 capstone).
It answers from the company documents, holds a multi-turn conversation, verifies the caller before sharing order
details, hands over to a human when policy says so, and listens and speaks through **Gnani STT/TTS**.

```
customer audio ─► Gnani STT ─► agent (LLM + RAG + tools) ─► Gnani TTS ─► spoken reply
                                  ├─ retrieval over kb/*.md chunks (section ids kept)
                                  ├─ lookup_product / date_calc (business days) / search_kb
                                  └─ lookup_order (identity-checked) · create_ticket   (tools/tools.py from the dataset)
```

## 1. Setup

```bash
python -m venv .venv                                        # Python 3.10+
source .venv/bin/activate                                   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                        # Windows: copy .env.example .env  (then fill in the keys; never commit .env)
unzip Capstone_Project_Dataset_FTTT_2026.zip                # dataset is NOT part of this repo; set DATASET_DIR in .env to its full path
python scripts/check_setup.py                               # dataset valid? LLM reachable? Gnani TTS->STT round trip?
```

Needs: a Gnani API key (https://app.gnani.ai/voice) and one LLM key. Tested with Gemini 3.5 Flash and Gemini 3.1 Flash Lite
through Google's OpenAI-compatible endpoint (see `.env.example`). The Gemini free tier allows only about 20 requests per day per
model, so for the full evaluation use a model with a higher limit or turn on billing.

LLM: any hosted or open-source model through `LLM_PROVIDER=anthropic|openai` (`openai` = any OpenAI-compatible
endpoint: OpenAI, Gemini, Groq, OpenRouter, Ollama, vLLM). See `.env.example` for ready-made examples.
Speech: **Gnani only** (STT and TTS), model named explicitly (`timbre-v2.5`).

## 2. Run command (batch: folder of audio in, replies out)

```bash
python run.py ./test_audio --out ./out        # same thing: ./run.sh ./test_audio --out ./out
```

For every audio file (`.wav .mp3 .flac .ogg .m4a .aac`) it runs STT → agent → TTS and saves in `--out`:

* `<name>.reply.txt`: the reply text · `<name>.reply.wav`: the spoken reply
* `<name>.json`: what was heard, reply, **document chunks retrieved / section ids used**, tool calls, ticket id, latencies
* `results.csv` and `results.jsonl` for the whole folder

Conversations: sub-folders, or files named `PC002_t1.wav`, `PC002_t2.wav`… are played as one conversation in turn
order (`--mode auto`); everything else is one file = one caller. Other flags: `--mode single|conversation`,
`--workers N`, `--no-tts`, `--force` (re-run finished files; otherwise results are resumed), `--text-only`
(developer mode, no Gnani: `.txt` files are used as transcripts). A bad file never stops the batch; unclear audio gets a polite
"please say that again" in the caller's language.

No recordings yet? `python scripts/make_test_audio.py --out test_audio --limit 10` synthesises practice questions
with Gnani TTS (record a few in your own voice too).

## 3. Other commands

```bash
python run.py chat                        # terminal chat (text); /sources shows chunks and tool calls
python run.py serve --port 8000           # browser demo with microphone: record, hear the reply (use it for the video)
python run.py eval --suite questions --agents baseline,rag     # Step 1 vs Step 2
python run.py eval --suite conversations --agents rag          # Step 3
python run.py eval --suite all --field asr_text                # ASR-style input
python eval/retrieval_eval.py             # offline, no keys: retrieval recall against gold_evidence
python -m pytest -q                       # 24 offline tests (fake LLM, no keys needed)
```

## 4. Design in one page

* **Retrieval** (`retriever.py`, `kb.py`): one chunk per `## [SECTION-ID]` with doc id, status and effective date;
  BM25 + character n-gram TF-IDF (+ optional dense embeddings via `DENSE_MODEL`) fused with reciprocal-rank fusion,
  then re-weighted by document precedence (policy > agent guide > help > FAQ > marketing, KB-00.1). A small glossary
  maps customer wording and Hindi/Hinglish to the documents' words; an LLM query rewrite handles follow-ups and
  other languages. Superseded documents only apply where their successor says so. Catalogue rows are evidence too
  (`products.csv:SKU`), including SKUs spoken by ASR.
* **No hardcoded answers**: prompts hold rules and procedure; facts come from retrieved chunks, the catalogue, the
  verified order record or the date tool (business days = Mon–Sat minus `holidays.csv`; return windows = calendar days).
* **Safety in code** (`session.py`, `agent.py`): caller verification runs in code (phone last-4 or pincode, one value at a
  time, one retry, then locked); verification values are scrubbed from outgoing text, tickets and logs; the delivery
  pincode is not shown to the model; a handover is only reported after `create_ticket` returned `created=True`; the first
  reply always says it is an AI assistant.
* **Voice** (`voice.py`, `textnorm.py`): clips over 60 s are split; browser audio is converted with ffmpeg if needed; unclear or
  empty audio is re-asked; replies are rewritten for the ear (dates with year, rupees spoken, IDs in groups of four);
  per-language voices with fallbacks for TTS language codes.

## 5. Repo map

```
run.py run.sh            entry points         .env.example  requirements.txt
src/voiceops/            config kb retriever textnorm dates session agent prompts llm voice batch webapp scoring
eval/                    evaluate.py  retrieval_eval.py       scripts/  check_setup.py  make_test_audio.py
tests/                   offline tests with a fake LLM         sample_outputs/  put a real batch run here
report.pdf               3-page report   demo.mp4  2-minute demo video
```

## 6. Results on the practice set (details in report.pdf)

| 52 practice questions | Step 1: prompt only | Step 2: with documents |
|---|---|---|
| Correct answers | 12 / 52 (23%) | 39 / 52 (75%) |
| Right sections found | not applicable | 100% |

Conversations: 83 / 96 turns correct (86%), 11 of 23 conversations fully correct, expected tool calls in 94 / 96 turns.
Weak spots: questions the documents cannot answer (2 / 5), and Hinglish (6 / 11).

## 7. Rules respected

Gnani is the only STT/TTS (no fallback to any other speech service). Only the synthetic dataset is used. Test answers
come from the agent, never typed or edited. No `.env`, keys, virtualenv or dataset copy in the repo.
