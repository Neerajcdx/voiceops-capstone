# sample_outputs

Put the output of a real batch run here before submitting:

```bash
python scripts/make_test_audio.py --out test_audio --limit 8          # or record your own voice into test_audio/
python run.py test_audio --out sample_outputs/run1
```

Each audio file produces `<name>.json` (heard text, reply, **retrieved chunk IDs**, tool calls, latencies),
`<name>.reply.txt`, `<name>.reply.wav`, plus `results.csv` / `results.jsonl` for the whole folder.

`retrieval_eval_offline.json` is already real: recall of the gold evidence by the retriever alone (no LLM, no keys needed),
produced by `python eval/retrieval_eval.py`.
