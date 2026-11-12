# Telugu PromptNER

PromptNER runs few-shot, definition-guided entity extraction over the Telugu
SampurNER split saved by `download_data_telugu.py`. It does not fine-tune model
weights. The default two-stage method first predicts coarse entity spans, then
passes those proposals to a second prompt that assigns fine-grained labels. The
one-stage baseline remains available for comparison. Prompts use five labeled
training examples and English definitions for the SampurNER taxonomy.

## Configure a hosted model

Install dependencies and set `OPENAI_API_KEY` or `SARVAM_API_KEY` in `.env`.
The key is read at runtime and is never printed. Select the provider with
`PROMPTNER_PROVIDER=openai` or `PROMPTNER_PROVIDER=sarvam`; model names can be
overridden with `OPENAI_MODEL` and `SARVAM_MODEL`.

```bash
python -m pip install -r requirements-promptner.txt
python run_promptner.py --provider sarvam --method two-stage --limit 500 --seed 42
```

The CLI defaults to `sampurner_telugu_saved`, a 500-sentence deterministic test
sample, and a provider/method-specific output path under `outputs/`. It also
writes a matching `*_summary.json` with exact-span/type micro precision, recall,
F1, and failure counts. The Sarvam runner paces requests below the published Starter limit for
Sarvam-105B; set `SARVAM_RPM` if your plan has a different limit. The default
model request disables reasoning to reserve completion tokens for structured
JSON output.

Resume only failed API requests from an existing matching evaluation:

```bash
python run_promptner.py --provider sarvam --method two-stage --limit 500 --seed 42 --resume
```

Use `--workers` to control concurrent requests. Keep this compatible with your
provider's rate limits. Each result line stores the sampled sentence, gold spans,
predicted spans, raw model response, and any API or parsing errors.
