# Prompt and Format Robustness in LLM Evaluation

This repo runs a scoped empirical study of prompt/output-format sensitivity in LLM evaluation.

Core question:

> How much do semantically equivalent prompt and format changes affect measured LLM performance, and how much does that effect depend on the scoring method?

## Locked MVP

- Tasks: `trec6`, `gsm8k`
- Prompt variants: 8 total, from `2 wordings x 4 output formats`
- Scorers: strict and equivalence-aware
- Pilot: 20 examples per task
- Main run: 100 examples per task
- Initial models: `gpt-5-nano`, `gpt-5-mini`

## Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If your venv was created by MSYS/MinGW Python and has `.venv\bin` instead of `.venv\Scripts`, activate it with:

```powershell
.\.venv\bin\Activate.ps1
```

However, prefer the `py -3.10 -m venv .venv` command on Windows. MSYS/MinGW Python may try to build packages such as `pandas` and `numpy` from source instead of using normal Windows wheels.

Set your OpenAI API key before inference:

```powershell
$env:OPENAI_API_KEY="..."
```

Or create a local `.env` file in the repo root:

```text
OPENAI_API_KEY=...
```

The inference script loads `.env` automatically. `.env` is ignored by Git.

## Dry Run

This validates prompt rendering, output file structure, and scoring without API calls or dataset downloads. It uses tiny local fixtures in `data/fixtures`.

```powershell
python -m prompt_robustness.inference --limit 2 --dry-run --output runs/raw_outputs/dry_run.jsonl
python -m prompt_robustness.scoring --input runs/raw_outputs/dry_run.jsonl --output runs/scores/dry_run_scored.jsonl
```

## Pilot Run

```powershell
python -m prompt_robustness.inference --limit 20 --output runs/raw_outputs/pilot.jsonl
python -m prompt_robustness.scoring --input runs/raw_outputs/pilot.jsonl --output runs/scores/pilot_scored.jsonl
python -m prompt_robustness.analyze --input runs/scores/pilot_scored.jsonl --out-dir runs/plots/pilot
```

After the pilot, inspect failures and freeze scoring rules.

## Dataset Sources

- `trec6`: `lukasgarbas/trec`, a Parquet-converted mirror of the classic TREC question classification dataset. The original `CogComp/trec` Hugging Face repo uses a dataset script, which recent `datasets` versions no longer load by default.
- `gsm8k`: `openai/gsm8k`, subset `main`.

## Freeze Examples

Before real API runs, freeze the sampled examples so pilot/main runs use the exact same subset:

```powershell
python -m prompt_robustness.freeze_examples --limit 100 --out-dir data/frozen/main
```

Then run inference from the frozen files:

```powershell
python -m prompt_robustness.inference --frozen-dir data/frozen/main --output runs/raw_outputs/main.jsonl
```

Add `--log-file runs/logs/main.log` to save inference progress logs:

```powershell
python -m prompt_robustness.inference --frozen-dir data/frozen/main --output runs/raw_outputs/main.jsonl --log-file runs/logs/main.log
```

To retry a small subset of failed or empty records from a prior run with the current model config:

```powershell
python -m prompt_robustness.retry_failed --input runs/raw_outputs/pilot.jsonl --limit 8 --models gpt-5-nano --output runs/raw_outputs/pilot_retry_2048.jsonl
python -m prompt_robustness.scoring --input runs/raw_outputs/pilot_retry_2048.jsonl --output runs/scores/pilot_retry_2048_scored.jsonl
```

To summarize token usage, completion status, and rough timing:

```powershell
python -m prompt_robustness.summarize_run --input runs/raw_outputs/pilot.jsonl
```

## Main Run

```powershell
python -m prompt_robustness.inference --frozen-dir data/frozen/main --output runs/raw_outputs/main.jsonl
python -m prompt_robustness.scoring --input runs/raw_outputs/main.jsonl --output runs/scores/main_scored.jsonl
python -m prompt_robustness.analyze --input runs/scores/main_scored.jsonl --out-dir runs/plots/main
```

## Important Design Choice

The same raw outputs are scored twice:

```text
raw model output -> strict scorer
raw model output -> equivalence-aware scorer
```

This keeps the evaluation-method ablation clean. Existing benchmark loaders are fine; outsourcing the scoring logic is not.
