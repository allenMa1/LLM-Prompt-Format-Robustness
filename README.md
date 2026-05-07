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

## Post-Inference Analysis Tools

After any inference run, score the raw outputs:

```powershell
python -m prompt_robustness.scoring --input runs/raw_outputs/<run>.jsonl --output runs/scores/<run>_scored.jsonl
```

Then generate aggregate tables and plots:

```powershell
python -m prompt_robustness.analyze --input runs/scores/<run>_scored.jsonl --out-dir runs/plots/<run>
```

The analysis step writes:

- `accuracy_by_prompt.csv`: accuracy grouped by task, model, scorer, and prompt.
- `sensitivity_by_task_model_scorer.csv`: average accuracy, min/max prompt accuracy, prompt std dev, and prompt sensitivity range.
- `artifact_gap.csv`: strict prompt sensitivity minus equivalence-aware prompt sensitivity.
- `run_health_by_prompt.csv`: completion rate, empty-output rate, and average token usage by task/model/prompt.
- `scorer_delta_by_prompt.csv`: strict vs equivalence-aware accuracy and recovery rate by prompt.
- `accuracy_by_prompt_axes.csv`: accuracy grouped by wording and output-format axes.
- `per_example_prompt_sensitivity.csv`: per-example correctness spread across prompts.
- `accuracy_<task>_<model>.png`: bar charts of prompt accuracy under strict and equivalence-aware scoring.

To summarize token usage, completion status, and timing from raw inference output:

```powershell
python -m prompt_robustness.summarize_run --input runs/raw_outputs/<run>.jsonl
```

To retry failed or empty records from a previous run using the current model config:

```powershell
python -m prompt_robustness.retry_failed --input runs/raw_outputs/<old_run>.jsonl --limit 8 --models gpt-5-nano --output runs/raw_outputs/<retry_run>.jsonl --log-file runs/logs/<retry_run>.log
```

Then score and summarize the retry output like any other run.

To merge retry outputs back into a cleaned raw-output file:

```powershell
python -m prompt_robustness.merge_retries --base runs/raw_outputs/main.jsonl --retry runs/raw_outputs/main_retry_4096.jsonl --output runs/raw_outputs/main_clean.jsonl
python -m prompt_robustness.scoring --input runs/raw_outputs/main_clean.jsonl --output runs/scores/main_clean_scored.jsonl
python -m prompt_robustness.analyze --input runs/scores/main_clean_scored.jsonl --out-dir runs/plots/main_clean
python -m prompt_robustness.summarize_run --input runs/raw_outputs/main_clean.jsonl
```

## Private Progress Notes

Completed so far:

- Repo scaffolded with config-driven tasks, prompts, models, inference, scoring, and analysis.
- Python venv set up with standard Windows Python via `py -3.10`.
- Local package installed in editable mode with `pip install -e .`.
- `.env` loading added for `OPENAI_API_KEY`; `.env` is ignored by Git.
- Dry-run fixture pipeline verified: 64 rows for `2 tasks x 2 examples x 8 prompts x 2 models`.
- TREC dataset source changed from deprecated `CogComp/trec` to `lukasgarbas/trec`.
- Real dataset loading verified for TREC-6 and GSM8K.
- Frozen main examples created under `data/frozen/main`: 100 TREC examples and 100 GSM8K examples.
- Frozen examples sanity-checked for row counts, uniqueness, valid labels, and parsed GSM8K answers.
- Prompt rendering checked on frozen examples across all 8 variants.
- Tagged/XML prompt wording fixed to use `<answer>LABEL</answer>` and `<answer>NUMBER</answer>`.
- OpenAI API smoke tests run successfully.
- `temperature` removed from GPT-5-family requests because `gpt-5-nano` rejected that parameter.
- Response metadata logging added: `response_status`, `incomplete_details`, and `usage`.
- Model config changed to provider-default reasoning with `max_output_tokens=2048`.
- Pilot run completed with 640 records under the older `1024` cap.
- Pilot analysis found 31 incomplete/empty outputs, all from `max_output_tokens` cap hits.
- Targeted retry of 8 failed nano records under `2048` completed with no empty/incomplete outputs.
- Richer analysis outputs added for run health, scorer deltas, prompt axes, and per-example sensitivity.
- Inference and retry logging added with optional `--log-file`.

Current conclusion before main:

- Matrix generation is correct.
- Scoring recomputation matches saved scorer fields.
- JSON and tagged outputs are parseable when responses complete.
- The main known pilot issue was the old `1024` cap, now fixed by `2048`.
- The project is ready for the main run.

Post-main cleanup note:

- The first main run completed 3,200 records but had 14 `gpt-5-nano` / GSM8K records hit `max_output_tokens` under the `2048` cap.
- `configs/models.json` has been raised to `4096` so those failed records can be retried without rerunning the full experiment.
- Use `retry_failed` and `merge_retries` to produce `main_clean.jsonl`.

## Operational Notes

Generated experiment artifacts are intentionally ignored by Git:

- `runs/raw_outputs/*.jsonl`
- `runs/scores/*.jsonl`
- `runs/plots/**`
- `runs/logs/*.log`
- `data/frozen/`
- `.env`

Main run command with logs:

```powershell
python -m prompt_robustness.inference --frozen-dir data/frozen/main --output runs/raw_outputs/main.jsonl --log-file runs/logs/main.log
```

Then:

```powershell
python -m prompt_robustness.scoring --input runs/raw_outputs/main.jsonl --output runs/scores/main_scored.jsonl
python -m prompt_robustness.analyze --input runs/scores/main_scored.jsonl --out-dir runs/plots/main
python -m prompt_robustness.summarize_run --input runs/raw_outputs/main.jsonl
```

Keep the laptop awake during inference. If a run is interrupted, inspect the partial JSONL before deciding whether to rerun or recover.
