# Prompt and Format Robustness in LLM Evaluation

This repo contains a scoped empirical study of prompt/output-format sensitivity in LLM evaluation.

Core question:

> How much do semantically equivalent prompt and format changes affect measured LLM performance, and how much does that effect depend on the scoring method?

## Current Status

The main coding, inference, scoring, retry cleanup, aggregate analysis, and LLM-judge audit are complete.

Final clean run:

- Tasks: `trec6`, `gsm8k`
- Examples: 100 frozen examples per task
- Models: `gpt-5-nano`, `gpt-5-mini`
- Prompt variants: 8 total, from `2 wordings x 4 output formats`
- Main outputs: 3,200 / 3,200
- Noncompleted outputs after retry merge: 0
- Empty outputs after retry merge: 0
- LLM-judge audit: 326 strict-false records judged by `gpt-5.4`
- Judge parse failures: 0
- Judge/rule equivalence agreement: 100%

## Final Results

Committable final outputs live under `reports/results/`.

```text
reports/results/main_clean/
```

Aggregate CSVs and plots for the clean main run.

```text
reports/results/full_outputs/
```

Full per-record clean raw outputs and strict/equivalence-aware scored outputs.

```text
reports/results/judge_audit_gpt54/
```

LLM-judge audit summaries and full per-record judge output.

The generated `runs/` artifacts remain ignored by Git; `reports/results/` contains the curated committable copies.

## Headline Results

Prompt sensitivity is measured as:

```text
max_prompt_accuracy - min_prompt_accuracy
```

Clean main results:

```text
GSM8K / gpt-5-mini
strict: 0.03
equivalence-aware: 0.02
artifact gap: 0.01

GSM8K / gpt-5-nano
strict: 0.07
equivalence-aware: 0.06
artifact gap: 0.01

TREC-6 / gpt-5-mini
strict: 0.05
equivalence-aware: 0.05
artifact gap: 0.00

TREC-6 / gpt-5-nano
strict: 0.08
equivalence-aware: 0.08
artifact gap: 0.00
```

Interpretation:

- Prompt sensitivity is present but moderate.
- `gpt-5-nano` is more prompt-sensitive than `gpt-5-mini`.
- Equivalence-aware scoring mainly changes GSM8K results by recovering harmless numeric formatting differences.
- TREC-6 has essentially no strict-vs-equivalence scoring gap because outputs are usually valid labels or genuinely wrong labels.
- The `gpt-5.4` judge audit agrees exactly with the deterministic equivalence-aware scorer on all strict-false cases.

## Experiment Design

Tasks:

- `trec6`: TREC-6 question classification
- `gsm8k`: grade-school math short-answer reasoning

Dataset sources:

- `trec6`: `lukasgarbas/trec`, a Parquet-converted mirror of classic TREC. The original `CogComp/trec` Hugging Face repo uses a dataset script that recent `datasets` versions no longer load by default.
- `gsm8k`: `openai/gsm8k`, subset `main`.

Prompt variants:

```text
direct_answer_only
direct_sentence
direct_tagged
direct_json
formal_answer_only
formal_sentence
formal_tagged
formal_json
```

Scorers:

- Strict scorer: exact requested format and exact gold answer.
- Equivalence-aware scorer: deterministic extraction/normalization for JSON, XML-style tags, answer sentences, TREC aliases, and numeric formatting.
- LLM-judge audit: `gpt-5.4` audit over records with `strict_correct = false`.

## Setup

Use standard Windows Python, not MSYS/MinGW Python.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

If your venv was created by MSYS/MinGW Python and has `.venv\bin` instead of `.venv\Scripts`, activate it with:

```powershell
.\.venv\bin\Activate.ps1
```

However, prefer:

```powershell
py -3.10 -m venv .venv
```

Create a local `.env` file in the repo root:

```text
OPENAI_API_KEY=...
```

`.env` is ignored by Git. Inference and judge scripts load it automatically.

## Core Commands

Dry run without API calls or dataset downloads:

```powershell
python -m prompt_robustness.inference --limit 2 --dry-run --output runs/raw_outputs/dry_run.jsonl
python -m prompt_robustness.scoring --input runs/raw_outputs/dry_run.jsonl --output runs/scores/dry_run_scored.jsonl
```

Freeze examples:

```powershell
python -m prompt_robustness.freeze_examples --limit 100 --out-dir data/frozen/main
```

Main inference with logs:

```powershell
python -m prompt_robustness.inference --frozen-dir data/frozen/main --output runs/raw_outputs/main.jsonl --log-file runs/logs/main.log
```

Score and analyze:

```powershell
python -m prompt_robustness.scoring --input runs/raw_outputs/main.jsonl --output runs/scores/main_scored.jsonl
python -m prompt_robustness.analyze --input runs/scores/main_scored.jsonl --out-dir runs/plots/main
python -m prompt_robustness.summarize_run --input runs/raw_outputs/main.jsonl
```

Retry failed or empty records:

```powershell
python -m prompt_robustness.retry_failed --input runs/raw_outputs/main.jsonl --models gpt-5-nano --output runs/raw_outputs/main_retry_4096.jsonl --log-file runs/logs/main_retry_4096.log
```

Merge retries into a clean raw-output file:

```powershell
python -m prompt_robustness.merge_retries --base runs/raw_outputs/main.jsonl --retry runs/raw_outputs/main_retry_4096.jsonl --output runs/raw_outputs/main_clean.jsonl
```

Score/analyze clean outputs:

```powershell
python -m prompt_robustness.scoring --input runs/raw_outputs/main_clean.jsonl --output runs/scores/main_clean_scored_v2.jsonl
python -m prompt_robustness.analyze --input runs/scores/main_clean_scored_v2.jsonl --out-dir runs/plots/main_clean_v2
python -m prompt_robustness.summarize_run --input runs/raw_outputs/main_clean.jsonl
```

Run LLM-judge audit:

```powershell
python -m prompt_robustness.judge_equivalence --input runs/scores/main_clean_scored_v2.jsonl
```

Defaults:

- selection: `strict_false`
- judge model: `gpt-5.4`
- max output tokens: 2,048
- output: `runs/scores/main_clean_judge_audit_gpt54.jsonl`
- log file: `runs/logs/main_clean_judge_audit_gpt54.log`

Summarize judge audit:

```powershell
python -m prompt_robustness.summarize_judge_audit --input runs/scores/main_clean_judge_audit_gpt54.jsonl --out-dir runs/plots/main_clean_judge_audit_gpt54
```

## Analysis Outputs

`prompt_robustness.analyze` writes:

- `accuracy_by_prompt.csv`
- `sensitivity_by_task_model_scorer.csv`
- `artifact_gap.csv`
- `run_health_by_prompt.csv`
- `scorer_delta_by_prompt.csv`
- `accuracy_by_prompt_axes.csv`
- `per_example_prompt_sensitivity.csv`
- `accuracy_<task>_<model>.png`

`prompt_robustness.summarize_judge_audit` writes:

- `judge_audit_summary_by_task_model.csv`
- `judge_audit_summary_by_prompt.csv`
- `judge_rule_disagreements.csv`

## Implementation Notes

Important scripts:

- `prompt_robustness.data`: dataset loading, frozen fixtures, frozen examples.
- `prompt_robustness.prompts`: prompt rendering from `configs/prompts.json`.
- `prompt_robustness.inference`: OpenAI inference, response metadata, run logs.
- `prompt_robustness.scoring`: strict and deterministic equivalence-aware scoring.
- `prompt_robustness.analyze`: aggregate accuracy, sensitivity, artifact gap, plots.
- `prompt_robustness.retry_failed`: rerun failed/empty records.
- `prompt_robustness.merge_retries`: merge retry records into a clean run.
- `prompt_robustness.judge_equivalence`: LLM-judge audit.
- `prompt_robustness.summarize_run`: token/status/timing summary.
- `prompt_robustness.summarize_judge_audit`: judge audit summaries.

Generated local artifacts ignored by Git:

- `.venv/`
- `.env`
- `data/frozen/`
- `runs/raw_outputs/*.jsonl`
- `runs/scores/*.jsonl`
- `runs/plots/**`
- `runs/logs/*.log`
- `*.egg-info/`
- `*.pdf`

## Remaining Work

The engineering and inference pipeline is complete. Remaining work is report-focused:

- select final tables/figures
- write methods section
- interpret main results
- describe the LLM-judge audit
- write limitations and conclusion
