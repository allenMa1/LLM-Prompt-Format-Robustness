# Prompt and Format Robustness in LLM Evaluation

This repository contains a scoped empirical study of prompt and output-format robustness in LLM evaluation.

The project asks:

> How much do semantically equivalent prompt and format changes affect measured LLM performance on structured tasks, and how much does that effect depend on the scoring method?

The experiment evaluates two OpenAI models on two structured NLP tasks using eight semantically equivalent prompt variants. Each raw output is scored with both strict exact-format scoring and a deterministic equivalence-aware scorer. A secondary LLM-judge audit checks the scorer on strict-false cases.

## Project Summary

Main experiment:

- Tasks: TREC-6 question classification and GSM8K grade-school math
- Examples: 100 frozen test examples per task
- Models: `gpt-5-nano`, `gpt-5-mini`
- Prompt variants: 8 per task, from `2 instruction wordings x 4 output formats`
- Outputs: 3,200 completed model responses
- Scorers: strict exact-format scoring and equivalence-aware scoring
- Judge audit: `gpt-5.4` over all 326 strict-false outputs

The final report is available at:

```text
reports/final_report/final_report.pdf
```

Report source files are in:

```text
reports/final_report/
```

## Results

Prompt sensitivity is measured as:

```text
max prompt accuracy - min prompt accuracy
```

Main prompt-sensitivity results:

| Task | Model | Strict | Equivalence-Aware | Artifact Gap |
| --- | --- | ---: | ---: | ---: |
| GSM8K | `gpt-5-mini` | 0.03 | 0.02 | 0.01 |
| GSM8K | `gpt-5-nano` | 0.07 | 0.06 | 0.01 |
| TREC-6 | `gpt-5-mini` | 0.05 | 0.05 | 0.00 |
| TREC-6 | `gpt-5-nano` | 0.08 | 0.08 | 0.00 |

High-level findings:

- Prompt sensitivity is present but moderate in this controlled setting.
- `gpt-5-nano` is more prompt-sensitive than `gpt-5-mini`.
- Equivalence-aware scoring mainly affects GSM8K by recovering harmless numeric formatting differences.
- TREC-6 has no strict-versus-equivalence gap because outputs are usually canonical labels or genuinely wrong labels.
- The `gpt-5.4` judge audit agrees exactly with the deterministic equivalence-aware scorer on all strict-false cases.

Curated result artifacts are stored under:

```text
reports/results/main_clean/
reports/results/full_outputs/
reports/results/judge_audit_gpt54/
```

## Repository Layout

```text
configs/
  models.json              # model configuration
  prompts.json             # prompt family definition
  tasks.json               # task and dataset configuration

data/
  fixtures/                # tiny local fixtures for dry runs

src/prompt_robustness/
  data.py                  # dataset loading and frozen examples
  prompts.py               # prompt rendering
  inference.py             # OpenAI inference runner
  scoring.py               # strict and equivalence-aware scoring
  analyze.py               # aggregate metrics and plots
  freeze_examples.py       # freeze sampled evaluation examples
  retry_failed.py          # rerun failed or incomplete records
  merge_retries.py         # merge retry outputs into a clean run
  summarize_run.py         # token/status/timing summaries
  judge_equivalence.py     # LLM-judge audit
  summarize_judge_audit.py # judge audit summaries

reports/
  final_report/            # final PDF, LaTeX source, and report figures
  results/                 # committable final results

runs/
  raw_outputs/             # local generated inference outputs
  scores/                  # local generated scored outputs
  plots/                   # local generated analysis outputs
  logs/                    # local run logs
```

The `runs/` and `data/frozen/` directories are local generated artifacts and are ignored by Git. Final curated copies live under `reports/results/`.

## Setup

Use standard Windows Python, not MSYS/MinGW Python.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Create a local `.env` file in the repo root:

```text
OPENAI_API_KEY=...
```

`.env` is ignored by Git. The inference and judge scripts load it automatically.

## Reproducing the Experiment

Freeze examples:

```powershell
python -m prompt_robustness.freeze_examples --limit 100 --out-dir data/frozen/main
```

Run inference:

```powershell
python -m prompt_robustness.inference --frozen-dir data/frozen/main --output runs/raw_outputs/main.jsonl --log-file runs/logs/main.log
```

Score and analyze:

```powershell
python -m prompt_robustness.scoring --input runs/raw_outputs/main.jsonl --output runs/scores/main_scored.jsonl
python -m prompt_robustness.analyze --input runs/scores/main_scored.jsonl --out-dir runs/plots/main
python -m prompt_robustness.summarize_run --input runs/raw_outputs/main.jsonl
```

If any records are incomplete or empty, retry them and merge:

```powershell
python -m prompt_robustness.retry_failed --input runs/raw_outputs/main.jsonl --output runs/raw_outputs/main_retry.jsonl --log-file runs/logs/main_retry.log
python -m prompt_robustness.merge_retries --base runs/raw_outputs/main.jsonl --retry runs/raw_outputs/main_retry.jsonl --output runs/raw_outputs/main_clean.jsonl
```

Score and analyze the clean run:

```powershell
python -m prompt_robustness.scoring --input runs/raw_outputs/main_clean.jsonl --output runs/scores/main_clean_scored.jsonl
python -m prompt_robustness.analyze --input runs/scores/main_clean_scored.jsonl --out-dir runs/plots/main_clean
python -m prompt_robustness.summarize_run --input runs/raw_outputs/main_clean.jsonl
```

Run the LLM-judge audit:

```powershell
python -m prompt_robustness.judge_equivalence --input runs/scores/main_clean_scored.jsonl
python -m prompt_robustness.summarize_judge_audit --input runs/scores/main_clean_judge_audit_gpt54.jsonl --out-dir runs/plots/main_clean_judge_audit_gpt54
```

## References

The study is motivated by:

- Sclar et al., “Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design”
- Mizrahi et al., “State of What Art? A Call for Multi-Prompt LLM Evaluation”
- Hua et al., “Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs”
- Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena”
