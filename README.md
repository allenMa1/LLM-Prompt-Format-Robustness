# Prompt and Format Robustness in LLM Evaluation

This repository studies prompt and output-format robustness in LLM evaluation.

The main question is:

> How much do semantically equivalent prompt and format changes affect measured LLM performance on structured tasks, and how much does that effect depend on the scoring method?

The experiment evaluates two OpenAI models on TREC-6 and GSM8K using eight prompt variants. The same raw outputs are scored with strict exact-format scoring and with an equivalence-aware scorer that normalizes harmless answer-format differences. A secondary LLM-judge audit checks the equivalence-aware scorer on cases where strict scoring fails.

## Summary

- Tasks: TREC-6 question classification and GSM8K grade-school math
- Examples: 100 test examples per task
- Models: `gpt-5-nano`, `gpt-5-mini`
- Prompt variants: 8 per task, from `2 instruction wordings x 4 output formats`
- Main run: 3,200 completed model responses
- Scoring methods: strict exact-format scoring and equivalence-aware scoring
- Judge audit: `gpt-5.4` over 326 strict-false outputs

Final report:

```text
reports/final_report/final_report.pdf
```

Report source:

```text
reports/final_report/
```

Final result artifacts:

```text
reports/results/main_clean/
reports/results/full_outputs/
reports/results/judge_audit_gpt54/
```

## Results

Prompt sensitivity is:

```text
max prompt accuracy - min prompt accuracy
```

| Task | Model | Strict | Equivalence-Aware | Artifact Gap |
| --- | --- | ---: | ---: | ---: |
| GSM8K | `gpt-5-mini` | 0.03 | 0.02 | 0.01 |
| GSM8K | `gpt-5-nano` | 0.07 | 0.06 | 0.01 |
| TREC-6 | `gpt-5-mini` | 0.05 | 0.05 | 0.00 |
| TREC-6 | `gpt-5-nano` | 0.08 | 0.08 | 0.00 |

Main findings:

- Prompt choice changes measured accuracy even when task, examples, model, and scorer are fixed.
- Equivalence-aware scoring reduces measured sensitivity on GSM8K, but the sensitivity does not disappear.
- TREC-6 shows no strict-versus-equivalence gap, suggesting its prompt sensitivity is not caused by answer-format scoring artifacts.
- `gpt-5-nano` is more prompt-sensitive than `gpt-5-mini` in this experiment.
- The `gpt-5.4` judge audit agrees exactly with the deterministic equivalence-aware scorer on all strict-false cases.

## Repository Layout

```text
configs/
  models.json
  prompts.json
  tasks.json

data/
  fixtures/

src/prompt_robustness/
  data.py
  prompts.py
  inference.py
  scoring.py
  analyze.py
  freeze_examples.py
  retry_failed.py
  merge_retries.py
  summarize_run.py
  judge_equivalence.py
  summarize_judge_audit.py

reports/
  final_report/
  results/

runs/
  raw_outputs/
  scores/
  plots/
  logs/
```

`runs/` and `data/frozen/` are generated local artifacts and are ignored by Git. Curated final outputs are committed under `reports/results/`.

## Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Create a local `.env` file:

```text
OPENAI_API_KEY=...
```

The `.env` file is ignored by Git.

## Reproduction

Freeze evaluation examples:

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

Retry incomplete records if needed:

```powershell
python -m prompt_robustness.retry_failed --input runs/raw_outputs/main.jsonl --output runs/raw_outputs/main_retry.jsonl --log-file runs/logs/main_retry.log
python -m prompt_robustness.merge_retries --base runs/raw_outputs/main.jsonl --retry runs/raw_outputs/main_retry.jsonl --output runs/raw_outputs/main_clean.jsonl
```

Analyze the clean run:

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

- Sclar et al., "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design"
- Mizrahi et al., "State of What Art? A Call for Multi-Prompt LLM Evaluation"
- Hua et al., "Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs"
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
