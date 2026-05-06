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
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your OpenAI API key before inference:

```powershell
$env:OPENAI_API_KEY="..."
```

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

## Main Run

```powershell
python -m prompt_robustness.inference --output runs/raw_outputs/main.jsonl
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
