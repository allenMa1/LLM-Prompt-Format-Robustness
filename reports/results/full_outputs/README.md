# Full Clean Outputs

Full per-record outputs for the clean main experiment.

Included files:

- `main_clean_raw_outputs.jsonl`: raw model outputs after merging retry records for all incomplete responses.
- `main_clean_scored_outputs.jsonl`: the same 3,200 records with strict and rule-based equivalence-aware scoring fields added.

Provenance:

- Base raw run: `runs/raw_outputs/main.jsonl`
- Retry run for 14 incomplete records: `runs/raw_outputs/main_retry_4096.jsonl`
- Merged clean raw run: `runs/raw_outputs/main_clean.jsonl`
- Clean scored run: `runs/scores/main_clean_scored_v2.jsonl`

Run health:

- Records: 3,200
- Noncompleted responses after retry merge: 0
- Empty outputs after retry merge: 0

These files are copied into `reports/results` for bookkeeping and reproducibility. The generated `runs/` JSONL files remain ignored by Git.
