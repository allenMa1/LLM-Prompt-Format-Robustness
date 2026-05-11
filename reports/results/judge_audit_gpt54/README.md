# GPT-5.4 Judge Audit

Secondary LLM-judge audit for the clean main experiment.

Source artifacts:

- Input scored outputs: `runs/scores/main_clean_scored_v2.jsonl`
- Judge audit outputs: `runs/scores/main_clean_judge_audit_gpt54.jsonl`
- Generated summaries: `runs/plots/main_clean_judge_audit_gpt54/`

Included files:

- `judge_audit_summary_by_task_model.csv`
- `judge_audit_summary_by_prompt.csv`
- `judge_rule_disagreements.csv`
- `main_clean_judge_audit_gpt54.jsonl`

The JSONL file contains the full per-record judge audit outputs, including judged records, judge prompts, raw judge responses, parsed judge decisions, token usage, and timing metadata.

Audit setup:

- Judge model: `gpt-5.4`
- Selection: all records with `strict_correct = false`
- Number of judged records: 326
- Judge max output tokens: 2,048

Result:

- Judge parse failures: 0
- Rule-based equivalence scorer and LLM judge agreement: 100%
- Judge recovered rule-based misses: 0
- Judge rejected rule-based recoveries: 0

Interpretation:

The LLM-judge audit validates the deterministic equivalence-aware scorer on the subset where strict scoring failed. In this audit, the judge neither found additional semantically correct outputs missed by the rule-based scorer nor rejected any outputs recovered by the rule-based scorer.
