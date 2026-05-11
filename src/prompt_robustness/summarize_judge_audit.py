from __future__ import annotations

import argparse
from pathlib import Path

from .config import RUNS_DIR, ensure_parent
from .io import read_jsonl


def summarize_judge_audit(input_path: Path, out_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Judge audit summary requires pandas.") from exc

    ensure_parent(out_dir / "placeholder")
    df = pd.DataFrame(read_jsonl(input_path))
    if df.empty:
        raise ValueError(f"No judge records found in {input_path}")

    df["judge_parse_failed"] = df["judge_equiv_correct"].isna()
    df["rule_equiv_agrees_with_judge"] = df["equiv_correct"] == df["judge_equiv_correct"]
    df["judge_recovers_rule_miss"] = df["judge_equiv_correct"].eq(True) & df["equiv_correct"].eq(False)
    df["judge_rejects_rule_recovery"] = df["judge_equiv_correct"].eq(False) & df["equiv_correct"].eq(True)

    summary = (
        df.groupby(["task", "model"], as_index=False)
        .agg(
            n=("example_id", "count"),
            rule_equiv_true=("equiv_correct", "sum"),
            judge_equiv_true=("judge_equiv_correct", "sum"),
            judge_parse_failures=("judge_parse_failed", "sum"),
            rule_judge_agreement=("rule_equiv_agrees_with_judge", "mean"),
            judge_recovers_rule_misses=("judge_recovers_rule_miss", "sum"),
            judge_rejects_rule_recoveries=("judge_rejects_rule_recovery", "sum"),
        )
    )
    summary.to_csv(out_dir / "judge_audit_summary_by_task_model.csv", index=False)

    by_prompt = (
        df.groupby(["task", "model", "prompt_id", "output_format"], as_index=False)
        .agg(
            n=("example_id", "count"),
            rule_equiv_rate=("equiv_correct", "mean"),
            judge_equiv_rate=("judge_equiv_correct", "mean"),
            judge_parse_failure_rate=("judge_parse_failed", "mean"),
            rule_judge_agreement=("rule_equiv_agrees_with_judge", "mean"),
            judge_recovers_rule_misses=("judge_recovers_rule_miss", "sum"),
            judge_rejects_rule_recoveries=("judge_rejects_rule_recovery", "sum"),
        )
    )
    by_prompt.to_csv(out_dir / "judge_audit_summary_by_prompt.csv", index=False)

    disagreements = df[df["equiv_correct"] != df["judge_equiv_correct"]].copy()
    columns = [
        "task",
        "model",
        "example_id",
        "prompt_id",
        "gold",
        "raw_output",
        "strict_correct",
        "equiv_correct",
        "judge_equiv_correct",
        "judge_raw_output",
    ]
    disagreements[columns].to_csv(out_dir / "judge_rule_disagreements.csv", index=False)

    print(f"records={len(df)}")
    print(f"judge_parse_failures={int(df['judge_parse_failed'].sum())}")
    print(f"rule_judge_agreement={df['rule_equiv_agrees_with_judge'].mean():.4f}")
    print(f"judge_recovers_rule_misses={int(df['judge_recovers_rule_miss'].sum())}")
    print(f"judge_rejects_rule_recoveries={int(df['judge_rejects_rule_recovery'].sum())}")
    print(f"out_dir={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an LLM-judge equivalence audit.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR / "plots" / "judge_audit")
    args = parser.parse_args()
    summarize_judge_audit(args.input, args.out_dir)


if __name__ == "__main__":
    main()
