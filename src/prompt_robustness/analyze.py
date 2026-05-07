from __future__ import annotations

import argparse
from pathlib import Path

from .config import RUNS_DIR, ensure_parent


def analyze(scored_path: Path, out_dir: Path) -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Analysis requires pandas and matplotlib. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    ensure_parent(out_dir / "placeholder")
    df = pd.read_json(scored_path, lines=True)

    long_df = df.melt(
        id_vars=["example_id", "task", "model", "prompt_id", "wording", "output_format"],
        value_vars=["strict_correct", "equiv_correct"],
        var_name="scorer",
        value_name="correct"
    )
    long_df["scorer"] = long_df["scorer"].map(
        {"strict_correct": "strict", "equiv_correct": "equivalence_aware"}
    )
    long_df["correct"] = long_df["correct"].astype(float)

    by_prompt = (
        long_df.groupby(["task", "model", "scorer", "prompt_id"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )
    by_prompt.to_csv(out_dir / "accuracy_by_prompt.csv", index=False)

    sensitivity = (
        by_prompt.groupby(["task", "model", "scorer"], as_index=False)
        .agg(
            avg_accuracy=("accuracy", "mean"),
            min_prompt_accuracy=("accuracy", "min"),
            max_prompt_accuracy=("accuracy", "max"),
            prompt_accuracy_std=("accuracy", "std")
        )
    )
    sensitivity["prompt_sensitivity_range"] = (
        sensitivity["max_prompt_accuracy"] - sensitivity["min_prompt_accuracy"]
    )
    sensitivity.to_csv(out_dir / "sensitivity_by_task_model_scorer.csv", index=False)

    pivot = sensitivity.pivot_table(
        index=["task", "model"],
        columns="scorer",
        values="prompt_sensitivity_range"
    ).reset_index()
    if {"strict", "equivalence_aware"}.issubset(pivot.columns):
        pivot["artifact_gap"] = pivot["strict"] - pivot["equivalence_aware"]
    pivot.to_csv(out_dir / "artifact_gap.csv", index=False)

    if {"response_status", "usage"}.issubset(df.columns):
        status_cols = ["task", "model", "prompt_id", "output_format"]
        status_summary = (
            df.assign(
                completed=df["response_status"].eq("completed"),
                empty_output=df["raw_output"].fillna("").astype(str).str.strip().eq(""),
                input_tokens=df["usage"].apply(lambda u: (u or {}).get("input_tokens", 0)),
                output_tokens=df["usage"].apply(lambda u: (u or {}).get("output_tokens", 0)),
                total_tokens=df["usage"].apply(lambda u: (u or {}).get("total_tokens", 0)),
                reasoning_tokens=df["usage"].apply(
                    lambda u: ((u or {}).get("output_tokens_details") or {}).get("reasoning_tokens", 0)
                )
            )
            .groupby(status_cols, as_index=False)
            .agg(
                n=("example_id", "count"),
                completion_rate=("completed", "mean"),
                empty_output_rate=("empty_output", "mean"),
                avg_input_tokens=("input_tokens", "mean"),
                avg_output_tokens=("output_tokens", "mean"),
                avg_reasoning_tokens=("reasoning_tokens", "mean"),
                avg_total_tokens=("total_tokens", "mean")
            )
        )
        status_summary.to_csv(out_dir / "run_health_by_prompt.csv", index=False)

    scorer_delta = df.assign(
        strict_only_wrong=df["strict_correct"].eq(False) & df["equiv_correct"].eq(True),
        both_wrong=df["strict_correct"].eq(False) & df["equiv_correct"].eq(False),
        both_right=df["strict_correct"].eq(True) & df["equiv_correct"].eq(True)
    )
    delta_summary = (
        scorer_delta.groupby(["task", "model", "prompt_id", "output_format"], as_index=False)
        .agg(
            n=("example_id", "count"),
            strict_accuracy=("strict_correct", "mean"),
            equivalence_aware_accuracy=("equiv_correct", "mean"),
            scorer_recovered_rate=("strict_only_wrong", "mean"),
            both_wrong_rate=("both_wrong", "mean"),
            both_right_rate=("both_right", "mean")
        )
    )
    delta_summary["accuracy_delta_equiv_minus_strict"] = (
        delta_summary["equivalence_aware_accuracy"] - delta_summary["strict_accuracy"]
    )
    delta_summary.to_csv(out_dir / "scorer_delta_by_prompt.csv", index=False)

    axis_summary = (
        long_df.groupby(["task", "model", "scorer", "wording", "output_format"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )
    axis_summary.to_csv(out_dir / "accuracy_by_prompt_axes.csv", index=False)

    per_example = (
        long_df.groupby(["task", "model", "scorer", "example_id"], as_index=False)["correct"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    per_example["example_prompt_sensitivity"] = per_example["max"] - per_example["min"]
    per_example.to_csv(out_dir / "per_example_prompt_sensitivity.csv", index=False)

    for (task, model), group in by_prompt.groupby(["task", "model"]):
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_df = group.pivot(index="prompt_id", columns="scorer", values="accuracy")
        plot_df.plot(kind="bar", ax=ax)
        ax.set_title(f"{task} / {model}")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        safe_model = model.replace("/", "_").replace(":", "_")
        fig.savefig(out_dir / f"accuracy_{task}_{safe_model}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scored prompt robustness outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR / "plots")
    args = parser.parse_args()
    analyze(args.input, args.out_dir)


if __name__ == "__main__":
    main()
