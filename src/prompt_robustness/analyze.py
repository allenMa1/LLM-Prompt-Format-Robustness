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
        id_vars=["task", "model", "prompt_id", "wording", "output_format"],
        value_vars=["strict_correct", "equiv_correct"],
        var_name="scorer",
        value_name="correct"
    )
    long_df["scorer"] = long_df["scorer"].map(
        {"strict_correct": "strict", "equiv_correct": "equivalence_aware"}
    )

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
