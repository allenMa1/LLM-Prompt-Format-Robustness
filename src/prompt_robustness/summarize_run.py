from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .io import read_jsonl


def _usage_value(record: Dict[str, Any], key: str) -> int:
    usage = record.get("usage") or {}
    return int(usage.get(key) or 0)


def _reasoning_tokens(record: Dict[str, Any]) -> int:
    usage = record.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    return int(details.get("reasoning_tokens") or 0)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def summarize(path: Path) -> None:
    rows = list(read_jsonl(path))
    print(f"file={path}")
    print(f"rows={len(rows)}")
    if not rows:
        return

    timestamps = [_parse_timestamp(r["created_at"]) for r in rows if r.get("created_at")]
    if timestamps:
        wall_seconds = (max(timestamps) - min(timestamps)).total_seconds()
        print(f"timestamp_span_s={wall_seconds:.1f}")

    elapsed_values = [float(r["call_elapsed_s"]) for r in rows if r.get("call_elapsed_s") is not None]
    if elapsed_values:
        print(f"sum_call_elapsed_s={sum(elapsed_values):.1f}")
        print(f"avg_call_elapsed_s={sum(elapsed_values) / len(elapsed_values):.2f}")
        print(f"max_call_elapsed_s={max(elapsed_values):.2f}")

    print(f"total_input_tokens={sum(_usage_value(r, 'input_tokens') for r in rows)}")
    print(f"total_output_tokens={sum(_usage_value(r, 'output_tokens') for r in rows)}")
    print(f"total_reasoning_tokens={sum(_reasoning_tokens(r) for r in rows)}")
    print(f"total_tokens={sum(_usage_value(r, 'total_tokens') for r in rows)}")
    print(f"noncompleted={sum(r.get('response_status') != 'completed' for r in rows if 'response_status' in r)}")
    print(f"empty_outputs={sum(not str(r.get('raw_output', '')).strip() for r in rows)}")

    groups: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row.get("task", ""), row.get("model", "")), []).append(row)

    print("\nby task/model:")
    for (task, model), group in sorted(groups.items()):
        n = len(group)
        total_tokens = sum(_usage_value(r, "total_tokens") for r in group)
        output_tokens = sum(_usage_value(r, "output_tokens") for r in group)
        reasoning_tokens = sum(_reasoning_tokens(r) for r in group)
        print(
            f"{task} {model}: rows={n} "
            f"avg_total_tokens={total_tokens / n:.1f} "
            f"avg_output_tokens={output_tokens / n:.1f} "
            f"avg_reasoning_tokens={reasoning_tokens / n:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize token usage and timing for a raw inference JSONL file.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input)


if __name__ == "__main__":
    main()
