from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .io import read_jsonl, write_jsonl


RecordKey = Tuple[str, str, str, str]


def _key(record: Dict[str, Any]) -> RecordKey:
    return (
        record["task"],
        record["example_id"],
        record["prompt_id"],
        record["model"],
    )


def _is_failed(record: Dict[str, Any]) -> bool:
    return record.get("response_status") != "completed" or not str(record.get("raw_output", "")).strip()


def merge_retries(base_path: Path, retry_path: Path, output_path: Path, require_base_failed: bool = True) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output already exists. Pick a new path or delete it: {output_path}")

    base_records = list(read_jsonl(base_path))
    retry_records = list(read_jsonl(retry_path))
    retry_by_key = {_key(record): record for record in retry_records}

    if len(retry_by_key) != len(retry_records):
        raise ValueError("Retry file contains duplicate task/example/prompt/model keys.")

    base_keys = {_key(record) for record in base_records}
    missing = sorted(set(retry_by_key).difference(base_keys))
    if missing:
        raise ValueError(f"Retry records not present in base file: {missing[:5]}")

    merged = []
    replaced = 0
    skipped = 0
    for record in base_records:
        key = _key(record)
        if key not in retry_by_key:
            merged.append(record)
            continue

        if require_base_failed and not _is_failed(record):
            skipped += 1
            merged.append(record)
            continue

        replacement = dict(retry_by_key[key])
        replacement["merged_from_retry"] = {
            "base_path": str(base_path),
            "retry_path": str(retry_path),
            "base_created_at": record.get("created_at"),
            "base_response_status": record.get("response_status"),
            "base_incomplete_details": record.get("incomplete_details"),
        }
        merged.append(replacement)
        replaced += 1

    write_jsonl(output_path, merged)
    print(f"base_rows={len(base_records)}")
    print(f"retry_rows={len(retry_records)}")
    print(f"replaced={replaced}")
    print(f"skipped_nonfailed={skipped}")
    print(f"output={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge retry records into a base raw-output JSONL file.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-replace-nonfailed",
        action="store_true",
        help="Allow retry records to replace base records even when the base record was completed and nonempty."
    )
    args = parser.parse_args()

    merge_retries(
        base_path=args.base,
        retry_path=args.retry,
        output_path=args.output,
        require_base_failed=not args.allow_replace_nonfailed,
    )


if __name__ == "__main__":
    main()
