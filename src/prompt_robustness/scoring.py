from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Optional

from .config import RUNS_DIR
from .io import read_jsonl, write_jsonl


TREC_ALIASES = {
    "ABBR": {"ABBR", "ABBREVIATION", "ABBREVIATED"},
    "ENTY": {"ENTY", "ENTITY", "THING", "OBJECT"},
    "DESC": {"DESC", "DESCRIPTION", "DEFINITION", "EXPLANATION"},
    "HUM": {"HUM", "HUMAN", "PERSON", "PEOPLE"},
    "LOC": {"LOC", "LOCATION", "PLACE"},
    "NUM": {"NUM", "NUMBER", "NUMERIC", "QUANTITY"}
}


def _normalize_text(value: str) -> str:
    return value.strip().strip('"').strip("'").strip().rstrip(".").strip()


def _expected_strict_output(task: str, gold: str, output_format: str) -> Optional[str]:
    if output_format == "answer_only":
        return gold
    if output_format == "sentence":
        return f"The answer is {gold}."
    if output_format == "tagged":
        return f"<answer>{gold}</answer>"
    return None


def strict_correct(record: Dict[str, Any]) -> bool:
    output = record["raw_output"].strip()
    gold = str(record["gold"]).strip()
    output_format = record["output_format"]

    expected = _expected_strict_output(record["task"], gold, output_format)
    if expected is not None:
        return output == expected

    if output_format == "json":
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return False
        return set(parsed.keys()) == {"answer"} and str(parsed["answer"]) == gold

    raise KeyError(f"Unknown output format: {output_format}")


def _canonical_trec(value: str) -> Optional[str]:
    normalized = re.sub(r"[^A-Za-z]+", " ", value).upper().strip()
    tokens = set(normalized.split())

    exact = normalized.replace(" ", "")
    for canonical, aliases in TREC_ALIASES.items():
        if exact == canonical or exact in aliases:
            return canonical
        if tokens.intersection(aliases):
            return canonical
    return None


def _extract_json_answer(output: str) -> Optional[str]:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and "answer" in parsed:
        return str(parsed["answer"])
    return None


def _extract_tagged_answer(output: str) -> Optional[str]:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", output, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_after_answer_is(output: str) -> Optional[str]:
    matches = re.findall(r"answer\s+is\s+([^.\n]+)", output, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def _extract_trec_answer(output: str) -> Optional[str]:
    for extractor in (_extract_json_answer, _extract_tagged_answer, _extract_after_answer_is):
        extracted = extractor(output)
        if extracted:
            canonical = _canonical_trec(extracted)
            if canonical:
                return canonical

    canonical = _canonical_trec(output)
    if canonical:
        return canonical

    candidates = []
    upper = output.upper()
    for label in TREC_ALIASES:
        if re.search(rf"\b{label}\b", upper):
            candidates.append(label)
    return candidates[-1] if len(candidates) == 1 else None


def _decimal_from_text(value: str) -> Optional[Decimal]:
    cleaned = value.strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.rstrip(".")
    cleaned = re.sub(r"\s+", "", cleaned)
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _extract_last_number(output: str) -> Optional[str]:
    matches = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", output)
    if not matches:
        return None
    return matches[-1]


def _extract_gsm8k_answer(output: str) -> Optional[str]:
    for extractor in (_extract_json_answer, _extract_tagged_answer, _extract_after_answer_is):
        extracted = extractor(output)
        if extracted:
            number = _extract_last_number(extracted)
            if number:
                return number
            if _decimal_from_text(extracted) is not None:
                return extracted
    return _extract_last_number(output)


def equivalence_correct(record: Dict[str, Any]) -> bool:
    task = record["task"]
    gold = str(record["gold"]).strip()
    output = record["raw_output"].strip()

    if task == "trec6":
        predicted = _extract_trec_answer(output)
        return predicted == gold

    if task == "gsm8k":
        predicted = _extract_gsm8k_answer(output)
        pred_num = _decimal_from_text(predicted) if predicted is not None else None
        gold_num = _decimal_from_text(gold)
        return pred_num is not None and gold_num is not None and pred_num == gold_num

    raise KeyError(f"No equivalence-aware scorer for task: {task}")


def score_record(record: Dict[str, Any]) -> Dict[str, Any]:
    scored = dict(record)
    scored["strict_correct"] = strict_correct(record)
    scored["equiv_correct"] = equivalence_correct(record)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw model outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "scores" / "scored.jsonl")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output already exists. Pick a new path or delete it: {args.output}")

    write_jsonl(args.output, (score_record(record) for record in read_jsonl(args.input)))


if __name__ == "__main__":
    main()
