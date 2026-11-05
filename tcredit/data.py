"""Prompt sets: JSONL with one object per line: {"id": str, "problem": str, "answer": str}."""

from __future__ import annotations

import json


def load_prompts(path: str) -> list[dict]:
    out = []
    seen = set()
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for k in ("id", "problem", "answer"):
                if k not in row:
                    raise ValueError(f"{path}:{n}: missing field {k!r}")
            row["id"] = str(row["id"])
            if row["id"] in seen:
                raise ValueError(f"{path}:{n}: duplicate id {row['id']!r}")
            seen.add(row["id"])
            out.append(row)
    return out


def gsm8k_answer(solution: str) -> str:
    """GSM8K reference solutions end with '#### <answer>'."""
    if "####" not in solution:
        raise ValueError("GSM8K solution without '####'")
    return solution.rsplit("####", 1)[1].strip().replace(",", "")
