#!/usr/bin/env python3
"""Convert a Hugging Face dataset into the Stage 1 prompt format (JSONL: id, problem, answer).

  python scripts/prepare_data.py --preset math500 --out data/math500.jsonl
  python scripts/prepare_data.py --preset gsm8k   --out data/gsm8k_test.jsonl
  python scripts/prepare_data.py --dataset <hf name> --split test \
      --problem-field problem --answer-field answer --out data/custom.jsonl

Presets (field names as published on the Hugging Face Hub; verify before use):
  math500: HuggingFaceH4/MATH-500, split test, fields problem / answer / unique_id
  gsm8k:   openai/gsm8k, config main, split test, fields question / answer ("#### <n>")
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tcredit.data import gsm8k_answer  # noqa: E402

PRESETS = {
    "math500": dict(dataset="HuggingFaceH4/MATH-500", config=None, split="test", problem="problem", answer="answer", id="unique_id"),
    "gsm8k": dict(dataset="openai/gsm8k", config="main", split="test", problem="question", answer="answer", id=None),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--dataset")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--problem-field", default="problem")
    ap.add_argument("--answer-field", default="answer")
    ap.add_argument("--id-field", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.preset:
        p = PRESETS[args.preset]
        name, config, split = p["dataset"], p["config"], p["split"]
        pf, af, idf = p["problem"], p["answer"], p["id"]
    else:
        if not args.dataset:
            sys.exit("--preset or --dataset is required")
        name, config, split = args.dataset, args.config, args.split
        pf, af, idf = args.problem_field, args.answer_field, args.id_field

    from datasets import load_dataset

    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for i, row in enumerate(ds):
            ans = gsm8k_answer(row[af]) if args.preset == "gsm8k" else str(row[af])
            rid = str(row[idf]) if idf else f"{split}-{i}"
            f.write(json.dumps({"id": rid, "problem": row[pf], "answer": ans}) + "\n")
    print(f"wrote {len(ds)} prompts to {args.out}")


if __name__ == "__main__":
    main()
