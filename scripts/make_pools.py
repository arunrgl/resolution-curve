#!/usr/bin/env python3
"""Rebuild the prompt pools used in the Stage 1 runs from data/math500.jsonl.

  python scripts/prepare_data.py --preset math500 --out data/math500.jsonl
  python scripts/make_pools.py

Pools are the first 100 / 150 prompts of a fixed random permutation (seed 20261103) of
MATH-500 in file order, written in file order. With --selected, also rebuilds a run's
selected_prompts.jsonl from its results/<run>/selected_ids.json.
"""

import argparse
import json
import os

import numpy as np

SEED = 20261103


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--math500", default="data/math500.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--selected", nargs=2, metavar=("SELECTED_IDS_JSON", "OUT_JSONL"), action="append", default=[],
                    help="rebuild selected_prompts.jsonl (id, problem, answer, p_select) from a selected_ids.json")
    args = ap.parse_args()
    with open(args.math500) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    perm = np.random.default_rng(SEED).permutation(len(rows))
    for n in (100, 150):
        path = os.path.join(args.out_dir, f"math500_pool{n}_seed{SEED}.jsonl")
        with open(path, "w") as f:
            for i in sorted(perm[:n]):
                f.write(json.dumps(rows[i]) + "\n")
        print("wrote", path)
    by_id = {r["id"]: r for r in rows}
    for ids_path, out in args.selected:
        with open(ids_path) as f:
            sel = json.load(f)
        with open(out, "w") as f:
            for s in sel:
                q = dict(by_id[s["id"]])
                q["p_select"] = s["p_select"]
                f.write(json.dumps(q) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
