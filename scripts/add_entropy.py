#!/usr/bin/env python3
"""Add per-token policy entropy to measured trajectories (needed for the entropy control).

  python scripts/add_entropy.py --model <same model as Stage 1> --temperature <same> \
      --in runs/x/trajectories.jsonl --out runs/x/trajectories_entropy.jsonl
"""

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tcredit.entropy import token_entropies  # noqa: E402
from tcredit.records import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, required=True, help="must equal the Stage 1 sampling temperature")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch",
                    help="mlx: Apple silicon, stored dtype (--dtype is ignored)")
    args = ap.parse_args()
    if args.backend == "mlx":
        from mlx_lm.utils import _download, load_model

        from tcredit.backends_mlx import _check_unquantized
        from tcredit.entropy import token_entropies_mlx

        path = _download(args.model)
        model, _ = load_model(path)
        print(f"loaded {args.model} (mlx) with dtype {_check_unquantized(model)} from {path}")
        n = 0
        with open(args.out, "w") as f:
            for t in read_jsonl(args.inp):
                if t.token_ids is None or t.prompt_token_ids is None:
                    sys.exit("trajectories were measured with --store-tokens false; cannot compute entropy")
                t.entropy = token_entropies_mlx(model, t.prompt_token_ids, t.token_ids, t.finished, args.temperature)
                t.validate()
                f.write(json.dumps(dataclasses.asdict(t)) + "\n")
                n += 1
        print(f"wrote {n} trajectories with entropy to {args.out}")
        return

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    # Newer transformers take `dtype`, older ones `torch_dtype`. If this version threshold is off, the
    # effect is a deprecation warning or a load in the default dtype (more memory, slightly
    # different numerics); check the dtype printed below.
    try:
        from packaging.version import Version

        new_api = Version(transformers.__version__) >= Version("4.56.0")
    except Exception:
        new_api = False
    dtype_kw = {"dtype" if new_api else "torch_dtype": getattr(torch, args.dtype)}
    model = AutoModelForCausalLM.from_pretrained(args.model, **dtype_kw)
    model.to(args.device).eval()
    print(f"loaded {args.model} with dtype {next(model.parameters()).dtype}")
    n = 0
    with open(args.out, "w") as f:
        for t in read_jsonl(args.inp):
            if t.token_ids is None or t.prompt_token_ids is None:
                sys.exit("trajectories were measured with --store-tokens false; cannot compute entropy")
            t.entropy = token_entropies(model, t.prompt_token_ids, t.token_ids, t.finished, args.temperature)
            t.validate()
            f.write(json.dumps(dataclasses.asdict(t)) + "\n")
            n += 1
    print(f"wrote {n} trajectories with entropy to {args.out}")


if __name__ == "__main__":
    main()
