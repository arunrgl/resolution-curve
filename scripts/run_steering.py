#!/usr/bin/env python3
"""H1 steering influence (docs/methods.md §11) on Stage 1 base trajectories (MLX).

  python scripts/run_steering.py --model Qwen/Qwen2.5-0.5B-Instruct --temperature 0.7 \
      --trajectories runs/pilotA_q05b/trajectories.jsonl --out runs/pilotA_q05b/steering.jsonl

Use the same model, revision, dtype and temperature as the Stage 1 run. Resumable.
"""

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tcredit.records import read_jsonl  # noqa: E402
from tcredit.steering import SteeringRecord, steering_for_trajectory, steering_for_trajectory_torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--mlx-dtype", choices=["auto", "float32"], default="auto")
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--swaps", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--torch-dtype", default="bfloat16")
    args = ap.parse_args()

    if args.backend == "mlx":
        from tcredit.backends import SamplingConfig
        from tcredit.backends_mlx import MLXBackend

        b = MLXBackend(args.model, SamplingConfig(args.temperature), revision=args.revision, dtype=args.mlx_dtype, seed=args.seed)
        model, stop_ids, rev, dtype = b.model, b.stop_ids, b.model_revision, b.model_dtype
        fn = steering_for_trajectory
        extra = {}
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=getattr(torch, args.torch_dtype))
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        stop_ids = {int(tok.eos_token_id)} if tok.eos_token_id is not None else set()
        try:
            e = GenerationConfig.from_pretrained(args.model, revision=args.revision).eos_token_id
            stop_ids |= {e} if isinstance(e, int) else set(e or [])
        except Exception:
            pass
        rev, dtype = args.revision, str(next(model.parameters()).dtype)
        fn = steering_for_trajectory_torch
        extra = {"generator": torch.Generator(device=next(model.parameters()).device).manual_seed(args.seed)}
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["prompt_id"], r["traj_id"]) for r in map(json.loads, f) if r}
    meta = {"model": args.model, "revision": rev, "dtype": dtype, "stop_ids": sorted(stop_ids), **vars(args)}
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(meta, f, indent=1)
    trajs = list(read_jsonl(args.trajectories))
    t0 = time.time()
    for i, tr in enumerate(trajs):
        if (tr.prompt_id, tr.traj_id) in done:
            continue
        if tr.token_ids is None or tr.prompt_token_ids is None:
            sys.exit("trajectories lack token ids (run Stage 1 with --store-tokens true)")
        cols = fn(model, tr.prompt_token_ids, tr.token_ids, stop_ids=stop_ids, temperature=args.temperature,
                  stride=args.stride, swaps=args.swaps, window=args.window, **extra)
        rec = SteeringRecord(tr.prompt_id, tr.traj_id, tr.n_tokens, args.temperature, args.window, args.swaps, **cols)
        with open(args.out, "a") as f:
            f.write(json.dumps(dataclasses.asdict(rec)) + "\n")
        print(f"{i + 1}/{len(trajs)} {tr.prompt_id}#{tr.traj_id}: {len(cols['t'])} positions, {time.time() - t0:.0f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
