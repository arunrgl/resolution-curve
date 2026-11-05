#!/usr/bin/env python3
"""Throughput of the MLX backend on a few MATH-500 prompts (generated tokens/s, peak memory).

  python scripts/bench_mlx.py --model Qwen/Qwen2.5-0.5B-Instruct --prompts data/math500.jsonl \
      --n-prompts 3 --n 16 --max-new-tokens 1024 --max-batch 16 32 48
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tcredit.backends import GenRequest, SamplingConfig  # noqa: E402
from tcredit.backends_mlx import MLXBackend  # noqa: E402
from tcredit.data import load_prompts  # noqa: E402

USER = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-batch", type=int, nargs="+", default=[48])
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--continuations", action="store_true",
                    help="also time continuation requests from mid-response prefixes (as in measure_chunk)")
    ap.add_argument("--out", default=None, help="append JSON lines with results")
    args = ap.parse_args()

    import mlx.core as mx

    b = MLXBackend(args.model, SamplingConfig(args.temperature), user_template=USER, dtype=args.dtype)
    prompts = load_prompts(args.prompts)[: args.n_prompts]
    pids = [b.prompt_ids(p) for p in prompts]
    b.generate([GenRequest(pids[0], [], 2, 8)])  # warm-up (kernel compilation)
    for mb in args.max_batch:
        b.max_batch = mb
        for kind in ["base"] + (["continuation"] if args.continuations else []):
            if kind == "base":
                reqs = [GenRequest(p, [], args.n, args.max_new_tokens) for p in pids]
            else:
                seed = b.generate([GenRequest(p, [], 1, args.max_new_tokens) for p in pids])
                reqs = []
                for p, (o,) in zip(pids, seed):
                    t = len(o.token_ids) // 2
                    reqs.append(GenRequest(p, o.token_ids[:t], args.n, args.max_new_tokens - t))
            b.stats = {k: 0 for k in b.stats}
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            outs = b.generate(reqs)
            el = time.perf_counter() - t0
            lens = [len(o.token_ids) for r in outs for o in r]
            fin = sum(o.finished for r in outs for o in r)
            res = {
                "model": args.model, "dtype": b.model_dtype, "revision": b.model_revision, "kind": kind,
                "max_batch": mb, "samples": len(lens), "finished": fin, "mean_len": round(sum(lens) / len(lens), 1),
                "max_len": max(lens), "generated_tokens": b.stats["generated_tokens"], "seconds": round(el, 2),
                "gen_tok_per_s": round(b.stats["generated_tokens"] / el, 1),
                "prefill_seconds": round(b.stats["prefill_seconds"], 2),
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
            }
            print(json.dumps(res), flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
