#!/usr/bin/env python3
"""Measure response lengths, finish rate, accuracy and throughput before sizing a Stage 1 run.

  python scripts/probe_lengths.py --backend vllm --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --prompts data/math500_pool150_seed20261103.jsonl --n-prompts 20 --n 4 \
      --temperature 0.6 --max-new-tokens 16384 --max-model-len 17408 --out probe.json

Prints the methods §6 cost terms: mean N and E[N^2] (continuation cost ~ P*M*k*E[N^2]/(2L)).
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tcredit.backends import GenRequest, SamplingConfig  # noqa: E402
from tcredit.data import load_prompts  # noqa: E402
from tcredit.rewards import make_math_reward_fn  # noqa: E402

USER = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["vllm", "mlx"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s = SamplingConfig(args.temperature, args.top_p)
    if args.backend == "vllm":
        from tcredit.backends import VLLMBackend

        b = VLLMBackend(args.model, s, chat_template=not args.no_chat_template, user_template=USER,
                        max_model_len=args.max_model_len)
    else:
        from tcredit.backends_mlx import MLXBackend

        b = MLXBackend(args.model, s, chat_template=not args.no_chat_template, user_template=USER)
    prompts = load_prompts(args.prompts)[: args.n_prompts]
    rf = make_math_reward_fn(b.decode)
    t0 = time.time()
    outs = b.generate([GenRequest(b.prompt_ids(p), [], args.n, args.max_new_tokens) for p in prompts])
    el = time.time() - t0
    N = np.array([len(o.token_ids) + o.finished for r in outs for o in r], dtype=float)
    fin = np.array([o.finished for r in outs for o in r])
    acc = np.mean([rf(p, o.token_ids, o.finished) for p, r in zip(prompts, outs) for o in r])
    res = {
        "model": args.model, "backend": args.backend, "platform": platform.platform(), "args": vars(args),
        "samples": int(len(N)), "finished_frac": float(fin.mean()), "accuracy": float(acc),
        "N_mean": float(N.mean()), "N_p50": float(np.median(N)), "N_p90": float(np.percentile(N, 90)),
        "N_max": float(N.max()), "EN2_sqrt": float(np.sqrt((N ** 2).mean())),
        "seconds": round(el, 1), "gen_tok_per_s": round(float(N.sum()) / el, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
