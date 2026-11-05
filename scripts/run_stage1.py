#!/usr/bin/env python3
"""Stage 1 measurement: where does a policy resolve its outcome variance?

Real model (needs a GPU and vLLM):
  python scripts/run_stage1.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --prompts data/math500.jsonl --out runs/r1d15b_pilot \
      --max-new-tokens 8192 --interval 512 --k0 32 --m-base 4 --k 16 --max-prompts 50

Apple silicon (MLX backend, original fp16/bf16 weights):
  python scripts/run_stage1.py --backend mlx --model Qwen/Qwen2.5-1.5B-Instruct ...

Toy smoke test (CPU, no model; exercises the full pipeline against exact ground truth):
  python scripts/run_stage1.py --toy --out runs/toy

Outputs in --out: manifest.json, selected_prompts.jsonl, trajectories.jsonl (resumable).
"""

import argparse
import dataclasses
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tcredit import __version__  # noqa: E402
from tcredit.data import load_prompts  # noqa: E402
from tcredit.measure import Stage1Config, run_measurement, select_prompts  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--toy", action="store_true", help="run on exactly solvable toy processes instead of a model")
    ap.add_argument("--model")
    ap.add_argument("--prompts", help="JSONL with id/problem/answer (see scripts/prepare_data.py)")
    # prompt formatting
    ap.add_argument("--no-chat-template", action="store_true", help="base models: feed the raw prompt")
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument(
        "--user-template",
        default="{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    )
    # sampling: must match the policy you want to characterise (e.g. the RL rollout settings)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--min-p", type=float, default=0.0)
    # reward
    ap.add_argument("--allow-truncated-reward", action="store_true", help="grade truncated responses too (default: R=0)")
    ap.add_argument("--grade-whole-text", action="store_true", help="do not restrict grading to text after </think>")
    # engine
    ap.add_argument("--backend", choices=["auto", "vllm", "mlx"], default="auto",
                    help="auto: mlx on macOS/arm64, vllm otherwise")
    ap.add_argument("--revision", default=None, help="model revision (branch, tag or commit); mlx backend")
    ap.add_argument("--mlx-dtype", choices=["auto", "float32"], default="auto",
                    help="mlx: 'auto' keeps the stored dtype; 'float32' upcasts exactly (prefill/decode agree)")
    ap.add_argument("--max-batch", type=int, default=96, help="mlx: rows decoded together")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--dtype", default="auto")
    # design
    for f in dataclasses.fields(Stage1Config):
        name = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        elif f.name in ("max_prompts", "max_checkpoints"):
            ap.add_argument(name, type=int, default=f.default)
        else:
            ap.add_argument(name, type=type(f.default), default=f.default)
    ap.add_argument("--skip-selection", action="store_true", help="measure all prompts (no pass-rate filter)")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="stop after this many chunks (resume later); exit code 3 means prompts remain")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg = Stage1Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Stage1Config)})
    os.makedirs(args.out, exist_ok=True)

    if args.toy:
        from tcredit.toy import ToyBackend, toy_reward_fn

        procs, prompts = _toy_setup()
        backend = ToyBackend(procs, seed=cfg.seed)
        reward_fn = toy_reward_fn(procs)
        cfg.max_new_tokens = procs[0].budget
    else:
        if not args.model or not args.prompts:
            sys.exit("--model and --prompts are required unless --toy")
        from tcredit.backends import SamplingConfig
        from tcredit.rewards import make_math_reward_fn

        prompts = load_prompts(args.prompts)
        sampling = SamplingConfig(args.temperature, args.top_p, args.top_k, args.min_p)
        which = args.backend
        if which == "auto":
            which = "mlx" if (sys.platform == "darwin" and platform.machine() == "arm64") else "vllm"
        args.backend = which
        if which == "mlx":
            from tcredit.backends_mlx import MLXBackend

            backend = MLXBackend(
                args.model,
                sampling,
                revision=args.revision,
                chat_template=not args.no_chat_template,
                system_prompt=args.system_prompt,
                user_template=args.user_template,
                seed=cfg.seed,
                max_batch=args.max_batch,
                max_model_len=args.max_model_len,
                dtype=args.mlx_dtype,
            )
        else:
            from tcredit.backends import VLLMBackend

            backend = VLLMBackend(
                args.model,
                sampling,
                chat_template=not args.no_chat_template,
                system_prompt=args.system_prompt,
                user_template=args.user_template,
                seed=cfg.seed,
                tensor_parallel_size=args.tp,
                gpu_memory_utilization=args.gpu_mem,
                max_model_len=args.max_model_len,
                dtype=args.dtype,
            )
        reward_fn = make_math_reward_fn(
            backend.decode, require_finished=not args.allow_truncated_reward, after_think=not args.grade_whole_text
        )

    manifest = {
        "tcredit_version": __version__,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "args": vars(args),
        "config": dataclasses.asdict(cfg),
    }
    if not args.toy and args.backend == "vllm":
        import vllm

        manifest["vllm_version"] = vllm.__version__
        try:  # the Hub commit that "main" resolved to at run time (vLLM records only the name)
            from huggingface_hub import HfApi

            manifest["model_revision"] = HfApi().model_info(args.model).sha
        except Exception as e:
            manifest["model_revision"] = f"unknown ({type(e).__name__})"
        import torch

        manifest["torch_version"] = torch.__version__
        manifest["cuda_version"] = torch.version.cuda
        manifest["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        manifest["platform"] = platform.platform()
    if not args.toy and args.backend == "mlx":
        from importlib.metadata import version

        import mlx.core as mx

        manifest["mlx_version"] = version("mlx")
        manifest["mlx_lm_version"] = version("mlx-lm")
        manifest["transformers_version"] = version("transformers")
        manifest["model_path"] = backend.model_path
        manifest["model_revision"] = backend.model_revision
        manifest["model_dtype"] = backend.model_dtype
        manifest["stop_ids"] = sorted(backend.stop_ids)
        manifest["device"] = mx.device_info().get("device_name")
        manifest["platform"] = platform.platform()
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    t0 = time.time()  # throughput covers selection and measurement
    sel_path = os.path.join(args.out, "selected_prompts.jsonl")
    if os.path.exists(sel_path):
        with open(sel_path) as f:
            selected = [json.loads(l) for l in f if l.strip()]
        print(f"reusing {len(selected)} selected prompts from {sel_path}")
    else:
        if args.skip_selection:
            selected = prompts[: cfg.max_prompts] if cfg.max_prompts else prompts
        else:
            selected = select_prompts(backend, prompts, reward_fn, cfg)
        with open(sel_path, "w") as f:
            f.write("".join(json.dumps(p) + "\n" for p in selected))
        print(f"selected {len(selected)} of {len(prompts)} prompts")

    complete = run_measurement(backend, selected, reward_fn, cfg, os.path.join(args.out, "trajectories.jsonl"),
                               max_chunks=args.max_chunks)
    stats = getattr(backend, "stats", None)
    if stats:
        el = time.time() - t0
        stats = dict(stats, seconds=round(el, 1), generated_tok_per_s=round(stats["generated_tokens"] / el, 1))
        try:
            if args.backend == "mlx":
                import mlx.core as mx

                stats["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
        except Exception:
            pass
        with open(os.path.join(args.out, "throughput.json"), "w") as f:
            json.dump(stats, f, indent=1)
        print(stats)
    if not complete:
        print("stopped after --max-chunks; prompts remain")
        sys.exit(3)
    print("done")


def _toy_setup():
    import numpy as np

    from tcredit.toy import ToyProcess

    rng = np.random.default_rng(0)
    procs = []
    for i in range(40):
        front = i % 2 == 0
        a = (4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1) if front else (1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4)
        procs.append(
            ToyProcess(
                a=a,
                q=tuple(float(x) for x in rng.uniform(0.35, 0.65, len(a))),
                theta=int(rng.integers(4, 8)),
                h_below=float(rng.uniform(0.02, 0.12)),
                h_above=float(rng.uniform(0.15, 0.5)),
                min_len=3,
            )
        )
    prompts = [{"id": f"toy{i}", "toy_index": i, "problem": "", "answer": ""} for i in range(len(procs))]
    return procs, prompts


if __name__ == "__main__":
    main()
