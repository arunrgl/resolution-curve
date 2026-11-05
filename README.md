# Resolution Curve: where does a language model decide its outcome?

Code for measuring, on a real language model, how much of the uncertainty about the final reward
is **resolved** at each position of a response, and how strongly each token steers the tokens
after it. It is the measurement stage of a study on whether earlier tokens deserve more credit in
RL and SFT, and is intended as the accompanying code for a paper in preparation. Every estimator is
unbiased, its identities are tested exactly against a solvable toy model, and every run carries a
built-in validity check. (The Python package is called `tcredit`.)

- `docs/methods.md`: estimands, proofs, design, pre-registered hypotheses and decision rules.
- `docs/corrections.md`: errors found along the way (in the plan and in the code) and their fixes.
- `docs/references.md` / `docs/references.bib`: citations, checked against arXiv abstracts. **Read
  the status column before citing.**
- `docs/stage2_design.md`: draft design for the RL stage (not frozen).
- `results/`: compact outputs of the completed Stage 1 runs.
- `HANDOFF.md`: running project log.

## Core idea

`V_t = E[R | prompt, first t tokens]` is a martingale that ends at the reward `R`. For checkpoints
placed at stopping times (every `L` tokens, or at delimiters, never at fractions of the final
length), the squared value jumps between checkpoints add up **exactly** to `Var(R | prompt)`. Each
segment's share of that total is where the outcome gets decided. The pipeline estimates `V` at
checkpoints with independent continuations, removes the Monte Carlo noise from the squared jumps,
and reports the cumulative resolution curve `C(u)` against relative position, the front-loading
index `F = ∫(C(u) − u) du`, and the **conservation ratio** (estimated total over an independent
estimate of `Var(R)`), whose 95% CI must contain 1 for a run to count.

A second measurement (H1, methods §11) asks whether earlier tokens shift the distribution of later
tokens more: swap a token for an alternative drawn from the policy, hold the rest of the text
fixed, and sum the KL divergences of the next 64 next-token distributions.

## Current results (Stage 1)

MATH-500, 1.5B models, 30–50 prompts per model; details and caveats in `results/README.md`.

| Model | Conservation | Outcome decided early? | Early tokens steer more? | Position effect beyond token entropy |
|---|---|---|---|---|
| Qwen2.5-1.5B (base) | 1.07 [0.85, 1.33] | yes, excluding the final segment (F = +0.12 [+0.07, +0.18]) | yes, strongly (first ~16 tokens) | none |
| Qwen2.5-1.5B-Instruct | 0.94 [0.65, 1.25] | no (F = −0.08 [−0.15, +0.00]) | somewhat (first ~128 tokens) | none |
| R1-Distill-Qwen-1.5B | 1.23 [0.89, 1.63] | no (F = −0.03 [−0.14, +0.11]) | no (flat) | none |

Wherever early tokens matter more, the extra weight is accounted for by their higher entropy.
Stages 2 (RL) and 3 (SFT) are not implemented.

## Install

```bash
pip install -r requirements.txt
```

That covers the estimators, the toy validation and the tests. For real models:

```bash
pip install vllm==0.33.0 torch transformers datasets   # NVIDIA GPU; tested on Colab A100 (torch 2.14.0+cu130)
```

```bash
pip install mlx-lm datasets torch   # Apple silicon; tested with mlx 0.32.4, mlx-lm 0.32.2, torch 2.15.0 (CPU, tests only)
```

## Verify the code

```bash
python -m pytest -q tests
```

```bash
python scripts/run_stage1.py --toy --out runs/toy --k0 64 --m-base 200 --k 8 --interval 2
python scripts/analyze_stage1.py --trajectories runs/toy/trajectories.jsonl --out runs/toy/analysis --abs-bin-width 2
```

The toy run uses processes whose values are known exactly; a correct installation reports a
conservation ratio of about 1.00.

## Data

```bash
python scripts/prepare_data.py --preset math500 --out data/math500.jsonl
python scripts/make_pools.py
```

This rebuilds the exact prompt pools used in the runs (see `data/README.md`).

## Running a measurement

The runs in `results/` were made on an A100 with the notebook `colab/stage1_a100.ipynb` (see
`colab/README.md`). The equivalent commands, for the instruct model:

```bash
python scripts/run_stage1.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct --prompts data/math500_pool150_seed20261103.jsonl --out runs/q15b_instruct --temperature 0.7 --top-p 1.0 --top-k 0 --min-p 0.0 --max-new-tokens 2048 --max-model-len 4096 --interval 128 --k0 48 --m-base 4 --k 8 --n-select 8 --max-prompts 50 --chunk-prompts 25 --seed 0
python scripts/add_entropy.py --backend torch --device cuda --model Qwen/Qwen2.5-1.5B-Instruct --temperature 0.7 --in runs/q15b_instruct/trajectories.jsonl --out runs/q15b_instruct/trajectories_entropy.jsonl
python scripts/analyze_stage1.py --trajectories runs/q15b_instruct/trajectories_entropy.jsonl --out runs/q15b_instruct/analysis --abs-bin-width 128
python scripts/run_steering.py --backend torch --model Qwen/Qwen2.5-1.5B-Instruct --temperature 0.7 --trajectories runs/q15b_instruct/trajectories.jsonl --out runs/q15b_instruct/steering.jsonl --stride 16 --swaps 4 --window 64
python scripts/analyze_steering.py --steering runs/q15b_instruct/steering.jsonl --out runs/q15b_instruct/steering_analysis
```

On Apple silicon pass `--backend mlx` to `add_entropy.py` and `run_steering.py`
(`run_stage1.py` selects it automatically on macOS/arm64). For base models add `--no-chat-template`. Set the sampling
settings to those of the policy you want to describe, and use the same temperature in every step.
Runs resume after interruption; `--max-chunks N` stops after N chunks (exit code 3 if prompts remain).
`scripts/probe_lengths.py` measures response lengths to size a run (cost grows with the square of
the length; methods §6).

Check the conservation ratio before trusting a run. If its CI excludes 1, continuations are not
on-policy (chat template, tokenisation, sampling settings, length cap) or the reward is not
deterministic.

## Outputs

| File | Contents |
|---|---|
| `manifest.json` | arguments, configuration, versions, device, model revision |
| `selected_prompts.jsonl` | prompts kept by the pass-rate filter, with their selection pass rate |
| `trajectories.jsonl` | one record per base response: checkpoints, Monte Carlo means and variances, `k`, reward, token ids |
| `throughput.json` | generated tokens, time, tokens per second |
| `analysis/` | `summary.txt`, `report.json` (curves with CIs, strata, positional credit, regressions), `resolution_curve.png` |
| `steering.jsonl`, `steering_analysis/` | per-position steering influence; binned curves, lag profile, regressions |

## Validation status

| Component | Status |
|---|---|
| Estimators and identities | exact tests against enumeration (tolerance 1e-12) |
| Full pipeline (toy backend) | recovers exact shares within 0.006 and exact `F` within 0.0004 |
| vLLM backend | run on a Colab A100 (vLLM 0.33.0); per-prompt pass rates agree with the MLX backend on the same 50 prompts (difference −0.007 ± 0.011) |
| MLX backend | both generation paths match the exactly enumerated response distribution of a tiny model; batched logits match a cache-free forward pass to 1e-4; an output-ordering bug that biased the earlier Mac pilots was found and fixed, with regression tests (corrections 21) |
| Entropy pass | torch and MLX versions tested against direct computation |
| Steering | torch and MLX versions tested against naive forward passes |
| Reward grader | Math-Verify 0.9.0; unit tests for boxed extraction, `</think>` handling, equivalences |

## Layout

```
tcredit/records.py       Trajectory record + validation
tcredit/checkpoints.py   stopping-time checkpoint rules
tcredit/estimator.py     debiased increments, resolution curve, bootstrap, positional credit
tcredit/measure.py       selection + measurement loop (backend-agnostic, resumable)
tcredit/backends.py      vLLM backend (on-policy continuation from token ids)
tcredit/backends_mlx.py  MLX backend (Apple silicon), same contract
tcredit/steering.py      H1 steering influence (MLX and torch) and its analysis
tcredit/entropy.py       teacher-forced per-token entropy (torch and MLX)
tcredit/rewards.py       deterministic math reward
tcredit/analyze.py       fixed-effects regression, strata, report
tcredit/toy.py           exactly solvable toy policy + exact expectations + toy backend
tcredit/plots.py         figure
scripts/                 run_stage1, analyze_stage1, add_entropy, run_steering, analyze_steering,
                         prepare_data, make_pools, probe_lengths, bench_mlx
colab/                   A100 notebook (generated by make_notebook.py) and bundle script
tests/                   exact identities, end-to-end toy, units, backends, entropy, steering
results/                 compact outputs of the completed Stage 1 runs
```
