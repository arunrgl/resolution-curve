# Handoff: state of the project

Note on paths: this log refers to local run folders under `runs/`, which are not in the
repository (see `.gitignore`). The valid runs' compact outputs are in `results/stage1/`:
`runs/colab_A1`/`A2` → `qwen2.5-1.5b-instruct`, `runs/colab_E1`/`E2` → `qwen2.5-1.5b-base`,
`runs/colab_C`/`C3` → `r1-distill-qwen-1.5b`.

Read this first if you are a new session picking up the work.

## Goal
Test whether earlier tokens in an autoregressive response carry systematically more of the
"credit" for the final reward, and whether exploiting that improves published RL and SFT results.
The work is planned for an arXiv paper and a blog post.

- **H1**: earlier tokens shift the distribution of later tokens more.
- **H2**: earlier tokens resolve more of the reward's uncertainty. This is measured as the share
  of `Var(R | prompt)` resolved by position (docs/methods.md).
- **H3**: credit assignment that uses this improves RL/SFT sample efficiency.

The study runs in three stages: Stage 1 measures H2 (code here). Stage 2 runs RL arms and
Stage 3 runs SFT arms; both are not implemented yet and are gated on the Stage 1 result
(methods §8).

## Done
- Stage 1 pipeline: estimators, measurement loop, analysis, figure, and CLI scripts.
- 68 tests passing, checked against an exactly solvable toy model.
- `docs/methods.md`: proofs, design, and the decision rule, fixed before any data.
- `docs/corrections.md`: errors found in the original plan.
- `docs/references.md`: citation status. It was checked by web search only, because arxiv.org
  could not be reached; it must be re-checked against arXiv before publishing.

## Mac session (2026-11-03): MLX backend
Machine: Apple M1 Pro, 32 GB unified memory (MLX max recommended working set 26.8 GB),
macOS 26.6.2, Python 3.12.15, mlx 0.32.4, mlx-lm 0.32.2, transformers 5.22.1, datasets 5.1.0.

- `tcredit/backends_mlx.py`: `MLXBackend`, same contract as `VLLMBackend`; `--backend mlx` in
  `scripts/run_stage1.py` (default on macOS/arm64); versions, model commit, dtype and stop ids
  go into `manifest.json`, throughput into `throughput.json`. `scripts/bench_mlx.py` benchmarks.
- Tests (`tests/test_mlx_backend.py`, 15): own sampler vs independent numpy reference
  (chi-square); simple and batched paths vs the *exactly enumerated* response distribution of a
  tiny random Qwen2 model; teacher-forced batched-path logits vs a cache-free forward (1e-4);
  budget/stop/`finished` semantics; real Qwen2.5-0.5B: unquantized, prompt ids equal HF's,
  prefill vs incremental, 2-prompt `measure_chunk`. Suite: 83 passed, 1 skipped (torch entropy).
- Findings (docs/corrections.md 15–17): mlx-lm's sampler filters before tempering (we use our
  own); its chat wrapper injects a thinking kwarg (we use the HF template); bf16 prefill vs
  decode TV up to ~0.03 on Qwen2.5-0.5B, < 1e-5 in fp32 (`--mlx-dtype float32`).
- Throughput (bf16, decode ceiling at ~600-1500-token context, B≈96): 0.5B ≈ 1,400–1,500
  generated tok/s; 1.5B ≈ 500–600. Real 2-prompt MATH-500 run with 1.5B (k0 8, M 2, k 8,
  L 64, budget 2048): 194,607 tokens in 591 s = 329 tok/s, peak MLX memory 10.7 GB. The
  2-prompt analysis numbers are not meaningful (one prompt had zero variance).
- Fixed along the way: MLX's allocator cache grew to 24 GB resident and swapped; now capped
  (`TCREDIT_MLX_CACHE_GB`, default 4). Finished rows are compacted in bulk (BatchKVCache
  filter/extend copy the whole cache).
- Unexplained: one benchmark (1.5B fp32 continuations) ran at 97 tok/s; it ran before the
  cache cap and probably swapped, but it was not re-run.

- H1 steering measurement (methods §11, pre-registered before any data): `tcredit/steering.py`,
  `scripts/run_steering.py`, `scripts/analyze_steering.py`; tested against naive forward passes.
- Citations checked against arXiv abstracts; official BibTeX in `docs/references.bib`
  (corrections 18–20: the Temporal Scheduling summary was wrong; OAR is close related work).
- Pilot A launched: Qwen2.5-0.5B-Instruct, bf16, T=0.7 no filters, 30 prompts selected from a
  fixed random 100-prompt pool (`data/math500_pool100_seed20261103.jsonl`), M=4, k=8, k0=32,
  L=128, budget 2048. Output: `runs/pilotA_q05b/`. **Finished; results in
  `runs/pilotA_q05b/RESULTS.md`.** Conservation 1.29 [0.98, 1.67] (passes, wide); F = +0.02
  [−0.09, +0.12]; excluding the final segment +0.15 [+0.03, +0.24]; U-shaped shares; position
  not significant once entropy is controlled, for both H2 and H1 (steering).
- `scripts/add_entropy.py --backend mlx` (Apple silicon entropy pass), tested.
- Pilot B finished (2026-11-04): Qwen2.5-1.5B-Instruct, 50 prompts from a 150-prompt pool,
  k0=48, M=4, k=8, L=128, budget 2048; 8.53M tokens in 6 h 38 min. Results in
  `runs/pilotB_q15b/RESULTS.md`. Conservation 1.21 [0.90, 1.58] (passes); F = +0.001
  [−0.08, +0.09]; near-flat profile; position not significant in any test. Steering falls with
  position but is fully accounted for by entropy (as in pilot A).
- 2026-11-04: moved compute to a Colab A100 (80 GB) at the user's request; the Mac queue was
  stopped (runs C/D unfinished, both resumable). `colab/` holds the notebook (no Google Drive: upload the code zip, download results_<step>.zip after each step)
  (`stage1_a100.ipynb`), the code bundle and instructions. Step A is the first-ever execution
  of the vLLM backend and must pass the conservation gate before anything else runs there.
  Added: vLLM throughput counters, `scripts/probe_lengths.py`, a torch steering implementation
  (`steering_for_trajectory_torch`, `run_steering.py --backend torch`). torch 2.15.0 (CPU) is
  now in `.venv`; `tests/test_entropy.py` ran for the first time and passes. Suite: 94 passed.
- **2026-11-04: MLX ordering bug (corrections 21) invalidates the Mac pilots A and B.** Fixed and
  regression-tested; `measure_chunk` now also shuffles the base request's samples. The first valid
  result is the GPU run `runs/colab_A1` (RESULTS.md): conservation 0.94 [0.65, 1.25]; F = −0.077
  [−0.149, +0.004]; mildly back-loaded; position coefficient with entropy *positive* and significant.
  The GPU and Mac backends agree on per-prompt pass rates. Colab code zip rebuilt with the fix.
  Steering on the GPU run (results_A2): early tokens (first ~128) steer more, fully accounted for
  by entropy; H1-primary not supported.
- R1-Distill-Qwen-1.5B probe (`runs/colab_B`, T=0.6, top_p 1, budget 16,384, 80 samples): mean
  length 4,615, median 2,146, p90 16,024, 90% finished, 76% correct. Notebook updated: step E
  (base model, Qwen2.5-1.5B raw prompt) then step C (R1-Distill, 30 prompts, k0 32, M 4, k 8,
  L 512; ~50M tokens, est. 2–4 h). Steps A/B switched off (done).
- Base model result (`runs/colab_E1/RESULTS.md`): conservation 1.07 [0.85, 1.33]; F = +0.02
  (n.s.); excluding the final segment +0.12 [+0.07, +0.18]; position effect significant alone but
  not with entropy. H2-secondary holds; H2-primary and H2-robust do not. Steering: first 16
  tokens steer ~6× more than later ones, fully accounted for by entropy (0.69 vs ~0.1 nats).
- R1-Distill step C (2026-11-04): Colab runtime disconnected after chunk 2 of 3 (~5 h; ~2k tok/s,
  far below the 8k assumed). Notebook now runs one chunk per call (`run_stage1.py --max-chunks 1`,
  exit code 3 = prompts remain) and offers `results_C1_r1_<done>of30.zip` after each chunk.
  The first 20 prompts were recovered from the old runtime (`runs/colab_C_partial`, all 80 records
  validate); resume package `colab/results_C1_r1_20of30.zip`. Preliminary 20-prompt analysis:
  conservation 1.14 [0.74, 1.60], F = −0.02 [−0.15, +0.15]. Mean length 7,648 (median 5,508).
  R1 steering stride set to 128 (cost). Colab balance at that point: 54 units, 6.73/h.
- R1-Distill complete (`runs/colab_C/RESULTS.md`): 30 prompts, conservation 1.23 [0.89, 1.63],
  F = −0.03 [−0.14, +0.11], flat profile; no position or entropy effect on resolution. Steering
  (after an OOM fix): pre-registered H1 test formally met, but driven entirely by the first token
  (t = 1); excluding it, no position effect (see RESULTS.md sensitivity analysis).

### Stage 1 summary so far (valid GPU runs, MATH-500, 1.5B models)
| Model | conservation | F (all) | F (excl. final) | position effect beyond entropy | early steering |
|---|---|---|---|---|---|
| Qwen2.5-1.5B base | 1.07 | +0.02 n.s. | **+0.12 sig.** | none | 6× in first 16 tokens, = entropy |
| Qwen2.5-1.5B-Instruct | 0.94 | −0.08 n.s. | −0.00 | none (slightly later) | 1.5× in first 128, = entropy |
| R1-Distill-Qwen-1.5B | 1.23 | −0.03 n.s. | −0.01 | none | flat; formal H1 pass driven by token 1 only |
- (Superseded) Queue started 2026-11-04 ~04:30 (`runs/queue_20261104.sh`, log `runs/queue_20261104.log`):
  C = pilot B re-run in fp32 on the same 50 prompts (`runs/pilotB_q15b_fp32`, max-batch 64),
  then D = Qwen2.5-1.5B *base*, raw prompt (`runs/pilotD_q15b_base`), with entropy, analysis and
  steering. R1-Distill-Qwen-1.5B (run R) not started: needs a length measurement and sizing.
- Stage 2 draft: `docs/stage2_design.md` (entropy-calibrated credit β(h), fork-placed Monte
  Carlo advantages, baselines incl. OAR and top-20% entropy). Not frozen; compute undecided.
- Open: the conservation point estimate is above 1 in both pilots (1.29, 1.21); CIs contain 1
  but it is worth checking with a larger run (possible small bf16 off-policy effect, corrections 17).

## Not done / not verified
1. No real-model *pilot* yet (only the 2-prompt plumbing run above).
2. vLLM on CUDA: API checked against 0.33.0; never executed.
3. MLX backend done (above). Not checked: that vLLM 0.33.0 applies top-k/top-p/min-p in the
   same order as our MLX sampler; compare backends only with those filters off.
4. `tests/test_entropy.py` now runs (CPU); not yet on CUDA.
5. Stage 2 and Stage 3.

## Next steps on a Mac
1. `python -m pytest -q tests`: expect 94 passed (with torch in .venv).
2. Decide decoding settings for the pilot (Qwen2.5's shipped generation_config uses
   repetition_penalty 1.1, which the contract excludes).
3. Size the pilot from the measured throughput (token formula in methods §6). Prefer short-answer
   models first, such as a Qwen2.5 1.5B instruct model on MATH-500.
4. The pilot is valid only if the conservation ratio's 95% CI contains 1.
5. Use full-precision (fp16/bf16) weights. A quantized model is a different policy from the
   published checkpoint.
