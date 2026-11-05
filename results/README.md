# Results

Compact outputs of the valid Stage 1 runs (MATH-500, 1.5B models, vLLM 0.33.0 on one NVIDIA
A100-SXM4-40GB, Colab, 2026-11-04). Each folder holds the write-up (`RESULTS.md`), the run
manifest, throughput, the selected prompt ids with their selection pass rates, the Stage 1
analysis (`analysis/`: summary, full report, figure) and the steering analysis
(`steering_analysis/`). Raw per-sample files (trajectories with token ids, per-token entropies,
steering records; up to 25 MB each) are not in the repository; they are intended for release as a
dataset with the paper.

| Folder | Model | Prompts | Conservation (95% CI) | F, all segments | F, excl. final segment | Position effect beyond entropy | Early steering (H1) |
|---|---|---|---|---|---|---|---|
| `stage1/qwen2.5-1.5b-base` | Qwen/Qwen2.5-1.5B, raw prompt | 50 | 1.07 [0.85, 1.33] | +0.02 [−0.04, +0.09] | **+0.12 [+0.07, +0.18]** | none | first 16 tokens ~6× later ones; explained by entropy |
| `stage1/qwen2.5-1.5b-instruct` | Qwen/Qwen2.5-1.5B-Instruct | 50 | 0.94 [0.65, 1.25] | −0.08 [−0.15, +0.00] | −0.00 [−0.08, +0.08] | none (slightly later) | first ~128 tokens ~1.5×; explained by entropy |
| `stage1/r1-distill-qwen-1.5b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 30 | 1.23 [0.89, 1.63] | −0.03 [−0.14, +0.11] | −0.01 [−0.13, +0.12] | none | flat; the formal H1 pass is driven by the first token only (see its RESULTS.md) |

`probes/r1_distill_length_probe.json`: response-length probe used to size the R1 run.

Model revisions: the GPU manifests record the model name only (vLLM loaded `main`). On
2026-11-04, `main` resolved to Qwen2.5-1.5B-Instruct `989aa79`, Qwen2.5-1.5B `8faed76`,
R1-Distill-Qwen-1.5B `ad9f0ae` (checked with the Hub API after the runs, not recorded by them).
`scripts/run_stage1.py` now records the commit for vLLM runs.

Not included: the earlier pilots run on Apple silicon (MLX backend). Their base trajectories were
length-biased by a backend bug, so their results are invalid (`docs/corrections.md`, item 21).
The R1 run was completed in two Colab sessions after a runtime disconnect; the resumed file
contains the first session's records unchanged.

To rebuild a run's prompt file: `python scripts/make_pools.py --selected
results/stage1/<run>/selected_ids.json <out.jsonl>` (after `scripts/prepare_data.py`).
