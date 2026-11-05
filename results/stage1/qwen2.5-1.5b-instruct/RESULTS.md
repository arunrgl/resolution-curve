# GPU run A1: Qwen2.5-1.5B-Instruct, vLLM 0.33.0 on Colab (2026-11-04)

First valid Stage 1 measurement (the Mac pilots are invalid, corrections 21).

Setup: NVIDIA A100-SXM4-40GB, torch 2.14.0+cu130, Python 3.13.16; bf16; T = 0.7, no
top-p/top-k/min-p; budget 2048; the same 50 prompts as Mac pilot B; k0 = 48, M = 4, k = 8,
checkpoints every 128 tokens. 4.49M generated tokens in 319 s (14,084 tok/s; selection reused).
Produced with the code before the corrections-21 shuffle; vLLM's sample order shows no bias
(base reward 0.54 vs V̂_0 0.58; 97% finished).

## Backend agreement (same prompts, Mac vs GPU)
Identical prompt token ids. Per-prompt pass rates from 48 rollouts each: Mac 0.586, GPU 0.579,
paired difference −0.007 (SE 0.011); per-prompt z-scores mean −0.12, SD 1.01. The two backends
sample the same policy, as far as pass rates can show.

## Validity
Conservation ratio 0.944, 95% CI [0.652, 1.254]: passes. Wide.

## H2
- F (all segments) = −0.077 [−0.149, +0.004]: if anything back-loaded; CI just includes 0.
- F excluding the final segment = −0.001 [−0.080, +0.077].
- Final segment alone: 0.180 [0.093, 0.265].
- Per-bin shares: 0.091 0.084 0.063 0.055 0.073 0.121 0.083 0.122 0.132 0.176 (rising).
- Segment regression with prompt fixed effects: position +5.4e-5 [−1.4e-5, +1.3e-4] alone;
  **+8.8e-5 [+1.8e-5, +1.7e-4] with entropy**, i.e. later segments resolve more per token at
  equal entropy.
- H2-primary (F > 0): not supported. H2-robust (negative position coefficient): not supported;
  the estimate has the opposite sign.

## H1: steering (results_A2; torch bf16, stride 16, 4 swaps, window 64; 7,150 full-window positions)
- Mean per-token influence S/W by relative position: 0.042 in the first tenth, then roughly
  flat at 0.020–0.029.
- By absolute position: ~0.040 for tokens 1–128, ~0.024–0.028 for 128–1024 (0.005 beyond 1024,
  few positions). Mean entropy has the same shape (0.15–0.18 nats in the first 128 tokens,
  0.08–0.11 later).
- Regression: position alone u +1.4e-3 [−3.6e-3, +5.6e-3], log t −5.9e-3 [−1.2e-2, −1.1e-4]
  (very early tokens steer more); with entropy, u +6.5e-4 [−2.2e-3, +3.1e-3], log t +2.9e-4
  [−3.4e-3, +4.0e-3], entropy +0.069 [+0.064, +0.074].
- H1-primary not supported: early tokens steer more, but only as much as their higher entropy
  predicts.
- Lag profile: mean KL 1.17 at lag 1, 0.038 at lag 4, 0.006 at lag 16, 0.0008 at lag 64.

## Reading
For Qwen2.5-1.5B-Instruct on MATH-500 at T = 0.7: the outcome is not decided early (slightly
late, if anything), and the extra steering power of the first ~100 tokens is what their higher
uncertainty predicts. Neither measurement supports a positional effect beyond entropy.
