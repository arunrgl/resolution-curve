# GPU run E1: Qwen2.5-1.5B (base), raw prompt, vLLM 0.33.0 on Colab (2026-11-04)

Setup: NVIDIA A100-SXM4-40GB; bf16; no chat template; user text
"{problem}\nPlease reason step by step, and put your final answer within \boxed{}."; T = 0.7, no
top-p/top-k/min-p; budget 2048; 50 prompts selected (pass rate in [1/8, 7/8], 8 rollouts) from the
150-prompt pool; k0 = 48, M = 4, k = 8, checkpoints every 128 tokens. 5.24M generated tokens in
361 s (14,511 tok/s, includes selection). Base responses: mean 610 tokens (median 517), 97%
finished. Base-trajectory reward 0.52 vs mean V̂_0 0.53 (no split bias).

## Validity
Conservation ratio 1.070, 95% CI [0.846, 1.329]: passes.

## H2
- F (all segments) = +0.023 [−0.039, +0.093]: not significant.
- F excluding the final segment = **+0.123 [+0.068, +0.180]: front-loaded**.
- Final segment alone: 0.212 [0.113, 0.307].
- Per-bin shares: 0.122 0.156 0.102 0.074 0.079 0.094 0.080 0.059 0.088 0.144 (high early, dip,
  rise at the end).
- Segment regression with prompt fixed effects: position **−1.1e-4 [−2.0e-4, −3.3e-5]** alone
  (earlier segments resolve more per token); **+1.6e-4 [−8.5e-5, +5.0e-4] with entropy** (not
  significant).
- H2-primary (F > 0 over all segments): not supported. H2-secondary (excluding the final
  segment): supported. H2-robust (position survives the entropy control): not supported.

## H1: steering (results_E2; torch bf16, stride 16, 4 swaps, window 64; 6,904 full-window positions)
- Mean per-token influence S/W: 0.071 in the first tenth, then 0.037, 0.033, … down to ~0.02.
- By absolute position: **0.161 for the first 16 tokens**, 0.051 for 16–64, 0.043 for 64–128,
  ~0.026–0.028 for 128–512, ~0.013–0.016 after. Mean entropy: 0.69 nats in the first 16 tokens,
  0.21, 0.17, 0.12, 0.10, 0.08, 0.06.
- Regression: position alone u +1.2e-2 [+6.2e-3, +1.8e-2], log t −3.4e-2 [−4.3e-2, −2.6e-2];
  with entropy, u +2.7e-4 [−4.6e-3, +4.8e-3], log t −1.2e-3 [−8.3e-3, +5.8e-3], entropy +0.073
  [+0.068, +0.077].
- H1-primary not supported: the very first tokens steer far more, exactly as much as their much
  higher entropy predicts.
- Lag profile: mean KL 1.25 at lag 1, 0.057 at lag 4, 0.009 at lag 16, 0.0007 at lag 64.
