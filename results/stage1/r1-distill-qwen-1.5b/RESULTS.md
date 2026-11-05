# GPU run C: DeepSeek-R1-Distill-Qwen-1.5B, vLLM 0.33.0 on Colab (2026-11-04)

Setup: NVIDIA A100-SXM4-40GB; bf16; chat template (no system prompt), user text
"{problem}\nPlease reason step by step, and put your final answer within \boxed{}."; graded after
`</think>`. T = 0.6, no top-p/top-k/min-p (model card: T 0.6, top-p 0.95; filter dropped for
consistency with the other runs and an exact entropy control); budget 16,384. 30 prompts selected
from the 150-prompt pool; k0 = 32, M = 4, k = 8, checkpoints every 512 tokens. Run in two sessions
(runtime disconnect after 20 prompts; the 80 recovered records are byte-identical in the final
file). Last session: 32.6M tokens in 7,037 s (4,632 tok/s). Base responses: mean 7,886 tokens
(median 5,746), 84% finished; base reward 0.62 vs mean V̂_0 0.64.

## Validity
Conservation ratio 1.233, 95% CI [0.894, 1.628]: passes. Wide (30 prompts).

## H2
- F (all segments) = −0.032 [−0.145, +0.107]; excluding the final segment −0.011 [−0.128, +0.118].
- Final segment alone: 0.048 [0.000, 0.140] (small: the answer after `</think>` is short relative
  to the reasoning, and checkpoints are 512 tokens apart).
- Per-bin shares: 0.078 0.126 0.073 0.061 0.135 0.098 0.102 0.060 0.107 0.160: flat within noise.
- Segment regression with prompt fixed effects: position +2.1e-6 [−8.0e-6, +1.2e-5] alone,
  +2.7e-6 [−7.9e-6, +1.3e-5] with entropy. Entropy itself is not significant here either
  (ent_mean +6.1e-6 [−5.7e-6, +1.8e-5]).
- H2-primary, H2-secondary and H2-robust: not supported. The outcome is resolved uniformly along
  the response, as far as 30 prompts can tell.

## H1: steering (results_C3; torch bf16, **stride 128**, 4 swaps, window 64; 7,395 full-window positions)
- Mean per-token influence S/W by relative position: ~0.05 throughout (0.048–0.062), 0.030 in
  the last tenth. By absolute position: 0.011 at the first token (t = 1; the only grid point
  below 128), then 0.061, 0.061, 0.053, 0.050, 0.050, 0.048 for 128–512, …, 8192–16384.
- Mean entropy: 0.60 at t = 1, then 0.22–0.28 (relative bins 0.32 → 0.14).
- **Pre-registered regression** (u, entropy, log t): u −7.9e-3 [−1.1e-2, −4.1e-3], entropy +0.089
  [+0.084, +0.095], log t +1.6e-2 [+1.4e-2, +1.9e-2]. By the letter of methods §11, H1-primary is met
  (negative coefficient on t/N with the CI excluding 0).
- **Sensitivity (exploratory, not pre-registered):** the result is driven entirely by the 120 first-token
  points (t = 1, log t = 0, low influence despite high entropy). Checked on the data: R1 opens with
  "Okay" (90 of 120) or "Alright" (30 of 120), usually followed by ", so I have this…"; the first token is a
  high-entropy choice between interchangeable words that barely changes what follows.
  Excluding t = 1: u +1.2e-3 [−3.1e-3, +5.8e-3], log t −1.4e-3 [−6.3e-3, +3.6e-3]. Excluding the last
  tenth instead leaves the result unchanged (u −7.8e-3 [−1.2e-2, −4.0e-3]). Excluding both, with u and
  entropy only: u +2.4e-4 [−2.3e-3, +2.8e-3].
- Reading: apart from the first token, steering in R1-Distill does not depend on position; entropy
  explains it. The formal H1-primary "pass" reflects one leverage point and should be reported with
  this sensitivity analysis, not as support for the hypothesis.
- Lag profile: mean KL 2.28 at lag 1, 0.081 at lag 4, 0.010 at lag 16, 0.0008 at lag 64.
