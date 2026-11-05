# Stage 2 design (RL credit assignment): DRAFT for review, 2026-11-04

Status: proposal. Nothing here is implemented. Arms and decision rules must be frozen before
any Stage 2 training run, as Stage 1's were (methods §8).

## 1. What Stage 1 changed

The original Stage 2 tested *positional* credit: weight each token's policy-gradient term by a
function of its position. Methods §8 makes the positional arm conditional on H2-primary and
H2-robust holding on at least one model family. In both pilots so far (Qwen2.5-0.5B/1.5B
Instruct, MATH-500, T = 0.7):

- the outcome is not front-loaded overall (F ≈ 0), and
- position adds nothing once token entropy is controlled, both for outcome resolution (H2) and
  for steering of later tokens (H1). Entropy is strongly positive in every model.

Base and long-reasoning models are still to be measured (runs D and R). Unless one of them
satisfies the §8 rule, the positional arm is dropped as pre-registered, and Stage 2 asks the
question the data point to: **does credit concentrated at high-uncertainty ("forking") tokens
improve RL, and does a principled version beat existing heuristics?**

## 2. The principled weight: least-squares credit given features known before the token

Methods §5 (Prop. 3b) shows that for any event known before token t is sampled, the best
*proportional* predictor of the value jump `Δ_t` from the sequence advantage `A` is
`β = E[Δ_t² 1{·}] / E[A² 1{·}]`. The proof only needs the conditioning to be
`F_{t−1}`-measurable, so it extends to any feature `z_t` computed from the prefix, in
particular the policy entropy `h_t = H(π(·|x, y_<t))`:

```
β(z) = E[Δ_t² | z_t = z, N ≥ t] / E[A² | z_t = z, N ≥ t]
```

- position: `z_t = t` (the original arm);
- entropy: `z_t = h_t` (new);
- both: `z_t = (t, h_t)`.

`β(z)` is estimable from Stage 1 data, with one modelling assumption. Stage 1 identifies the
debiased jump only per *segment* (methods §4), not per token. Assume the per-token resolution
density is a function of the features, `E[Δ_t² | z_t] = g(z_t)`, with `g` in a small basis
(for example piecewise-constant in entropy bins). Then `E[d̂_j] = Σ_{t∈segment j} g(z_t)` is
linear in the basis coefficients, so they can be fitted by regressing segment `d̂_j` on
segment sums of basis features. That regression is unbiased because `d̂_j` is (Prop. 2). The
denominator uses the debiased `E[A²]` as in §5. This gives a *calibrated* entropy weight (how much outcome variance a token
of entropy h actually resolves on this policy) instead of a hand-picked top-20% rule. Caveat
(methods §5): any per-token weight makes the gradient a biased estimate of `∇E[R]`; it is a
heuristic re-weighting and has to earn its place empirically.

## 3. Arms

| Arm | Credit for token t | Gradient | Notes |
|---|---|---|---|
| A0 GRPO baseline | `A` (group-normalised), Dr. GRPO normalisation | biased by clipping only | the comparison everything is against |
| A1 positional β(t) | `β(t)·A` | biased | only if §8 go/no-go is met on some family |
| A2 entropy β(h) | `β(h_t)·A`, β fitted on Stage 1 data of the starting policy | biased | new; the main hypothesis |
| A3 top-20% entropy | `1{h_t in top 20%}·A` | biased | Beyond the 80/20 Rule (2506.01939) as an external baseline |
| A4 fork-placed Monte Carlo advantages | `V̂_j − V̂_{j−1}` per segment, checkpoints placed online where `h_t > τ` | unbiased (checkpoints are stopping times) | VinePPO-style, branching at forks instead of fixed intervals |
| A5 OAR | as published (2601.07408) | — | include if its code is available; otherwise reimplement OAR-G and say so |

A4 is the clean version of the idea: it puts the compute where uncertainty is and keeps the
gradient unbiased. A2 is the cheap version. A1 stays only as the original hypothesis.

## 4. Protocol (to freeze before running)

- **Policy:** start from the model measured in Stage 1, with the same decoding settings.
- **Data:** MATH train split for training, never MATH-500. Evaluation on MATH-500, AIME24/25,
  AMC23, with fixed sampling settings and n ≥ 4 samples per problem.
- **Matched budgets:** report results at equal numbers of policy samples, and separately at
  equal wall-clock. A4 uses more samples per update, so compare it at matched samples.
- **Seeds:** at least 3 per arm.
- **Primary metric:** MATH-500 accuracy at a fixed sample budget. Analysis: paired bootstrap
  over problems × seeds (corrections 8).
- **Decision rule:** an arm "helps" if its gain over A0 has a 95% CI excluding 0 on the primary
  metric and the sign agrees on at least two of the other benchmarks.
- **Weight refresh:** `β(h)` is fitted once from the starting policy's Stage 1 data, with one
  fixed constant normalising mean weight to 1 over Stage 1 tokens (corrections 5). A refreshed
  variant is a secondary analysis.

## 5. Compute: needs a decision

RL training does not fit comfortably on this Mac. Full fine-tuning of a 1.5B model with Adam
needs roughly 16 bytes/parameter (≈24 GB) before activations and generation; 0.5B (≈8 GB) fits
but would be slow. Options:
1. 0.5B full fine-tuning on the Mac (MLX), small scale, as a feasibility study.
2. LoRA on the Mac. Cheaper, but changes what is being compared; all arms would need it.
3. Rented GPUs for the real runs (the existing vLLM backend targets CUDA). The cost and the
   provider are the user's decision.

## 6. Stage 3 (SFT) in one paragraph

Same logic for SFT. Weight the token-level cross-entropy by `β(z_t)` fitted from the teacher's
Stage 1 data (entropy, position, both), against uniform SFT and UPFT-style prefix-only training
(2503.02875), at matched tokens. Same evaluation and decision rule.

## 7. Open items before freezing

- Results of runs C (fp32 validity check), D (base model) and R (R1-Distill, not yet started).
- Whether OAR and the 80/20 paper released code, and under what licences.
- Read Temporal Scheduling (2605.25381) in full (corrections 18).
