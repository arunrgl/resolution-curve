# Stage 1 methods: where does a policy resolve its outcome?

This document defines what Stage 1 measures, proves the identities the estimators rely on, and
fixes the decision rules before any data are collected. Every identity marked *(tested)* is
checked exactly against a toy process by `tests/test_exact.py` (tolerance 1e-12).

## 1. Setup and notation

- `x`: a prompt. `π`: the policy **together with its decoding settings** (temperature, top-p, …).
  Everything below characterises *that* sampling distribution.
- Response tokens `y_1, y_2, …, y_N`. If the response finished, `y_N` is the stop (EOS) token; if
  it hit the budget `T`, then `N = T`. In both cases `N ≤ T`.
- `F_t`: the information in `(x, y_1, …, y_t)`. The length `N` is a stopping time, because whether
  the response has ended by step `t` is decided by `y_≤t`.
- `R = r(x, y) ∈ [0, 1]`: a **deterministic** verifiable reward of the finished response.
- `V_t = E[R | F_t]` for `t ≤ N`, and `V_t = R` for `t ≥ N`. `V_0 = E[R | x]`.
- `Δ_t = V_t − V_{t−1}` (zero for `t > N`), and the sequence advantage `A = R − V_0`.

`(V_t)_{t=0..T}` is a bounded martingale (Doob's martingale of `R`).

## 2. The variance decomposition

**Theorem 1.** Let `0 = τ_0 ≤ τ_1 ≤ … ≤ τ_J = N` be stopping times, and let
`D_j = V_{τ_j} − V_{τ_{j−1}}`. Then

```
Var(R | x) = Σ_j E[D_j² | x].                                                  (1)
```

*Proof.* The stopping times are bounded by `T`, so by optional stopping `(V_{τ_j})_j` is a
martingale with respect to `(F_{τ_j})_j`. Therefore `E[D_j | F_{τ_{j−1}}] = 0`, and for `i < j`,
`E[D_i D_j] = E[D_i · E[D_j | F_{τ_{j−1}}]] = 0`. Expanding the square,
`E[(R − V_0)²] = Σ_j E[D_j²]`, and `E[(R − V_0)² | x] = Var(R | x)` because `V_0 = E[R | x]`. ∎
*(tested: per token, and for both checkpoint rules below)*

The quantity `E[D_j²]` is therefore the share of the outcome's variance that gets **resolved**
inside segment `j`. These shares add up exactly to the total.

**Checkpoints must be stopping times.** The pipeline places checkpoints in one of two ways:
- **fixed:** every `L` tokens, i.e. `τ_j = min(jL, N)`;
- **delimiter:** after any token that completes a delimiter such as a blank line, at least
  `min_gap` tokens after the previous checkpoint. This is decided online from the prefix alone.

Cutting at fixed *fractions of the final length* (`τ_j = round(jN/J)`) is **not** a stopping time,
because it uses `N`. Then (1) fails. On a toy process where successful responses stop sooner, the
error reaches about 10% of `Var(R|x)`, and its sign depends on the process
(`test_relative_position_checkpoints_break_the_decomposition`). Relative position is used only
to **bin** segments after the fact, never to place checkpoints.

## 3. Estimation

For each selected prompt:
- `k_0` rollouts from the prompt give `V̂_0` and the unbiased sample variance `s_0²`, which estimates `Var(R|x)`;
- `M` independent base trajectories are sampled;
- at each interior checkpoint `τ_j` of each base trajectory, `k` continuations sampled from the
  prefix `y_≤τ_j` give `V̂_j` and `s_j²`. Each continuation's budget is `T − τ_j`, so its total
  length cap matches the base trajectory's.

All of these rollouts are mutually independent given the base trajectory.

**Proposition 2 (debiasing).** Given the base trajectory,

```
E[ (V̂_j − V̂_{j−1})² − s_j²/k_j − s_{j−1}²/k_{j−1} ] = D_j²,                   (2)
```

where the terminal value is exact (`V̂_J = R`, no noise term). For 0/1 rewards,
`s²/k = V̂(1 − V̂)/(k − 1)`.

*Proof.* Write `V̂_j = V_{τ_j} + e_j`, with independent, zero-mean errors of variance
`σ_j²/k_j`. Then `E[(D_j + e_j − e_{j−1})²] = D_j² + σ_j²/k_j + σ_{j−1}²/k_{j−1}`, and
`E[s_j²] = σ_j²`. ∎ *(tested: 400k replications; the uncorrected squares are biased upward)*

Call the left-hand side of (2) `d̂_j`. It is unbiased for `D_j²` and can be negative. It should
**not** be truncated at zero, because truncating reintroduces bias.

**Conservation check (built-in validity test).** By (1) and (2),
`E[Σ_j d̂_j | x] = Var(R | x)`, and `E[s_0²] = Var(R | x)` from independent rollouts. The pooled
ratio `Σ_x mean_traj(Σ_j d̂_j) / Σ_x s_0²` must therefore be about 1. If its 95% confidence
interval excludes 1, the run is **invalid**. Typical causes are continuations that are not
on-policy (a chat-template or re-tokenisation mismatch, different sampling settings, penalties
that treat the prefix differently), a reward that is not deterministic, or inconsistent length
capping. On the toy process the ratio is 1.004 with debiasing and 1.53 without.

## 4. Estimand for H2: the resolution profile

Segment `j` covers tokens `τ_{j−1}+1 … τ_j`. Its mass is spread uniformly over those tokens, and
from there over relative-position bins `b` (equal width on `[0, 1]`, by `t/N`). The **pooled
share** for bin `b` is

```
m_b = Σ_x E[ Σ_j D_j² w_jb | x ]  /  Σ_x Var(R | x),        Σ_b m_b = 1,
```

where `w_jb` is the fraction of segment `j`'s tokens that fall in bin `b`. Pooling weights each
prompt by `Var(R|x)`, so `m_b` is the share of the total outcome variance across the prompt
population that is resolved in bin `b`. It is estimated as a ratio of sums using `d̂`, which is
consistent and stable even when individual prompts have little variance.

- Cumulative curve `C(u)` at the bin edges, with `C(0) = 0` and `C(1) = 1`.
- **Front-loading index** `F = ∫_0^1 (C(u) − u) du ∈ [−½, ½]` (trapezoid rule, which is exact for
  piecewise-linear `C`). `F > 0` means variance is resolved earlier than uniformly.
- First-half excess `C(½) − ½`, and the share resolved in each response's **final segment**.

*What is identified:* the segment totals. How mass is distributed **within** a segment is a
presentation choice, so finer checkpoints give finer resolution.

**Squared vs. absolute effects.** The hypothesis was first stated in terms of the "average
absolute effect" `E|Δ_t|`. We pre-register the squared form `E[Δ_t²]` as primary, because it
alone has an exact decomposition (1) and an unbiased Monte Carlo estimator (2). `|·|` of a noisy
mean has no unbiased estimator. Front-loading in one measure does not logically imply
front-loading in the other.

## 5. From the profile to per-token credit (bridge to Stage 2)

**Proposition 3.**
(a) For every `t`, `E[Δ_t A | x] = E[Δ_t² | x]`. Hence the least-squares predictor of `Δ_t` from
the sequence advantage alone is `c_t A` with `c_t = E[Δ_t² | x] / Var(R | x)`, and `Σ_t c_t = 1`.
Because `E[A|x] = E[Δ_t|x] = 0`, this is also the best affine predictor. Uniform credit
(GRPO-style broadcast) corresponds to a constant `c_t`. *(tested)*

(b) For a token known to exist (`N ≥ t`), the least-squares *proportional* predictor `β_t A` has

```
β_t = E[Δ_t² 1{N ≥ t} | x] / E[A² 1{N ≥ t} | x],
```

because `{N ≥ t}` is `F_{t−1}`-measurable, so all cross terms vanish. Conditional on `N ≥ t`, `A`
need not have mean zero, so the best *affine* predictor generally includes an intercept. `β_t` is
optimal among predictors of the form `w_t · A`, which is the form a GRPO-style per-token weight
takes.

*Proof of (a).* `A = Σ_s Δ_s`, and martingale differences are uncorrelated, so
`E[Δ_t A] = E[Δ_t²]`. Part (b) is the same argument restricted to the `F_{t−1}`-measurable event
`{N ≥ t}`. ∎

`positional_credit()` estimates `β` per absolute-position bin, pooled over prompts. The
denominator is debiased with the independent prompt rollouts:
`E[(R − V̂_0)²] = E[A²] + Var(V̂_0)`.

*Caveats:* `c_t` and `β_t` are per-prompt quantities, so a single profile applied to all prompts
is an approximation. GRPO normalises `A` by the group's standard deviation. That changes the
scale for each prompt but not the shape across positions. A per-token weight `w_t` multiplies
per-timestep terms of the policy gradient, so the update is **no longer an unbiased estimate**
of `∇E[R]`. It changes how states are weighted and must be treated as a heuristic. By contrast,
Monte Carlo per-segment advantages `V̂_{j} − V̂_{j−1}` (VinePPO-style) keep the gradient unbiased.

## 6. Sampling design and inference

- **Prompt selection** uses `n_select` *separate* rollouts and keeps prompts with pass rate in
  `[select_lo, select_hi]` (default `[1/8, 7/8]`). Those rollouts are then discarded, so the
  selection cannot bias `V̂_0` or `s_0²`.
- **Unit of inference:** the prompt. All confidence intervals come from a cluster bootstrap over
  prompts (percentile, 2000 replicates). With fewer than about 30 prompts the intervals are
  unreliable; treat such runs as pilots.
- **Sampling settings:** use the settings of the policy you want to describe (for RL, the rollout
  temperature). Prefer `top_p = 1`, `top_k = 0` and `min_p = 0`, so that the entropy control (§7)
  measures the actual sampling distribution.
- **Compute:** continuation rollouts = `P · M · J̄ · k`, where `J̄` is the mean number of interior
  checkpoints. Each continuation averages roughly `(N − τ)` tokens. Example pilot: `P = 50`,
  `M = 4`, `J̄ ≈ 8`, `k = 16` gives 25,600 continuations; with responses around 4k tokens that is
  roughly 50M generated tokens, plus `P(n_select + k_0 + M)` full-length rollouts. Most prefill
  work is served from vLLM's prefix cache. (On the MLX backend prefill is under 2% of the time.)

## 7. Controls

1. **Final segment.** Writing out the answer often settles the remaining variance mechanically,
   and in finished responses the final segment includes the decision to stop. Report `F` with and
   without the final segment, and the final segment's share on its own.
2. **Entropy.** Regress the per-token resolution density `d̂_j / |segment|` on the segment's
   relative midpoint, `log|segment|`, a final-segment indicator, and the segment's mean and max
   token entropy, with **prompt fixed effects** (demeaning within prompt). Confidence intervals
   come from the cluster bootstrap. Position "survives" if its coefficient stays negative with a
   CI that excludes 0 once entropy is included. If not, the effect is better described by
   high-entropy forking tokens than by position.
3. **Length strata** (descriptive only). Curves are recomputed within tertiles of `N`.
   Conditioning on `N` uses the future, so (1) does not hold inside a stratum. The shares are
   renormalised within each stratum and serve only as a robustness description.
4. **Model families.** Run the measurement on a base or zero-RL model, an SFT-distilled
   long-chain-of-thought model, and an RL-trained model, using the same prompts.

## 8. Pre-registered hypotheses and decision rule

- **H2-primary:** `F > 0` (all segments), with the 95% CI excluding 0.
- **H2-secondary:** `F > 0` excluding the final segment.
- **H2-robust:** the coefficient on relative position is `< 0` with the 95% CI excluding 0 in the
  regression **with** entropy covariates.
- **Validity gate:** the conservation-ratio CI contains 1. Otherwise the run is discarded.
- **Go/no-go for Stage 2 positional arms:** H2-primary and H2-robust both hold on at least one
  model family. If `F ≤ 0`, or position does not survive the entropy control, the positional
  version of the hypothesis is not supported. That result is still reportable.

## 9. Validation status

| Check | Result |
|---|---|
| Identity (1), per token and for both checkpoint rules, 3 toy processes | exact to 1e-12 |
| Checkpoints at fractions of final length | biased by about 10% of Var(R) on a toy process |
| Proposition 2 (debiasing), 400k replications | unbiased within 5 SE; raw squares biased upward |
| Proposition 3(a) | exact to 1e-12 |
| End-to-end on 10 toy prompts × 400 trajectories | max share error 0.006; F = −0.1571 vs exact −0.1575; conservation 1.005 |
| `β_t` recovery (early positions) | within 0.03 of exact |
| vLLM backend | checked against the vLLM 0.33.0 source and a stand-in module; **not executed on a GPU** |
| MLX backend | simple and batched paths match exact enumeration (tiny model, chi-square); batched logits = cache-free forward to 1e-4; fp32 prefill vs decode TV < 1e-5 on Qwen2.5-0.5B, bf16 up to ~0.03 |
| Entropy (transformers) | logit-row alignment tested; full torch test written, not run here |

## 10. Limitations

- The measurement describes a policy at specific decoding settings. Temperature changes the
  profile.
- Numerical differences between vLLM's prefill and decode kernels make continuations very
  slightly off-policy. The conservation ratio is the guard against this. Measured on MLX with
  Qwen2.5-0.5B-Instruct in bf16: next-token total variation between prefill and incremental
  decoding up to ~0.03 (mean ~0.003) at a position; with the same weights upcast to fp32, < 1e-5.
  Only the first token after each checkpoint is affected. On the MLX backend, `--mlx-dtype float32`
  removes the effect (an exact upcast of the stored weights, so the same policy).
- The reward grader (Math-Verify 0.9.0, with a string-normalisation fallback) defines `R`.
  Grader errors are part of `R` and are not measurement noise.
- H2 does not imply H3. A front-loaded profile shows where credit *could* be concentrated, not
  that concentrating it improves training. That is Stage 2's question.

## 11. H1: steering (added 2026-11-03, fixed before any steering data were computed)

H1 says earlier tokens shift the distribution of *later tokens* more. It is about the text, not
the reward, so it needs its own measurement.

**Definition.** Let `π(· | context)` be the sampling distribution at the run's temperature (no
top-p/top-k/min-p filters, as in the pilot). For a base trajectory with content tokens
`y_1 … y_L` and a position `t ≤ L − 1`:
- draw `r` alternative tokens `y'_t ~ π(· | x, y_<t)`, conditioned on not being a stop token
  (so the response continues); `y'_t = y_t` is allowed and contributes 0;
- for `s = t+1 … t+w` with `w = min(W, L − t)`, compare the next-token distribution at `s` with
  the real token versus the alternative, holding the real continuation fixed (teacher forcing):
  `D_s = KL( π(· | x, y_<s) ‖ π(· | x, y_<t, y'_t, y_{t+1..s−1}) )`;
- the **steering influence** of position `t` is `S(t) = E_{y'_t}[ Σ_{s=t+1}^{t+w} D_s ]`,
  estimated by the mean over the `r` draws (unbiased for that expectation), and its per-token
  form `S(t)/w`. The lag profile `E[D_{t+d}]`, `d = 1 … W`, is also recorded.

Positions are a fixed grid `t = 1, 1 + stride, …` along each base trajectory.

**Interpretation and limits.** `S(t)` measures how sensitive the model's later conditionals are
to the choice made at `t`, with the rest of the text held fixed. It is not the divergence
between full counterfactual continuations (that would need sampling from `y'_t` onwards). It is
mechanically small where `π(·|y_<t)` is peaked (the alternative is usually the same token), so
the entropy of `π(·|y_<t)` is a required covariate.

**Truncation.** Positions within `W` of the end have shorter windows and so less total
influence by construction. The primary analysis uses only positions with a full window
(`t + W ≤ L`). Restricting on `L` conditions on the future, so this analysis is descriptive and
has no conservation identity.

**Pre-registered tests** (prompt fixed effects, cluster bootstrap over prompts, as in §7):
- **H1-primary:** in a regression of `S(t)/W` on relative position `t/N`, the entropy of
  `π(·|y_<t)` and `log t`, the coefficient on `t/N` is `< 0` with the 95% CI excluding 0.
- **H1-descriptive:** mean `S(t)/W` by relative-position bin, and the lag profile.

Default settings: `stride = 16`, `r = 4`, `W = 64`.
