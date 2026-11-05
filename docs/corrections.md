# Corrections to the earlier plan

These are errors or overstatements in the plan I wrote before this code existed. Each was found
either by deriving the result carefully, by exact tests on the toy process, or by checking the
cited paper through web search. The code and `docs/methods.md` already reflect every correction.

## Method

1. **Checkpoint placement (an error that affects results).** I wrote "cut each [trajectory] at
   about 16 checkpoints, at equal fractions of its length". That rule depends on the final length
   `N`, so the checkpoints are not stopping times and the variance decomposition no longer holds.
   On a toy process the error reaches about 10% of `Var(R)`, and its sign varies.
   **Fix:** place checkpoints every `L` tokens, or online at delimiters. Use relative position
   only to bin segments afterwards (methods §2).

2. **"Your hypothesis is equivalent to saying the measured `c_t` decreases with position."** This
   overstates it. If `c_t` decreases, the curve `C(u)` is front-loaded (`F > 0`), but the converse
   does not hold. The hypothesis is now pre-registered as `F > 0`, with the decreasing-trend test
   (a negative position coefficient) reported separately (methods §8).

3. **Squared vs. absolute effects.** The plan swapped "average absolute effect" (`E|Δ|`) for
   squared increments without saying so. The swap is deliberate, because only the squared form
   has an exact decomposition and an unbiased estimator, but it is now stated explicitly
   (methods §4).

4. **"The best linear estimate of `Δ_t` from the sequence advantage alone is `c_t · A`."** This is
   correct per prompt when `Δ_t := 0` after the response ends. For a token known to exist, the
   right coefficient is `β_t = E[Δ_t² 1{N≥t}] / E[A² 1{N≥t}]`, and that is optimal only among
   *proportional* predictors. Applying one population-average profile to every prompt is an
   approximation (methods §5).

5. **Stage 2 arm A1 `N · c(t/N) · A`, "normalised to mean 1 per sequence".** Normalising per
   sequence makes each token's weight depend on the final length, so it uses the future again.
   **Fix:** weights are a function of absolute position (`β_t` from Stage 1), divided by one fixed
   constant computed from the Stage 1 data.

6. **"Positional weights … are a heuristic preconditioner."** "Preconditioner" was the wrong
   term. The weights change how per-timestep terms of the policy gradient are weighted, so the
   update is no longer an unbiased estimate of `∇E[R]`. They are a heuristic re-weighting.

7. **Prompt selection.** The plan said to pick prompts with a pass rate between 0.1 and 0.9
   without saying the pass rate must come from **separate** rollouts. If it doesn't, selection
   biases `V̂_0`. The code uses held-out selection rollouts.

8. **"A gap under about 5 points on AIME alone is not evidence."** Too strong. It is not reliable
   evidence *on its own*. The right analysis is a paired bootstrap over problems and seeds.

## Related work (checked by web search; see references.md)

9. **GRPO-λ.** I wrote that it reports gains on "Qwen2.5 and R1-Distill-Qwen-7B". The abstract
   reports LLaMA-3.1 and Qwen-2.5 models from 1.5B to 7B. An R1-Distill figure appears only in the
   introduction with an implausible number. The method also has more than one variant (ε-trace
   and ε-weight). Summarising it as "pushes credit toward early tokens" is accurate for how the
   traces work but simplifies the paper.

10. **Beyond the 80/20 rule.** "Wherever they occur" was my own gloss. The paper selects tokens by
    entropy. I have not confirmed that it analyses position at all.

11. **UPFT.** "The first ~8–32 tokens" should be "prefixes as short as 8 tokens". I could not
    confirm the 32.

12. **SimpleRL-Zoo.** It covers a Llama-3 8B model (listed as "Llama3-8B"; check the exact
    checkpoint), Mistral-7B/24B, DeepSeek-Math-7B, Qwen2.5-Math-7B and Qwen2.5 0.5B–32B. It does
    **not** include OLMo. My table implied OLMo-2 could use "the same" recipe. OLMo-2 appears in
    the Spurious Rewards paper instead.

13. **Open-RS.** "A few GPUs for about a day" should read: 4×A40 GPUs, about $42, 46.7% on AIME24
    (as reported). The abstract states the constraint as "within 24 hours" (checked 2026-11-03).

14. **Outcome-grounded Advantage Reshaping** (arXiv 2601.07408) appeared in my sources list but I
    never read or described it. It is listed as unreviewed.

## Implementation (MLX backend, added 2026-11-03)

15. **mlx-lm's sampler is not the vLLM/HF sampler.** `mlx_lm.sample_utils.make_sampler` (0.32.2)
    applies top-p, min-p and top-k to the *untempered* log-probabilities and divides by the
    temperature only in the final draw. vLLM and HF temper first. With any filter active the two
    define different policies. `tcredit/backends_mlx.py` uses its own sampler
    (temperature → top-k → top-p → min-p), tested against an independent numpy implementation.
    With `top_p = 1, top_k = 0, min_p = 0` all of these coincide. I have **not** checked that
    vLLM 0.33.0 applies the three filters in this same order; until that is checked, compare
    MLX and vLLM runs only with the filters disabled.

16. **mlx-lm's chat-template wrapper adds a thinking argument.** `TokenizerWrapper.apply_chat_template`
    injects `enable_thinking` (or the model's equivalent). The MLX backend calls the Hugging Face
    tokenizer's template directly; a test checks that the prompt ids equal what `transformers`
    produces.

17. **bf16 prefill and decode disagree measurably.** On Qwen2.5-0.5B-Instruct in bf16 the
    next-token total variation between a prefilled and an incrementally decoded position reached
    0.028–0.036 on a 20-token check (mean ~0.003). Both are about equally far from fp32. With the
    weights upcast to fp32 the gap is < 1e-5. Methods §10 called this "very slightly off-policy"
    without a number; this is the number for MLX bf16. vLLM on CUDA has not been measured.

## Citations checked against arXiv (2026-11-03)

18. **Temporal Scheduling for RLVR (2605.25381).** The earlier summary, "trains later tokens
    first, then adds earlier ones", is not in the abstract. The abstract describes prioritising
    "targeted tokens" distinguished by "trajectory percentiles", then attenuating toward ordinary
    optimisation. Whether that is about token position needs the paper. It may be the closest
    prior work to the positional hypothesis.

19. **Outcome-grounded Advantage Reshaping (2601.07408)**, previously "not reviewed", reshapes
    GRPO advantages by each token's estimated influence on the final answer (counterfactual
    perturbation or an input-gradient proxy). That is close to Stage 2's idea, so it must be
    discussed as related work, and any Stage 2 comparison should consider it as a baseline.

20. Several numbers in the earlier table are not in the abstracts and remain unchecked against
    the papers: VinePPO "9× fewer gradient steps", s1 "26 min on 16 H100", LIMO "817 samples",
    GRPO-λ variant names, the Dr. GRPO model name. Venues for Forking Paths (ICLR 2025), UPFT
    (NeurIPS 2025) and Spurious Rewards (PMLR) are not on arXiv and remain unverified.

## Bug found by the GPU cross-check (2026-11-04)

21. **The Mac (MLX) pilots used length-biased base trajectories. Their H1 and H2 results are
    invalid.** `MLXBackend`'s batched path appended each sample to its request's output list when
    the sample *finished*, so samples came back sorted by finish time. `measure_chunk` takes the
    first `k0` samples of the base request for `V̂_0` and the last `M` as base trajectories, so on
    the Mac the base trajectories were the slowest-finishing (longest) samples of each request.
    Found by running the same 50 prompts on vLLM: per-prompt pass rates agreed (difference
    −0.007 ± 0.011), but Mac base trajectories averaged 1,055 tokens, 84% finished, reward 0.42,
    against the same request's `V̂_0` of 0.59; vLLM's averaged 630 tokens, 97% finished, reward 0.54.
    The distribution tests compared multisets of outputs and could not see order.
    **Fixes:** outputs are returned in sample order (`_Row.slot`); `measure_chunk` now permutes each
    base request's samples at random (seeded) before splitting, so the split is exchangeable for any
    backend; regression tests `test_batched_outputs_are_in_sample_order_not_finish_order` and
    `tests/test_measure_order.py` (both verified to fail on the old behaviour).
    **Affected:** `runs/pilotA_q05b`, `runs/pilotB_q15b` (Stage 1 and steering), and the conclusions
    reported from them on 2026-11-03/04 ("position adds nothing once entropy is controlled"; the
    U-shape; the steering decline). Continuation estimates and prompt selection are unaffected
    (they use all samples symmetrically). The vLLM run (`runs/colab_A1`) is not affected by this bug;
    its base-trajectory reward matches its `V̂_0`.
