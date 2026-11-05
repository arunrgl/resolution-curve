# References and verification status

**Read this before citing anything.** On 2026-11-03 every arXiv entry below was checked against
its arXiv metadata and abstract (arXiv API, latest version shown) and its official BibTeX was
exported, unedited, to `docs/references.bib`. "Abstract-checked" means the claim appears in the
abstract or the arXiv comments field. Claims marked "body only" were **not** in the abstract and
still need checking against the PDF. Venues come from the arXiv comments field; where that is
silent the venue is unverified. The BibTeX entries are `@misc` arXiv records: add venues by hand.

| Work | arXiv (version read) | What we cite it for | Status |
|---|---|---|---|
| VinePPO (Kazemnejad, Aghajohari, Portelance, Sordoni, Reddy, Courville, Le Roux) | 2410.01679v2 | Monte Carlo value estimates replace PPO's value network; beats PPO on MATH/GSM8K in up to 3.0× less wall-clock time; ICML 2025 | abstract-checked. "Up to 9× fewer gradient steps": body only |
| GRPO-λ (Parthasarathi, Reymond, Chen, Cui, Chandar) | 2510.00194v1 | critic-free eligibility traces for GRPO; LLaMA-3.1 and Qwen-2.5, 1.5B–7B; average gain over 3 points, 4.5 on 7B; 30–40% better during training | abstract-checked; authors confirmed. Variant names (ε-trace, ε-weight): body only |
| Not only where, But when: Temporal Scheduling for RLVR (Zhang, Li, Zhao, Wang) | 2605.25381v1 | schedules credit-allocation criteria over training: first prioritise "targeted tokens" (distinguished by trajectory percentiles), then attenuate toward ordinary optimisation | abstract-checked. **Our earlier summary ("trains later tokens first, then adds earlier ones") is not supported by the abstract**; whether "trajectory percentiles" means token position must be read in the paper. Closest related work to our premise: read in full |
| Beyond the 80/20 Rule (Wang, Yu, Gao, Zheng, Liu, Lu, … 18 authors) | 2506.01939v2 | ~20% high-entropy forking tokens; updating only them matches full updates on Qwen3-8B; +11.04 AIME'25 / +7.71 AIME'24 on Qwen3-32B; NeurIPS 2025 | abstract-checked. The paper does not address position (abstract is about entropy only) |
| Forking Paths in Neural Text Generation (Bigelow, Holtzman, Tanaka, Ullman) | 2412.07961v1 | resample at individual tokens and track the outcome distribution; forking tokens include punctuation | abstract-checked. ICLR 2025: not on arXiv, unverified |
| The First Few Tokens Are All You Need / UPFT (Ji, Xu, Liang, Liu, He, Chen, … 13 authors) | 2503.02875v1 | training on prefixes as short as 8 tokens matches rejection-sampling fine-tuning; −75% training time, −99% sampling cost; errors tend to appear later | abstract-checked. NeurIPS 2025: unverified |
| Spurious Rewards (Shao, Li, Xin, Geng, Wang, Oh, … 14 authors) | 2506.10947v2 | Qwen2.5-Math-7B on MATH-500: +21.4 points with random rewards vs +29.1 with ground truth; gains often absent for Llama3 / OLMo2 | abstract-checked. PMLR v306: unverified |
| Understanding R1-Zero-Like Training / Dr. GRPO (Liu, Chen, Li, Qi, Pang, Du, … 8 authors) | 2503.20783v2 | GRPO optimisation bias that lengthens (especially incorrect) responses; 43.3% on AIME 2024 with a 7B base model | abstract-checked. The name "Oat-Zero-7B": body only |
| SimpleRL-Zoo (Zeng, Huang, Liu, Liu, He, Ma, … 7 authors) | 2503.18892v3 | zero-RL on 10 base models: LLama3-8B, Mistral-7B/24B, DeepSeek-Math-7B, Qwen2.5-math-7B, Qwen2.5 0.5B–32B; COLM 2025 | abstract-checked. Exact checkpoint names: body only |
| Reinforcement Learning for Reasoning in Small LLMs / Open-RS (Dang, Ngo) | 2503.16219v2 | DeepSeek-R1-Distill-Qwen-1.5B; 4× A40 (48 GB) within 24 hours; 7,000 samples; $42; AIME24 46.7%; AMC23 63%→80% | abstract-checked ("within 24 hours" is the stated constraint) |
| s1: Simple test-time scaling (Muennighoff, Yang, Shi, Li, Fei-Fei, Hajishirzi, … 10 authors) | 2501.19393v3 | s1K = 1,000 examples; Qwen2.5-32B-Instruct; budget forcing | abstract-checked. "26 min on 16 H100": body only |
| LIMO: Less is More for Reasoning (Ye, Huang, Xiao, Chern, Xia, Liu) | 2502.03387v3 | SFT on few examples; v3 abstract: 63.3% AIME24, 95.6% MATH500, "1% of the training data"; COLM 2025 | abstract-checked. "817 samples": body only; numbers differ across versions, cite the version |
| DeepSeekMath (Shao, Wang, Zhu, Xu, Song, Bi, … 11 authors) | 2402.03300v3 | introduces GRPO | abstract-checked |
| DeepSeek-R1 (DeepSeek-AI; 200 authors listed) | 2501.12948v2 | RL-incentivised reasoning; patterns distilled into smaller models | abstract-checked. The specific R1-Distill checkpoints: body only |
| 2 OLMo 2 Furious (Team OLMo; 43 authors) | 2501.00656v3 | fully open 7B/13B/32B family; COLM 2025 (shorter version) | abstract-checked |
| Outcome-grounded Advantage Reshaping, OAR (Li, Kang, Xiao, Xing, Si, Li, … 10 authors) | 2601.07408v2 | redistributes GRPO advantages by each token's influence on the final answer, via counterfactual token perturbations (OAR-P) or an input-gradient proxy (OAR-G) | abstract-checked. **Directly related to Stage 2 (token-level credit from outcome sensitivity); read in full and position our method against it** |
| DeepScaleR (Agentica, blog, Feb 2025) | — | RL recipe on R1-Distill-Qwen-1.5B | **not checked** (not on arXiv) |

## Software

| Tool | Version | Role |
|---|---|---|
| vLLM | 0.33.0 (API checked against the source; not run here) | generation (CUDA) |
| mlx / mlx-lm | 0.32.4 / 0.32.2 | generation (Apple silicon) |
| Math-Verify | 0.9.0 | answer equivalence for `R` |
| NumPy | 2.5.3 | estimators |

## Policy note on crediting AI assistance

arXiv's policy (blog post of 31 January 2023, quoted in later sources) says generative AI tools
cannot be listed as authors, that significant use of text-generating tools should be reported,
and that authors are responsible for any errors, incorrect references or misleading content such
tools produce. Check arXiv's current moderation page before submitting; a later CS-specific
change (31 October 2025) concerns review and position papers.
