"""Per-token policy entropy along base trajectories (teacher forcing with Hugging Face transformers).

entropy[t-1] is the entropy of the distribution from which response token t was sampled,
softmax(logits / temperature), for t = 1..N (N includes the EOS step for finished responses).

Limitation: if the policy was sampled with top-p/top-k/min-p truncation, the sampling distribution
is the truncated one; this computes the entropy of the untruncated temperature-scaled softmax.
Prefer top_p=1, top_k=0, min_p=0 for the Stage 1 measurement so the two coincide.
"""

from __future__ import annotations


def distribution_rows(n_prompt: int, n_content: int, finished: bool) -> tuple[int, int]:
    """Rows [start, stop) of the causal-LM logits (input = prompt + content) that give the sampling
    distribution of response tokens 1..N. Row i predicts input position i+1, so response token t
    (input position n_prompt + t - 1) is predicted by row n_prompt + t - 2.
    For a finished response the EOS step (t = n_content + 1) uses the last row."""
    if n_prompt < 1:
        raise ValueError("empty prompt")
    n_dist = n_content + (1 if finished else 0)
    return n_prompt - 1, n_prompt - 1 + n_dist


def token_entropies(model, prompt_ids: list[int], content_ids: list[int], finished: bool,
                    temperature: float = 1.0, chunk: int = 2048) -> list[float]:
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    start, stop = distribution_rows(len(prompt_ids), len(content_ids), finished)
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(content_ids)], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]  # [P+L, V]; row i predicts token i+1
    rows = logits[start:stop]
    out = []
    for i in range(0, stop - start, chunk):
        z = rows[i : i + chunk].float() / temperature
        logp = torch.log_softmax(z, dim=-1)
        out.append(-(logp.exp() * logp).sum(dim=-1))
    return torch.cat(out).tolist() if out else []


def token_entropies_mlx(model, prompt_ids: list[int], content_ids: list[int], finished: bool,
                        temperature: float = 1.0, chunk: int = 512) -> list[float]:
    """MLX version of token_entropies (same row alignment; prefilled through a KV cache in chunks)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    start, stop = distribution_rows(len(prompt_ids), len(content_ids), finished)
    ids = list(prompt_ids) + list(content_ids)
    cache = make_prompt_cache(model)
    out: list[float] = []
    for a in range(0, len(ids), chunk):
        z = model(mx.array(ids[a : a + chunk])[None], cache=cache)[0].astype(mx.float32) / temperature
        lo, hi = max(start, a), min(stop, a + z.shape[0])  # global rows wanted in this chunk
        if lo < hi:
            zz = z[lo - a : hi - a]
            lp = zz - mx.logsumexp(zz, axis=-1, keepdims=True)
            ent = -mx.sum(mx.exp(lp) * lp, axis=-1)
            out.extend(ent.tolist())
    mx.clear_cache()
    return out
