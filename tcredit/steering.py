"""H1 steering influence (docs/methods.md §11), MLX implementation.

For a base trajectory (prompt x, content y_1..y_L) and grid positions t, draw r alternative
tokens y'_t ~ pi(. | x, y_<t) (no stop tokens), teacher-force the real continuation after each,
and record D_s = KL(pi(.|x, y_<s) || pi(.|x, y_<t, y'_t, y_{t+1..s-1})) for s = t+1..t+w.

Efficiency: one single-row KV cache is advanced along the trajectory; at each grid position it
is copied into r+1 rows (real token + r alternatives) and the w-token window is run as one batch.
`naive_window_kl` recomputes the same quantity with independent cache-free forward passes; the
tests check that the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SteeringRecord:
    prompt_id: str
    traj_id: int
    n_tokens: int  # N as in the Trajectory record (includes the EOS position if finished)
    temperature: float
    window: int
    swaps: int
    t: list[int] = field(default_factory=list)  # 1-based position of the swapped token
    w: list[int] = field(default_factory=list)  # window length actually used
    entropy: list[float] = field(default_factory=list)  # entropy of pi(.|x, y_<t), nats
    n_same: list[int] = field(default_factory=list)  # draws with y'_t == y_t
    S: list[float] = field(default_factory=list)  # mean over draws of sum_s D_s
    S_sd: list[float] = field(default_factory=list)  # SD over draws
    lag: list[list[float]] = field(default_factory=list)  # mean D_{t+d}, d = 1..w


def _log_softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def _kl_rows(lp_ref, lp_alt):
    """KL(ref || alt) along the last axis (mlx arrays of log-probs)."""
    import mlx.core as mx

    return mx.sum(mx.exp(lp_ref) * (lp_ref - lp_alt), axis=-1)


def window_kl(model, cache, real_tok: int, alts: list[int], cont: list[int], temperature: float):
    """D[row, j] for the window: rows = alternatives, j = 0..w-1 (s = t+1+j).

    cache: single-row cache holding x, y_<t (not modified; merge copies it).
    cont: y_{t+1..t+w-1} (the w-1 real tokens fed after the swapped one).
    """
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache

    toks = [real_tok] + list(alts)
    R = len(toks)
    bc = [BatchKVCache.merge([c] * R) for c in cache]
    X = mx.array([[a] + list(cont) for a in toks])
    out = model(X, cache=bc).astype(mx.float32) / temperature
    lp = out - mx.logsumexp(out, axis=-1, keepdims=True)
    D = _kl_rows(lp[:1], lp[1:])  # [R-1, w]
    mx.eval(D)
    return np.array(D)


def naive_window_kl(model, prefix: list[int], real_tok: int, alts: list[int], cont: list[int], temperature: float):
    """Reference: independent cache-free forward passes (tests only)."""
    import mlx.core as mx

    def lp(seq):
        z = np.array(model(mx.array(seq)[None])[0].astype(mx.float32)) / temperature
        return _log_softmax_np(z)[len(prefix):]

    ref = lp(prefix + [real_tok] + cont)
    rows = []
    for a in alts:
        alt = lp(prefix + [a] + cont)
        rows.append((np.exp(ref) * (ref - alt)).sum(-1))
    return np.array(rows)


def steering_for_trajectory(
    model,
    prompt_ids: list[int],
    content: list[int],
    *,
    stop_ids: set[int],
    temperature: float,
    stride: int = 16,
    swaps: int = 4,
    window: int = 64,
    prefill_step: int = 512,
    alt_fn=None,
) -> dict:
    """Returns columns t, w, entropy, n_same, S, S_sd, lag for grid positions t = 1, 1+stride, ...
    with t <= L-1. `alt_fn(probs_np, t) -> list[int]` overrides sampling (tests)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    L = len(content)
    cache = make_prompt_cache(model)
    logits = None
    for a in range(0, len(prompt_ids), prefill_step):
        logits = model(mx.array(prompt_ids[a : a + prefill_step])[None], cache=cache)[:, -1, :]
        mx.eval(logits)
    out = {k: [] for k in ("t", "w", "entropy", "n_same", "S", "S_sd", "lag")}
    pos = 1  # next content position whose token is not yet in the cache
    for t in range(1, L, stride):
        if t > pos:  # advance the cache over y_pos .. y_{t-1}
            seg = content[pos - 1 : t - 1]
            for a in range(0, len(seg), prefill_step):
                logits = model(mx.array(seg[a : a + prefill_step])[None], cache=cache)[:, -1, :]
                mx.eval(logits)
            pos = t
        z = logits.astype(mx.float32) / temperature
        lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
        ent = float(-mx.sum(mx.exp(lp) * lp).item())
        if alt_fn is None:
            zs = _mask_stop(z, stop_ids)
            alts = mx.random.categorical(mx.broadcast_to(zs, (swaps, z.shape[-1])), axis=-1).tolist()
        else:
            alts = list(alt_fn(np.array(mx.exp(lp)[0]), t))
        w = min(window, L - t)
        real = content[t - 1]
        D = window_kl(model, cache, real, alts, content[t : t + w - 1], temperature)
        sums = D.sum(-1)
        out["t"].append(t)
        out["w"].append(w)
        out["entropy"].append(ent)
        out["n_same"].append(int(sum(a == real for a in alts)))
        out["S"].append(float(sums.mean()))
        out["S_sd"].append(float(sums.std(ddof=1)) if len(sums) > 1 else 0.0)
        out["lag"].append([float(x) for x in D.mean(0)])
    mx.clear_cache()
    return out


def _mask_stop(z, stop_ids):
    import mlx.core as mx

    m = np.zeros(z.shape[-1], dtype=bool)
    m[list(stop_ids)] = True
    return mx.where(mx.array(m)[None], -mx.inf, z)


# ----------------------------------------------------------------------------------------------
# PyTorch / Hugging Face implementation (for CUDA). Deliberately simple: each position recomputes
# the shared prefix inside one batch of r+1 rows, so no KV-cache API is involved.
# ----------------------------------------------------------------------------------------------


def _last_logits_torch(model, ids, keep: int):
    """Logits at the last `keep` positions of each row of `ids` ([B, T] long tensor), float32."""
    import torch

    with torch.no_grad():
        try:
            out = model(input_ids=ids, logits_to_keep=keep).logits
        except TypeError:  # older transformers
            out = model(input_ids=ids).logits[:, -keep:]
    return out[:, -keep:].float()


def steering_for_trajectory_torch(
    model,
    prompt_ids: list[int],
    content: list[int],
    *,
    stop_ids: set[int],
    temperature: float,
    stride: int = 16,
    swaps: int = 4,
    window: int = 64,
    generator=None,
    alt_fn=None,
    max_batch_tokens: int = 24576,
) -> dict:
    """Same quantities and conventions as `steering_for_trajectory` (MLX).

    Rows are run in micro-batches of at most `max_batch_tokens` tokens (at least one row), so
    long prefixes (R1-style responses of ~16k tokens) fit in GPU memory."""
    import torch

    dev = next(model.parameters()).device
    L = len(content)
    out = {k: [] for k in ("t", "w", "entropy", "n_same", "S", "S_sd", "lag")}
    stop = torch.tensor(sorted(stop_ids), device=dev)
    for t in range(1, L, stride):
        prefix = list(prompt_ids) + list(content[: t - 1])
        z = _last_logits_torch(model, torch.tensor([prefix], device=dev), 1)[0, -1] / temperature
        lp = torch.log_softmax(z, -1)
        ent = float(-(lp.exp() * lp).sum())
        if alt_fn is None:
            zs = z.clone()
            zs[stop] = -float("inf")
            probs = torch.softmax(zs, -1)
            alts = torch.multinomial(probs, swaps, replacement=True, generator=generator).tolist()
        else:
            alts = list(alt_fn(lp.exp().cpu().numpy(), t))
        w = min(window, L - t)
        real = content[t - 1]
        cont = list(content[t : t + w - 1])
        seqs = [prefix + [a] + cont for a in [real] + alts]
        per = max(1, max_batch_tokens // len(seqs[0]))
        lps = torch.cat([
            torch.log_softmax(_last_logits_torch(model, torch.tensor(seqs[i : i + per], device=dev), w) / temperature, -1)
            for i in range(0, len(seqs), per)
        ])  # [R+1, w, V]
        D = (lps[:1].exp() * (lps[:1] - lps[1:])).sum(-1).cpu().double().numpy()  # [R, w]
        sums = D.sum(-1)
        out["t"].append(t)
        out["w"].append(w)
        out["entropy"].append(ent)
        out["n_same"].append(int(sum(a == real for a in alts)))
        out["S"].append(float(sums.mean()))
        out["S_sd"].append(float(sums.std(ddof=1)) if len(sums) > 1 else 0.0)
        out["lag"].append([float(x) for x in D.mean(0)])
    return out


# ----------------------------------------------------------------------------------------------
# Analysis (methods §11)
# ----------------------------------------------------------------------------------------------


def steering_rows(records: list[dict], full_window_only: bool = True) -> dict[str, np.ndarray]:
    """One row per (trajectory, position): y = S/W, covariates u = t/N, entropy, log t."""
    prompts = {}
    cols = {k: [] for k in ("prompt", "y", "u", "entropy", "log_t", "t", "w")}
    for r in records:
        p = prompts.setdefault(r["prompt_id"], len(prompts))
        W = r["window"]
        for t, w, e, S in zip(r["t"], r["w"], r["entropy"], r["S"]):
            if full_window_only and w < W:
                continue
            cols["prompt"].append(p)
            cols["y"].append(S / W if full_window_only else S / w)
            cols["u"].append(t / r["n_tokens"])
            cols["entropy"].append(e)
            cols["log_t"].append(np.log(t))
            cols["t"].append(t)
            cols["w"].append(w)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def binned_means(rows: dict[str, np.ndarray], key: str, edges: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of y by bin of `key`, pooled over rows; percentile CI from a cluster bootstrap over prompts."""
    from .estimator import bootstrap_counts

    b = np.clip(np.digitize(rows[key], edges) - 1, 0, len(edges) - 2)
    prompts = rows["prompt"].astype(int)
    P, B = int(prompts.max()) + 1, len(edges) - 1
    sums = np.zeros((P, B))
    cnts = np.zeros((P, B))
    np.add.at(sums, (prompts, b), rows["y"])
    np.add.at(cnts, (prompts, b), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = sums.sum(0) / cnts.sum(0)
        W = bootstrap_counts(P, n_boot, np.random.default_rng(seed))
        bs = (W @ sums) / (W @ cnts)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return {"edges": edges.tolist(), "mean": est.tolist(), "ci_low": lo.tolist(), "ci_high": hi.tolist(),
            "n": cnts.sum(0).astype(int).tolist()}
