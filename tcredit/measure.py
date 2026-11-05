"""Stage 1 measurement loop: select prompts, sample base trajectories, estimate V at checkpoints.

Sampling design (docs/methods.md §3):
  1. Selection: n_select rollouts per prompt; keep prompts with select_lo <= pass rate <= select_hi.
     These rollouts are discarded afterwards so that selection does not bias any estimate.
  2. Per kept prompt, one request with n = k0 + m_base i.i.d. samples: the first k0 estimate V_0 and
     Var(R|x); the remaining m_base are base trajectories.
  3. For every base trajectory, stopping-time checkpoints 0 < t_1 < ... < N; k continuations from
     each prefix y_<=t_j with budget max_new_tokens - t_j. All rollouts are independent.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Iterator

import numpy as np

from .backends import Backend, GenRequest
from .checkpoints import delimiter_checkpoints, fixed_interval, n_positions
from .records import Trajectory, mc_stats

RewardFn = Callable[[dict, list[int], bool], float]


@dataclass
class Stage1Config:
    n_select: int = 8
    select_lo: float = 0.125
    select_hi: float = 0.875
    max_prompts: int | None = None
    k0: int = 32
    m_base: int = 4
    k: int = 16
    max_new_tokens: int = 4096
    rule: str = "fixed"  # "fixed" | "delimiter"
    interval: int = 256
    delimiter: str = "\n\n"
    min_gap: int = 64
    max_checkpoints: int | None = None
    chunk_prompts: int = 8
    select_chunk_prompts: int = 256
    seed: int = 0
    store_tokens: bool = True  # keep base-trajectory token ids (needed for entropy / role analyses)

    def validate(self) -> None:
        if self.k0 < 2 or self.k < 2:
            raise ValueError("k0 and k must be >= 2 (unbiased variance needs two samples)")
        if self.m_base < 1:
            raise ValueError("m_base must be >= 1")
        if not 0.0 <= self.select_lo <= self.select_hi <= 1.0:
            raise ValueError("need 0 <= select_lo <= select_hi <= 1")
        if self.rule not in ("fixed", "delimiter"):
            raise ValueError("rule must be 'fixed' or 'delimiter'")


def _chunks(xs: list, n: int) -> Iterator[list]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _stable_hash(s: str) -> int:
    import zlib

    return zlib.crc32(s.encode())


def _check_budget(content_len: int, finished: bool, budget: int, what: str) -> None:
    # vLLM counts the stop token against max_tokens, so a finished response has <= budget-1 content tokens
    if content_len > budget - (1 if finished else 0) or (not finished and content_len != budget):
        raise RuntimeError(
            f"{what}: got {content_len} tokens (finished={finished}) with budget {budget}; "
            "the backend is capping generation inconsistently (check max_model_len)."
        )


def select_prompts(backend: Backend, prompts: list[dict], reward_fn: RewardFn, cfg: Stage1Config) -> list[dict]:
    cfg.validate()
    scored = []
    for chunk in _chunks(prompts, cfg.select_chunk_prompts):
        reqs = [GenRequest(backend.prompt_ids(p), [], cfg.n_select, cfg.max_new_tokens) for p in chunk]
        for p, outs in zip(chunk, backend.generate(reqs)):
            rs = [reward_fn(p, o.token_ids, o.finished) for o in outs]
            q = dict(p)
            q["p_select"] = float(np.mean(rs))
            scored.append(q)
    kept = [p for p in scored if cfg.select_lo <= p["p_select"] <= cfg.select_hi]
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(kept))
    kept = [kept[i] for i in order]
    if cfg.max_prompts is not None:
        kept = kept[: cfg.max_prompts]
    return kept


def _rule(cfg: Stage1Config, backend: Backend) -> Callable[[list[int], int], list[int]]:
    if cfg.rule == "fixed":
        return lambda content, N: fixed_interval(N, cfg.interval, cfg.max_checkpoints)
    return lambda content, N: delimiter_checkpoints(
        backend.token_pieces(content), N, cfg.delimiter, cfg.min_gap, cfg.max_checkpoints
    )


def measure_chunk(backend: Backend, prompts: list[dict], reward_fn: RewardFn, cfg: Stage1Config) -> list[Trajectory]:
    cfg.validate()
    rule = _rule(cfg, backend)
    pids = [backend.prompt_ids(p) for p in prompts]
    base = backend.generate([GenRequest(pid, [], cfg.k0 + cfg.m_base, cfg.max_new_tokens) for pid in pids])

    pending = []  # (prompt index, traj index, content, finished, N, checkpoints, R, (k0, m0, s0))
    for i, (p, outs) in enumerate(zip(prompts, base)):
        if len(outs) != cfg.k0 + cfg.m_base:
            raise RuntimeError("backend returned the wrong number of samples")
        # Split the k0 + m_base samples at random, independently of the outputs, so the split is
        # exchangeable even if a backend returns samples in an order correlated with their content
        # (e.g. by finish time; corrections 21).
        perm = np.random.default_rng([cfg.seed, _stable_hash(p["id"])]).permutation(len(outs))
        outs = [outs[j] for j in perm]
        for o in outs:
            _check_budget(len(o.token_ids), o.finished, cfg.max_new_tokens, "base sample")
        v0 = mc_stats(reward_fn(p, o.token_ids, o.finished) for o in outs[: cfg.k0])
        for b, o in enumerate(outs[cfg.k0 :]):
            content = list(o.token_ids)
            N = n_positions(len(content), o.finished)
            cps = sorted(set(t for t in rule(content, N) if 0 < t < N))
            R = float(reward_fn(p, content, o.finished))
            pending.append((i, b, content, o.finished, N, cps, R, v0))

    reqs, owners = [], []
    for j, (i, b, content, fin, N, cps, R, v0) in enumerate(pending):
        for t in cps:
            reqs.append(GenRequest(pids[i], content[:t], cfg.k, cfg.max_new_tokens - t))
            owners.append((j, t))
    outs_all = backend.generate(reqs) if reqs else []

    per_traj: dict[int, dict[int, tuple[int, float, float]]] = {}
    for (j, t), outs in zip(owners, outs_all):
        i, b, content, *_ = pending[j]
        prefix = content[:t]
        rs = []
        for o in outs:
            _check_budget(t + len(o.token_ids), o.finished, cfg.max_new_tokens, "continuation")
            rs.append(reward_fn(prompts[i], prefix + list(o.token_ids), o.finished))
        per_traj.setdefault(j, {})[t] = mc_stats(rs)

    trajs = []
    for j, (i, b, content, fin, N, cps, R, (k0, m0, s0)) in enumerate(pending):
        stats = per_traj.get(j, {})
        trajs.append(
            Trajectory(
                prompt_id=prompts[i]["id"],
                traj_id=b,
                n_tokens=N,
                finished=fin,
                reward=R,
                checkpoints=[0] + cps + [N],
                v_mean=[m0] + [stats[t][1] for t in cps] + [R],
                v_var=[s0] + [stats[t][2] for t in cps] + [0.0],
                k=[k0] + [stats[t][0] for t in cps] + [0],
                meta={"p_select": prompts[i].get("p_select")},
                token_ids=content if cfg.store_tokens else None,
                prompt_token_ids=pids[i] if cfg.store_tokens else None,
            )
        )
    for t in trajs:
        t.validate()
    return trajs


def done_prompt_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["prompt_id"])
    return out


def run_measurement(
    backend: Backend,
    prompts: list[dict],
    reward_fn: RewardFn,
    cfg: Stage1Config,
    out_path: str,
    log: Callable[[str], None] = print,
    max_chunks: int | None = None,
) -> bool:
    """Measure all `prompts` (already selected) in chunks, appending to out_path; resumable.
    Stops after `max_chunks` chunks if given. Returns True when every prompt is measured."""
    done = done_prompt_ids(out_path)
    todo = [p for p in prompts if p["id"] not in done]
    log(f"{len(done)} prompts already measured, {len(todo)} to go")
    for n, chunk in enumerate(_chunks(todo, cfg.chunk_prompts)):
        if max_chunks is not None and n >= max_chunks:
            return False
        trajs = measure_chunk(backend, chunk, reward_fn, cfg)
        with open(out_path, "a") as f:
            f.write("".join(json.dumps(asdict(t)) + "\n" for t in trajs))
            f.flush()
            os.fsync(f.fileno())
        log(f"chunk {n + 1}: wrote {len(trajs)} trajectories for {len(chunk)} prompts")
    return True
