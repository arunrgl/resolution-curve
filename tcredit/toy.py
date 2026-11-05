"""An exactly solvable toy "policy" used to validate the estimators and the pipeline.

Tokens: 0, 1 and EOS (=2). At step t (1-indexed position of the token being sampled), given the
running score s = sum_i a_i * y_i of the previous content tokens:
  * with probability h(t, s) the policy emits EOS and the episode ends with R = 1{s >= theta};
  * otherwise it emits 1 with probability q_t (adding a_t to s) and 0 otherwise.
If `budget` content tokens are produced without EOS, the response is truncated and R = 0.

Because the state is (t, s), V_t = E[R | y_<=t] is computable exactly by backward induction and
every quantity of interest can be computed exactly by enumerating all paths (small budgets).
The hazard can depend on s, which makes the length N correlated with the outcome, as in real
reasoning traces (wrong answers tend to be longer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterator

import numpy as np

EOS = 2


@dataclass(frozen=True)
class ToyProcess:
    a: tuple[int, ...]  # weight of a "1" at each position (length = budget)
    q: tuple[float, ...]  # P(token=1 | not EOS) at each position
    theta: int
    h_below: float  # EOS hazard when s < theta
    h_above: float  # EOS hazard when s >= theta
    min_len: int = 1  # no EOS before this position

    @property
    def budget(self) -> int:
        return len(self.a)

    def hazard(self, t: int, s: int) -> float:
        if t < self.min_len:
            return 0.0
        return self.h_above if s >= self.theta else self.h_below

    # ---------------------------------------------------------------- exact values
    def value(self, t: int, s: int) -> float:
        """V after t content tokens with running score s (episode still running)."""
        return _value(self, t, s)

    def prefix_value(self, content: list[int]) -> float:
        t, s = len(content), self.score(content)
        return self.value(t, s)

    def score(self, content: list[int]) -> int:
        return int(sum(self.a[i] for i, y in enumerate(content) if y == 1))

    def reward(self, content: list[int], finished: bool) -> float:
        if not finished:
            return 0.0
        return 1.0 if self.score(content) >= self.theta else 0.0

    # ---------------------------------------------------------------- sampling
    def sample(self, prefix: list[int], rng: np.random.Generator, max_new_tokens: int) -> tuple[list[int], bool]:
        """Continue a content prefix. Returns (content continuation without EOS, finished)."""
        t, s = len(prefix), self.score(prefix)
        out: list[int] = []
        while True:
            if t >= self.budget or len(out) >= max_new_tokens:
                return out, False
            pos = t + 1
            if rng.random() < self.hazard(pos, s):
                return out, True
            y = 1 if rng.random() < self.q[t] else 0
            out.append(y)
            s += self.a[t] * y
            t += 1

    # ---------------------------------------------------------------- enumeration
    def paths(self) -> Iterator[tuple[list[int], bool, float]]:
        """All complete responses as (content tokens, finished, probability)."""
        stack: list[tuple[list[int], int, float]] = [([], 0, 1.0)]
        while stack:
            content, s, p = stack.pop()
            t = len(content)
            if t >= self.budget:
                yield content, False, p
                continue
            h = self.hazard(t + 1, s)
            if h > 0:
                yield content, True, p * h
            if h < 1:
                q = self.q[t]
                if q > 0:
                    stack.append((content + [1], s + self.a[t], p * (1 - h) * q))
                if q < 1:
                    stack.append((content + [0], s, p * (1 - h) * (1 - q)))

    def value_path(self, content: list[int], finished: bool) -> np.ndarray:
        """[V_0, V_1, ..., V_N] along a complete path (N includes the EOS position if finished)."""
        vals = [self.prefix_value(content[:t]) for t in range(len(content) + 1)]
        if finished:
            vals.append(self.reward(content, True))
        else:
            vals[-1] = 0.0  # truncated at the budget: R = 0 (prefix_value already gives 0)
        return np.asarray(vals, dtype=float)


@lru_cache(maxsize=None)
def _value_cached(proc: ToyProcess, t: int, s: int) -> float:
    if t >= proc.budget:
        return 0.0
    h = proc.hazard(t + 1, s)
    r_stop = 1.0 if s >= proc.theta else 0.0
    q = proc.q[t]
    cont = q * _value_cached(proc, t + 1, s + proc.a[t]) + (1 - q) * _value_cached(proc, t + 1, s)
    return h * r_stop + (1 - h) * cont


def _value(proc: ToyProcess, t: int, s: int) -> float:
    return _value_cached(proc, t, s)


# ------------------------------------------------------------------------------------------------
# Exact expectations by enumeration
# ------------------------------------------------------------------------------------------------


@dataclass
class ExactSummary:
    v0: float
    var_r: float
    sum_sq_increments: float  # E[sum_t Delta_t^2]
    per_token_sq: np.ndarray  # [budget] E[Delta_t^2 1{N>=t}]
    per_token_a2_alive: np.ndarray  # [budget] E[A^2 1{N>=t}]
    cov_delta_a: np.ndarray  # [budget] E[Delta_t * A]


def exact_summary(proc: ToyProcess) -> ExactSummary:
    T = proc.budget
    v0 = proc.value(0, 0)
    sq = np.zeros(T)
    a2 = np.zeros(T)
    cov = np.zeros(T)
    total = 0.0
    var_r = 0.0
    for content, finished, p in proc.paths():
        v = proc.value_path(content, finished)
        R = v[-1]
        A = R - v0
        d = np.diff(v)
        N = len(d)
        sq[:N] += p * d**2
        cov[:N] += p * d * A
        a2[:N] += p * A**2
        total += p * float((d**2).sum())
        var_r += p * (R - v0) ** 2
    return ExactSummary(v0, var_r, total, sq, a2, cov)


def exact_segment_sum(proc: ToyProcess, rule: Callable[[list[int], bool, int], list[int]]) -> float:
    """E[sum_j D_j^2] for checkpoints chosen by `rule(content, finished, N) -> interior t's`."""
    out = 0.0
    for content, finished, p in proc.paths():
        v = proc.value_path(content, finished)
        N = len(v) - 1
        cps = [0] + sorted({t for t in rule(content, finished, N) if 0 < t < N}) + [N]
        vv = v[cps]
        out += p * float((np.diff(vv) ** 2).sum())
    return out


def exact_rel_bin_mass(
    proc: ToyProcess, rule: Callable[[list[int], bool, int], list[int]], n_bins: int
) -> np.ndarray:
    """E[sum_j D_j^2 * frac_jb] with the same uniform-within-segment spreading as the estimator."""
    from .estimator import interval_overlap

    edges = np.linspace(0, 1, n_bins + 1)
    mass = np.zeros(n_bins)
    for content, finished, p in proc.paths():
        v = proc.value_path(content, finished)
        N = len(v) - 1
        cps = np.array([0] + sorted({t for t in rule(content, finished, N) if 0 < t < N}) + [N], dtype=float)
        d2 = np.diff(v[cps.astype(int)]) ** 2
        ov = interval_overlap(cps[:-1] / N, cps[1:] / N, edges)
        frac = ov / ((cps[1:] - cps[:-1]) / N)[:, None]
        mass += p * (d2[:, None] * frac).sum(axis=0)
    return mass


# ------------------------------------------------------------------------------------------------
# A Backend implementation over a set of toy processes (one per "prompt")
# ------------------------------------------------------------------------------------------------

PROMPT_TOKEN_BASE = 10_000


@dataclass
class ToyBackend:
    processes: list[ToyProcess]
    seed: int = 0
    rng: np.random.Generator = field(init=False)
    n_generated_tokens: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def prompt_ids(self, prompt: dict) -> list[int]:
        return [PROMPT_TOKEN_BASE + int(prompt["toy_index"])]

    def token_pieces(self, ids: list[int]) -> list[str]:
        return ["\n\n" if y == 1 else "x" for y in ids]  # lets tests exercise the delimiter rule

    def generate(self, requests):
        from .backends import GenOutput

        results = []
        for r in requests:
            proc = self.processes[r.prompt_ids[0] - PROMPT_TOKEN_BASE]
            outs = []
            for _ in range(r.n):
                cont, fin = proc.sample(list(r.prefix_ids), self.rng, r.max_new_tokens)
                self.n_generated_tokens += len(cont) + int(fin)
                outs.append(GenOutput(token_ids=cont, finished=fin))
            results.append(outs)
        return results


def toy_reward_fn(processes: list[ToyProcess]):
    def reward(prompt: dict, content_ids: list[int], finished: bool) -> float:
        return processes[int(prompt["toy_index"])].reward(list(content_ids), finished)

    return reward
