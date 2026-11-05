"""Record types shared by the measurement and analysis code.

A Trajectory is one base response y_1..y_N sampled from the policy for prompt x, together with
Monte Carlo estimates of the value V_t = E[R | x, y_<=t] at a set of checkpoints
0 = t_0 < t_1 < ... < t_J = N.

Conventions
-----------
* ``checkpoints[0] == 0`` and ``checkpoints[-1] == n_tokens``.
* ``v_mean[j]`` is the Monte Carlo mean reward of ``k[j]`` continuations sampled from the prefix
  y_<=t_j, and ``v_var[j]`` is their *unbiased* sample variance (denominator k-1).
* The terminal checkpoint is exact: ``k[-1] == 0``, ``v_mean[-1] == reward``, ``v_var[-1] == 0``.
* ``v_mean[0]`` (the prompt value) is shared by all trajectories of a prompt; its rollouts are
  independent of the base trajectories and of every other estimate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator


@dataclass
class Trajectory:
    prompt_id: str
    traj_id: int
    n_tokens: int
    finished: bool
    reward: float
    checkpoints: list[int]
    v_mean: list[float]
    v_var: list[float]
    k: list[int]
    entropy: list[float] | None = None  # optional per-token entropy, length n_tokens
    meta: dict = field(default_factory=dict)
    token_ids: list[int] | None = None  # base response content tokens (stop token stripped)
    prompt_token_ids: list[int] | None = None

    def validate(self) -> None:
        c = self.checkpoints
        J = len(c)
        if J < 2:
            raise ValueError(f"{self.prompt_id}/{self.traj_id}: need at least checkpoints [0, N]")
        if not (len(self.v_mean) == len(self.v_var) == len(self.k) == J):
            raise ValueError(f"{self.prompt_id}/{self.traj_id}: length mismatch")
        if c[0] != 0 or c[-1] != self.n_tokens:
            raise ValueError(f"{self.prompt_id}/{self.traj_id}: checkpoints must start at 0 and end at N")
        if any(b <= a for a, b in zip(c, c[1:])):
            raise ValueError(f"{self.prompt_id}/{self.traj_id}: checkpoints must be strictly increasing")
        if self.k[-1] != 0 or self.v_var[-1] != 0 or self.v_mean[-1] != self.reward:
            raise ValueError(f"{self.prompt_id}/{self.traj_id}: terminal checkpoint must be exact (k=0, var=0, mean=R)")
        for j in range(J - 1):
            if self.k[j] < 2:
                raise ValueError(
                    f"{self.prompt_id}/{self.traj_id}: checkpoint {j} has k={self.k[j]}; need k>=2 to debias"
                )
            if self.v_var[j] < 0:
                raise ValueError(f"{self.prompt_id}/{self.traj_id}: negative sample variance")
        if self.entropy is not None and len(self.entropy) != self.n_tokens:
            raise ValueError(f"{self.prompt_id}/{self.traj_id}: entropy length != n_tokens")
        if self.token_ids is not None and len(self.token_ids) != self.n_tokens - (1 if self.finished else 0):
            raise ValueError(f"{self.prompt_id}/{self.traj_id}: token_ids length inconsistent with n_tokens")


def mc_stats(rewards: Iterable[float]) -> tuple[int, float, float]:
    """Return (k, mean, unbiased sample variance) of a list of rewards."""
    r = [float(x) for x in rewards]
    k = len(r)
    if k < 2:
        raise ValueError("need at least 2 rollouts for an unbiased variance")
    m = sum(r) / k
    s2 = sum((x - m) ** 2 for x in r) / (k - 1)
    return k, m, s2


def write_jsonl(path: str, trajs: Iterable[Trajectory], mode: str = "a") -> None:
    with open(path, mode) as f:
        for t in trajs:
            f.write(json.dumps(asdict(t)) + "\n")


def read_jsonl(path: str) -> Iterator[Trajectory]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield Trajectory(**json.loads(line))
