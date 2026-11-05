"""Estimators for the outcome-variance resolution profile (Stage 1).

Background (full derivation in docs/methods.md)
-----------------------------------------------
Let V_t = E[R | x, y_<=t] under the sampling policy, with V_N = R (reward is a deterministic
function of the finished response). V is a martingale, so for any *stopping-time* checkpoints
0 = t_0 < t_1 < ... < t_J = N the increments D_j = V_{t_j} - V_{t_{j-1}} are uncorrelated and

    sum_j E[D_j^2 | x] = Var(R | x).                                       (1)

E[D_j^2] is therefore the share of the outcome variance that is resolved inside segment j.

Debiasing. V_{t_j} is estimated by the mean of k_j i.i.d. continuations, so
E[(Vhat_j - Vhat_{j-1})^2] = D_j^2 + Var(Vhat_j) + Var(Vhat_{j-1}) when the two estimates use
independent rollouts. Var(Vhat_j) is estimated without bias by s_j^2 / k_j, where s_j^2 is the
unbiased sample variance of the k_j rewards (for 0/1 rewards: Vhat(1-Vhat)/(k-1)). The terminal
value is exact (k=0, no noise term).

Everything here is plain numpy so that it can be unit-tested against exact enumeration
(see tcredit/toy.py and tests/).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .records import Trajectory


# ----------------------------------------------------------------------------------------------
# Per-trajectory quantities
# ----------------------------------------------------------------------------------------------


def noise_terms(traj: Trajectory) -> np.ndarray:
    """Unbiased estimate of Var(Vhat_j | prefix) for every checkpoint (0 for the exact terminal)."""
    return np.array(
        [traj.v_var[j] / traj.k[j] if traj.k[j] > 0 else 0.0 for j in range(len(traj.checkpoints))],
        dtype=float,
    )


@dataclass
class SegmentTable:
    start: np.ndarray  # t_{j-1}
    end: np.ndarray  # t_j
    d2_raw: np.ndarray  # (Vhat_j - Vhat_{j-1})^2
    d2: np.ndarray  # debiased estimate of E[D_j^2 | prefix]; can be negative due to noise
    is_last: np.ndarray  # True for the segment that ends at N


def segment_table(traj: Trajectory) -> SegmentTable:
    c = np.asarray(traj.checkpoints, dtype=float)
    v = np.asarray(traj.v_mean, dtype=float)
    nz = noise_terms(traj)
    raw = np.diff(v) ** 2
    d2 = raw - nz[1:] - nz[:-1]
    is_last = np.zeros(len(raw), dtype=bool)
    is_last[-1] = True
    return SegmentTable(start=c[:-1], end=c[1:], d2_raw=raw, d2=d2, is_last=is_last)


def interval_overlap(start: np.ndarray, end: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Length of overlap between each interval (start_i, end_i] and each bin (edges_b, edges_{b+1}].

    Returns an array of shape [n_intervals, n_bins]. With integer token positions, interval
    (t_{j-1}, t_j] contains exactly the tokens t_{j-1}+1 .. t_j (token t occupies (t-1, t]), so
    the overlap length equals the number of tokens of the segment that fall in the bin.
    """
    s = np.asarray(start, dtype=float)[:, None]
    e = np.asarray(end, dtype=float)[:, None]
    lo = np.maximum(s, edges[None, :-1])
    hi = np.minimum(e, edges[None, 1:])
    return np.clip(hi - lo, 0.0, None)


# ----------------------------------------------------------------------------------------------
# Per-prompt aggregation
# ----------------------------------------------------------------------------------------------


@dataclass
class PromptAggregate:
    prompt_ids: list[str]
    rel_mass: np.ndarray  # [P, B] mean over the prompt's trajectories of debiased mass per rel. bin
    total: np.ndarray  # [P] mean over trajectories of sum_j d2  (estimates Var(R|x))
    last_mass: np.ndarray  # [P] mean over trajectories of d2 of the terminal segment
    var_r: np.ndarray  # [P] unbiased estimate of Var(R|x) from the independent prompt rollouts
    v0: np.ndarray  # [P] Vhat_0
    n_traj: np.ndarray  # [P]
    edges: np.ndarray  # [B+1] relative-position bin edges


def group_by_prompt(trajs: list[Trajectory]) -> "OrderedDict[str, list[Trajectory]]":
    groups: "OrderedDict[str, list[Trajectory]]" = OrderedDict()
    for t in trajs:
        groups.setdefault(t.prompt_id, []).append(t)
    for pid, ts in groups.items():
        k0 = {(t.k[0], t.v_mean[0], t.v_var[0]) for t in ts}
        if len(k0) != 1:
            raise ValueError(f"prompt {pid}: trajectories disagree on the prompt-level estimate Vhat_0")
    return groups


def aggregate(
    trajs: list[Trajectory], n_bins: int = 10, exclude_last_segment: bool = False
) -> PromptAggregate:
    """Aggregate debiased squared increments into relative-position bins, per prompt.

    Each segment's d2 is spread uniformly over its tokens (the data identify segment totals only;
    within-segment allocation is a presentation choice, see docs/methods.md §4).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    groups = group_by_prompt(trajs)
    P = len(groups)
    rel_mass = np.zeros((P, n_bins))
    total = np.zeros(P)
    last = np.zeros(P)
    var_r = np.zeros(P)
    v0 = np.zeros(P)
    n_traj = np.zeros(P, dtype=int)
    for i, (pid, ts) in enumerate(groups.items()):
        for t in ts:
            t.validate()
            seg = segment_table(t)
            keep = ~seg.is_last if exclude_last_segment else np.ones_like(seg.is_last)
            N = float(t.n_tokens)
            ov = interval_overlap(seg.start / N, seg.end / N, edges)  # rows sum to seg length / N
            frac = ov / ((seg.end - seg.start) / N)[:, None]
            rel_mass[i] += (seg.d2[:, None] * frac)[keep].sum(axis=0)
            total[i] += seg.d2[keep].sum()
            last[i] += seg.d2[seg.is_last].sum()
        n = len(ts)
        rel_mass[i] /= n
        total[i] /= n
        last[i] /= n
        var_r[i] = ts[0].v_var[0]
        v0[i] = ts[0].v_mean[0]
        n_traj[i] = n
    return PromptAggregate(list(groups.keys()), rel_mass, total, last, var_r, v0, n_traj, edges)


# ----------------------------------------------------------------------------------------------
# Resolution curve, front-loading index, bootstrap
# ----------------------------------------------------------------------------------------------


def shares_from(rel_mass: np.ndarray, total: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Pooled (ratio-of-sums) share of resolved variance per bin. weights: prompt multiplicities."""
    if weights is None:
        return rel_mass.sum(axis=0) / total.sum()
    return (weights @ rel_mass) / (weights @ total)[..., None]


def cumulative_curve(shares: np.ndarray) -> np.ndarray:
    """C at the bin edges: C(0)=0, C(1)=sum(shares)."""
    z = np.zeros(shares.shape[:-1] + (1,))
    return np.concatenate([z, np.cumsum(shares, axis=-1)], axis=-1)


def front_loading_index(shares: np.ndarray) -> np.ndarray:
    """F = integral_0^1 (C(u) - u) du for the piecewise-linear C implied by equal-width bins.

    F > 0: variance is resolved earlier than uniform; F < 0: later. Range [-1/2, 1/2].
    Requires shares to sum to 1 (true for the pooled estimator).
    """
    C = cumulative_curve(shares)
    B = shares.shape[-1]
    return ((C[..., :-1] + C[..., 1:]) / 2.0).sum(axis=-1) / B - 0.5


def first_half_excess(shares: np.ndarray) -> np.ndarray:
    """Share of variance resolved in the first half of the response, minus 1/2 (needs even #bins)."""
    B = shares.shape[-1]
    if B % 2:
        raise ValueError("first_half_excess needs an even number of bins")
    return shares[..., : B // 2].sum(axis=-1) - 0.5


@dataclass
class CurveResult:
    edges: np.ndarray
    shares: np.ndarray
    shares_ci: np.ndarray  # [2, B]
    curve: np.ndarray  # C at edges
    curve_ci: np.ndarray  # [2, B+1]
    F: float
    F_ci: tuple[float, float]
    F_p_le_0: float  # bootstrap fraction of replicates with F <= 0 (one-sided)
    first_half: float
    first_half_ci: tuple[float, float]
    last_segment_share: float
    last_segment_share_ci: tuple[float, float]
    conservation_ratio: float  # sum_x total_x / sum_x var_r_x ; should be ~1
    conservation_ci: tuple[float, float]
    n_prompts: int
    n_traj: int


def bootstrap_counts(n_units: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster-bootstrap multiplicities: each row resamples the n_units prompts with replacement."""
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=n_boot).astype(float)


def resolution_curve(
    agg: PromptAggregate, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> CurveResult:
    P, B = agg.rel_mass.shape
    if P < 2:
        raise ValueError("need at least 2 prompts")
    rng = np.random.default_rng(seed)
    W = bootstrap_counts(P, n_boot, rng)
    sh = shares_from(agg.rel_mass, agg.total)
    sh_b = shares_from(agg.rel_mass, agg.total, W)
    F = float(front_loading_index(sh))
    F_b = front_loading_index(sh_b)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    out_first = float(first_half_excess(sh)) if B % 2 == 0 else float("nan")
    fh_b = first_half_excess(sh_b) if B % 2 == 0 else np.full(n_boot, np.nan)
    last = float(agg.last_mass.sum() / agg.total.sum())
    last_b = (W @ agg.last_mass) / (W @ agg.total)
    cons = float(agg.total.sum() / agg.var_r.sum())
    cons_b = (W @ agg.total) / (W @ agg.var_r)
    return CurveResult(
        edges=agg.edges,
        shares=sh,
        shares_ci=np.percentile(sh_b, q, axis=0),
        curve=cumulative_curve(sh),
        curve_ci=np.percentile(cumulative_curve(sh_b), q, axis=0),
        F=F,
        F_ci=tuple(np.percentile(F_b, q)),
        F_p_le_0=float(np.mean(F_b <= 0.0)),
        first_half=out_first,
        first_half_ci=tuple(np.percentile(fh_b, q)),
        last_segment_share=last,
        last_segment_share_ci=tuple(np.percentile(last_b, q)),
        conservation_ratio=cons,
        conservation_ci=tuple(np.percentile(cons_b, q)),
        n_prompts=P,
        n_traj=int(agg.n_traj.sum()),
    )


# ----------------------------------------------------------------------------------------------
# Best proportional per-token credit, by absolute position (bridge to Stage 2)
# ----------------------------------------------------------------------------------------------


@dataclass
class PositionalCredit:
    edges: np.ndarray  # absolute token-position bin edges
    beta: np.ndarray  # [B] pooled estimate of E[D_t^2 1{N>=t}] / E[A^2 1{N>=t}] averaged over bin
    beta_ci: np.ndarray  # [2, B]
    alive_tokens: np.ndarray  # [B] mean number of trajectory tokens in the bin (survival)


def positional_credit(
    trajs: list[Trajectory],
    bin_width: int,
    max_pos: int | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PositionalCredit:
    """Estimate the MSE-optimal *proportional* token credit  Ahat_t = beta_t * A,  A = R - V_0(x).

    For a token that exists (N >= t), the minimiser of E[(D_t - beta A)^2 | N >= t] is
    beta_t = E[D_t^2 1{N>=t}] / E[A^2 1{N>=t}] (cross terms vanish because {N>=t} is
    F_{t-1}-measurable). Numerator and denominator are summed over the tokens of each bin and
    pooled over prompts (ratio of sums). D_t inside a segment is taken as uniform, i.e. the
    per-token density d2_j / (t_j - t_{j-1}); only segment totals are identified.

    A^2 is debiased with the independent prompt rollouts: E[(R - Vhat_0)^2] = E[A^2] + Var(Vhat_0).
    """
    groups = group_by_prompt(trajs)
    if max_pos is None:
        max_pos = max(t.n_tokens for t in trajs)
    n_bins = int(np.ceil(max_pos / bin_width))
    edges = np.arange(n_bins + 1, dtype=float) * bin_width
    P = len(groups)
    num = np.zeros((P, n_bins))
    den = np.zeros((P, n_bins))
    alive = np.zeros((P, n_bins))
    for i, ts in enumerate(groups.values()):
        for t in ts:
            seg = segment_table(t)
            ov = interval_overlap(seg.start, seg.end, edges)  # tokens of each segment per bin
            num[i] += ((seg.d2 / (seg.end - seg.start))[:, None] * ov).sum(axis=0)
            a2 = (t.reward - t.v_mean[0]) ** 2 - t.v_var[0] / t.k[0]
            tok = interval_overlap(np.array([0.0]), np.array([float(t.n_tokens)]), edges)[0]
            den[i] += a2 * tok
            alive[i] += tok
        num[i] /= len(ts)
        den[i] /= len(ts)
        alive[i] /= len(ts)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = num.sum(0) / den.sum(0)
        rng = np.random.default_rng(seed)
        W = bootstrap_counts(P, n_boot, rng)
        beta_b = (W @ num) / (W @ den)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return PositionalCredit(edges, beta, np.nanpercentile(beta_b, q, axis=0), alive.mean(0))
