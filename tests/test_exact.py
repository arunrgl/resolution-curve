"""Exact (enumeration-based) checks of the identities the study relies on."""

import numpy as np
import pytest

from tcredit.checkpoints import delimiter_checkpoints, fixed_interval
from tcredit.toy import ToyProcess, exact_segment_sum, exact_summary

FRONT = ToyProcess(a=(4, 3, 2, 1, 1, 1, 1, 1, 1, 1), q=(0.5,) * 10, theta=5, h_below=0.1, h_above=0.4, min_len=3)
BACK = ToyProcess(a=(1, 1, 1, 1, 1, 1, 1, 2, 3, 4), q=(0.5,) * 10, theta=5, h_below=0.05, h_above=0.3, min_len=3)
FLAT_LEN = ToyProcess(a=(2,) * 9, q=(0.4,) * 9, theta=6, h_below=0.15, h_above=0.15, min_len=2)
PROCS = [FRONT, BACK, FLAT_LEN]


def _pieces(content):
    return ["\n\n" if y == 1 else "x" for y in content]


@pytest.mark.parametrize("proc", PROCS)
def test_probabilities_sum_to_one(proc):
    assert sum(p for *_, p in proc.paths()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("proc", PROCS)
def test_value_function_matches_enumeration(proc):
    # V_0 from backward induction equals P(R=1) from enumerating complete paths.
    pr = sum(p * proc.reward(c, f) for c, f, p in proc.paths())
    assert proc.value(0, 0) == pytest.approx(pr, abs=1e-12)


@pytest.mark.parametrize("proc", PROCS)
def test_martingale_variance_decomposition(proc):
    s = exact_summary(proc)
    assert s.var_r == pytest.approx(s.v0 * (1 - s.v0), abs=1e-12)
    assert s.sum_sq_increments == pytest.approx(s.var_r, abs=1e-12)


@pytest.mark.parametrize("proc", PROCS)
@pytest.mark.parametrize("interval", [1, 2, 3, 4, 7])
def test_fixed_interval_checkpoints_conserve_variance(proc, interval):
    rule = lambda c, f, N: fixed_interval(N, interval)
    assert exact_segment_sum(proc, rule) == pytest.approx(proc.value(0, 0) * (1 - proc.value(0, 0)), abs=1e-12)


@pytest.mark.parametrize("proc", PROCS)
@pytest.mark.parametrize("min_gap", [1, 2, 3])
def test_delimiter_checkpoints_conserve_variance(proc, min_gap):
    rule = lambda c, f, N: delimiter_checkpoints(_pieces(c), N, "\n\n", min_gap)
    v0 = proc.value(0, 0)
    assert exact_segment_sum(proc, rule) == pytest.approx(v0 * (1 - v0), abs=1e-12)


def test_relative_position_checkpoints_break_the_decomposition():
    """Cutting at fixed *fractions of the final length* uses the future (N) and is biased.

    With outcome-dependent stopping (successes stop sooner) the error is ~10% of Var(R) here.
    """
    proc = ToyProcess(a=(4, 3, 2, 1, 1, 1, 1, 1, 1, 1), q=(0.5,) * 10, theta=5, h_below=0.02, h_above=0.9, min_len=2)
    J = 3
    rule = lambda c, f, N: [int(round(j * N / J)) for j in range(1, J)]
    v0 = proc.value(0, 0)
    rel_gap = (exact_segment_sum(proc, rule) - v0 * (1 - v0)) / (v0 * (1 - v0))
    assert rel_gap > 0.05, rel_gap
    # ...while a stopping-time rule on the same process is exact
    assert exact_segment_sum(proc, lambda c, f, N: fixed_interval(N, 3)) == pytest.approx(v0 * (1 - v0), abs=1e-12)


@pytest.mark.parametrize("proc", PROCS)
def test_increment_covariance_with_sequence_advantage(proc):
    # E[Delta_t * A] = E[Delta_t^2] for every t  =>  best linear predictor of Delta_t from A is c_t A.
    s = exact_summary(proc)
    np.testing.assert_allclose(s.cov_delta_a, s.per_token_sq, atol=1e-12)
    assert s.per_token_sq.sum() / s.var_r == pytest.approx(1.0, abs=1e-12)


def test_debiased_increment_is_unbiased_and_raw_is_not():
    rng = np.random.default_rng(0)
    content, finished = [1, 0, 1, 1, 0, 0, 1], True
    v = FRONT.value_path(content, finished)
    cps = [0, 2, 4, 6, len(v) - 1]
    true = np.diff(v[cps]) ** 2
    k, reps = 4, 400_000
    vh = np.empty((reps, len(cps)))
    for j, t in enumerate(cps[:-1]):
        vh[:, j] = rng.binomial(k, v[t], size=reps) / k
    vh[:, -1] = v[-1]  # exact terminal
    s2 = vh * (1 - vh) * k / (k - 1)  # unbiased sample variance of k Bernoulli draws
    noise = s2 / k
    noise[:, -1] = 0.0
    raw = np.diff(vh, axis=1) ** 2
    deb = raw - noise[:, 1:] - noise[:, :-1]
    se = deb.std(axis=0) / np.sqrt(reps)
    assert np.all(np.abs(deb.mean(axis=0) - true) < 5 * se + 1e-12)
    assert np.all(raw.mean(axis=0) - true > 10 * se)  # raw squares are inflated by MC noise
