"""End-to-end: measurement loop + estimators recover exact quantities on toy processes."""

import json

import numpy as np
import pytest

from tcredit.checkpoints import delimiter_checkpoints, fixed_interval
from tcredit.estimator import aggregate, positional_credit, resolution_curve
from tcredit.measure import Stage1Config, measure_chunk, run_measurement, select_prompts
from tcredit.records import read_jsonl
from tcredit.toy import ToyBackend, ToyProcess, exact_rel_bin_mass, exact_summary, toy_reward_fn

BUDGET = 10


def make_processes():
    procs = []
    rng = np.random.default_rng(123)
    for i in range(10):
        front = i % 2 == 0
        a = (4, 3, 2, 2, 1, 1, 1, 1, 1, 1) if front else (1, 1, 1, 1, 1, 1, 2, 2, 3, 4)
        procs.append(
            ToyProcess(
                a=a,
                q=tuple(float(x) for x in rng.uniform(0.35, 0.65, BUDGET)),
                theta=int(rng.integers(4, 7)),
                h_below=float(rng.uniform(0.02, 0.15)),
                h_above=float(rng.uniform(0.15, 0.6)),
                min_len=3,
            )
        )
    return procs


PROCS = make_processes()
PROMPTS = [{"id": f"p{i}", "toy_index": i, "problem": "", "answer": ""} for i in range(len(PROCS))]


@pytest.fixture(scope="module")
def measured():
    backend = ToyBackend(PROCS, seed=7)
    cfg = Stage1Config(k0=64, m_base=400, k=8, max_new_tokens=BUDGET, rule="fixed", interval=2)
    trajs = measure_chunk(backend, PROMPTS, toy_reward_fn(PROCS), cfg)
    return cfg, trajs


def test_all_trajectories_valid(measured):
    cfg, trajs = measured
    assert len(trajs) == len(PROMPTS) * cfg.m_base
    for t in trajs:
        t.validate()
        assert t.checkpoints[1:-1] == fixed_interval(t.n_tokens, cfg.interval)


def test_resolution_shares_match_exact(measured):
    cfg, trajs = measured
    B = 5
    rule = lambda c, f, N: fixed_interval(N, cfg.interval)
    exact_mass = sum(exact_rel_bin_mass(p, rule, B) for p in PROCS)
    exact_var = sum(p.value(0, 0) * (1 - p.value(0, 0)) for p in PROCS)
    exact_shares = exact_mass / exact_var
    res = resolution_curve(aggregate(trajs, n_bins=B), n_boot=500)
    assert exact_shares.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.max(np.abs(res.shares - exact_shares)) < 0.03, (res.shares, exact_shares)
    assert 0.93 < res.conservation_ratio < 1.07, res.conservation_ratio
    assert res.conservation_ci[0] < 1.0 < res.conservation_ci[1]


def test_positional_credit_matches_exact():
    backend = ToyBackend(PROCS, seed=11)
    cfg = Stage1Config(k0=64, m_base=400, k=8, max_new_tokens=BUDGET, rule="fixed", interval=1)
    trajs = measure_chunk(backend, PROMPTS, toy_reward_fn(PROCS), cfg)
    est = positional_credit(trajs, bin_width=1, max_pos=BUDGET, n_boot=300)
    num = np.zeros(BUDGET)
    den = np.zeros(BUDGET)
    for p in PROCS:
        s = exact_summary(p)
        num += s.per_token_sq
        den += s.per_token_a2_alive
    exact_beta = num / den
    early = slice(0, 6)  # later positions have few surviving trajectories and noisy ratios
    assert np.max(np.abs(est.beta[early] - exact_beta[early])) < 0.03, (est.beta, exact_beta)


def test_delimiter_rule_in_pipeline():
    backend = ToyBackend(PROCS, seed=3)
    cfg = Stage1Config(k0=16, m_base=50, k=4, max_new_tokens=BUDGET, rule="delimiter", min_gap=2)
    trajs = measure_chunk(backend, PROMPTS[:3], toy_reward_fn(PROCS), cfg)
    for t in trajs:
        # reconstruct content is not stored; check the rule's invariants instead
        cps = t.checkpoints[1:-1]
        assert all(b - a >= 2 for a, b in zip([0] + cps, cps))
        assert all(0 < c < t.n_tokens for c in cps)


def test_selection_band_and_resume(tmp_path):
    backend = ToyBackend(PROCS, seed=5)
    cfg = Stage1Config(n_select=16, select_lo=0.2, select_hi=0.8, k0=8, m_base=2, k=4,
                       max_new_tokens=BUDGET, interval=3, chunk_prompts=3, max_prompts=6)
    kept = select_prompts(backend, PROMPTS, toy_reward_fn(PROCS), cfg)
    assert 0 < len(kept) <= 6
    assert all(0.2 <= p["p_select"] <= 0.8 for p in kept)
    out = tmp_path / "m.jsonl"
    run_measurement(backend, kept[:2], toy_reward_fn(PROCS), cfg, str(out), log=lambda s: None)
    run_measurement(backend, kept, toy_reward_fn(PROCS), cfg, str(out), log=lambda s: None)
    trajs = list(read_jsonl(str(out)))
    ids = [t.prompt_id for t in trajs]
    assert sorted(set(ids)) == sorted(p["id"] for p in kept)
    assert len(ids) == len(kept) * cfg.m_base  # nothing measured twice


def test_report_and_figure(measured, tmp_path):
    from tcredit.analyze import build_report, summarize
    from tcredit.plots import plot_report

    _, trajs = measured
    rep = build_report(trajs, n_bins=10, abs_bin_width=2, n_boot=200)
    assert rep["conservation_ok"]
    assert "conservation_ratio" not in rep["curve_excluding_last_segment"]
    assert abs(sum(rep["curve_all_segments"]["shares"]) - 1) < 1e-9
    assert "u_mid" in summarize(rep)
    plot_report(rep, str(tmp_path / "fig.png"))
    assert (tmp_path / "fig.png").stat().st_size > 10_000


def test_inconsistent_backend_capping_is_rejected():
    from tcredit.backends import GenOutput

    class Bad:
        def prompt_ids(self, p):
            return [1]

        def token_pieces(self, ids):
            return ["x"] * len(ids)

        def generate(self, reqs):
            # "finished" with no room left for the stop token: impossible under a fixed budget
            return [[GenOutput([0] * r.max_new_tokens, True) for _ in range(r.n)] for r in reqs]

    cfg = Stage1Config(k0=2, m_base=1, k=2, max_new_tokens=5)
    with pytest.raises(RuntimeError):
        measure_chunk(Bad(), [{"id": "a"}], lambda p, ids, f: 0.0, cfg)
