"""MLX backend: contract, numerics, and sampling-distribution tests.

Most tests use a tiny randomly initialised Qwen2 model built from mlx-lm's own model class, so the
real cache / batching / sampling code paths run in seconds and the exact distribution of short
responses can be enumerated. Tests marked `real` load Qwen/Qwen2.5-0.5B-Instruct from the local
Hugging Face cache and are skipped if it is not there.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tcredit.backends import GenRequest, SamplingConfig  # noqa: E402
from tcredit.backends_mlx import MLXBackend, sample_logits  # noqa: E402
from tcredit.measure import Stage1Config, _check_budget, measure_chunk  # noqa: E402

V = 8
STOP = {0}


class _Tok:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [int(c) % V for c in text if c.isdigit()] or [1]


def _tree_scale(m, s):
    from mlx.utils import tree_map

    m.update(tree_map(lambda w: w * s, m.parameters()))
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module")
def model():
    import mlx.nn  # noqa: F401
    from mlx_lm.models.qwen2 import Model, ModelArgs

    mx.random.seed(0)
    args = ModelArgs(
        model_type="qwen2", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=V, num_key_value_heads=2,
    )
    return _tree_scale(Model(args), 1.0)  # default init: spread-out, context-dependent distributions


def backend(model, sampling=None, stop=STOP, **kw):
    return MLXBackend.from_loaded(model, _Tok(), stop, sampling or SamplingConfig(temperature=1.0), **kw)


def full_logits(model, ids):
    """Logits at every position from one forward pass with no cache, float32 numpy [L, V]."""
    out = model(mx.array(ids)[None])[0].astype(mx.float32)
    return np.array(out)


# ---------------------------------------------------------------- exact sampler reference (numpy)
def ref_probs(logits: np.ndarray, s: SamplingConfig) -> np.ndarray:
    """Independent numpy implementation of temperature -> top-k -> top-p -> min-p."""
    x = logits.astype(np.float64) / s.temperature
    if s.top_k and 0 < s.top_k < len(x):
        kth = np.sort(x)[::-1][s.top_k - 1]
        x = np.where(x < kth, -np.inf, x)
    if s.top_p < 1.0:
        p = np.exp(x - x.max()); p /= p.sum()
        order = np.argsort(-x, kind="stable")
        before = np.cumsum(p[order]) - p[order]
        kept = order[before < s.top_p]
        mask = np.full(len(x), True); mask[kept] = False
        x = np.where(mask, -np.inf, x)
    p = np.exp(x - x.max()); p /= p.sum()
    if s.min_p > 0:
        x = np.where(p < s.min_p * p.max(), -np.inf, x)
        p = np.exp(x - x.max()); p /= p.sum()
    return p


def chi2_pvalue(counts: np.ndarray, probs: np.ndarray) -> float:
    """Pearson chi-square goodness of fit, cells with expected count < 5 pooled; Wilson-Hilferty p."""
    n = counts.sum()
    e = probs * n
    small = e < 5
    c = np.append(counts[~small], counts[small].sum())
    e = np.append(e[~small], e[small].sum())
    if e[-1] == 0:
        c, e = c[:-1], e[:-1]
    if (c[e == 0] > 0).any():
        return 0.0
    x2 = float(((c - e) ** 2 / e).sum())
    df = len(e) - 1
    if df == 0:
        return 1.0
    z = ((x2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


SETTINGS = [
    SamplingConfig(temperature=1.0),
    SamplingConfig(temperature=0.7, top_k=5, top_p=0.9, min_p=0.05),
]


@pytest.mark.parametrize("s", SETTINGS, ids=["plain", "filtered"])
def test_sampler_matches_reference_distribution(s):
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2, size=12).astype(np.float32)
    p = ref_probs(logits, s)
    mx.random.seed(3)
    draws = np.array(sample_logits(mx.broadcast_to(mx.array(logits), (200_000, 12)), s))
    counts = np.bincount(draws, minlength=12)
    assert (counts[p == 0] == 0).all(), "sampled a token the filters exclude"
    assert chi2_pvalue(counts, p) > 1e-3


# ---------------------------------------------------------------- numerics
def test_prefill_matches_incremental_decoding(model):
    ids = [1, 5, 2, 7, 3, 3, 6, 4, 2, 5, 1, 7]
    ref = full_logits(model, ids)
    for split in (1, 5, 11):
        b = backend(model, prefill_step=3)
        cache, lg = b._prefill(ids[:split])
        rows = [np.array(lg[0].astype(mx.float32))]
        for t in ids[split:]:
            lg = model(mx.array([[t]]), cache=cache)[:, -1, :]
            rows.append(np.array(lg[0].astype(mx.float32)))
        got = np.stack(rows)
        np.testing.assert_allclose(got, ref[split - 1 :], atol=1e-4, rtol=1e-4)


def test_batched_path_logits_match_full_forward(model):
    """Teacher-force distinct tokens through the batched loop (left padding, admission of new
    requests mid-stream, a request split across admissions, row filtering) and compare every step's logits with a cache-free pass."""
    b = backend(model, max_batch=5)
    b.slack = 2  # exercise dead slots and bulk compaction
    reqs = [([1, 2, 3], [4], 3, 6), ([5, 6, 7, 1, 2], [3, 3], 2, 3), ([2], [], 7, 5), ([7, 7, 1, 2, 3, 4, 5, 6], [1], 2, 4)]
    rng = np.random.default_rng(0)
    forced = {}
    seen = []

    def choose(logits, rows):
        assert logits.shape[0] == len(rows) <= 5
        toks = []
        for i, r in enumerate(rows):
            if r.dead:
                toks.append(1)
                continue
            key = id(r)
            if key not in forced:
                forced[key] = (r, list(rng.integers(1, V, size=r.budget)))
            t = int(forced[key][1][len(r.tokens)])
            seen.append((r.req, list(r.tokens), np.array(logits[i].astype(mx.float32))))
            toks.append(t)
        return mx.array(toks)

    out = b._generate_batched([(p + q, n, m) for p, q, n, m in reqs], choose)
    for (p, q, n, m), res in zip(reqs, out):
        assert len(res) == n and all(not o.finished and len(o.token_ids) == m for o in res)
    worst = 0.0
    for req, prev, lg in seen:
        ids = reqs[req][0] + reqs[req][1] + prev
        ref = full_logits(model, ids)[-1]
        worst = max(worst, float(np.abs(lg - ref).max()))
    assert worst < 1e-4, worst


@pytest.mark.parametrize("mode", ["simple", "batched"])
@pytest.mark.parametrize("s", SETTINGS, ids=["plain", "filtered"])
def test_paths_sample_the_exact_sequence_distribution(model, mode, s):
    """Enumerate every possible response under a budget of 3 and compare frequencies (chi-square)."""
    budget = 3
    reqs = [([1, 2, 3], [4]), ([5, 6, 7, 1, 2], [3, 3]), ([2], [])]
    n = 4000 if mode == "batched" else 1500
    b = backend(model, s, mode=mode, max_batch=48, seed=11)
    b.slack = 7
    outs = b.generate([GenRequest(p, q, n, budget) for p, q in reqs])
    for (p, q), res in zip(reqs, outs):
        ctx = p + q
        outcomes, probs = [], []
        for L in range(budget + 1):
            for seq in itertools.product(range(1, V), repeat=L):
                ids = ctx + list(seq)
                lg = full_logits(model, ids)
                pr = 1.0
                for j in range(L):
                    pr *= ref_probs(lg[len(ctx) - 1 + j], s)[seq[j]]
                if L < budget:
                    pr *= sum(ref_probs(lg[-1], s)[t] for t in STOP)
                    outcomes.append((seq, True))
                else:
                    outcomes.append((seq, False))
                probs.append(pr)
        probs = np.array(probs)
        assert abs(probs.sum() - 1) < 1e-6
        assert (probs * n >= 5).sum() >= 5, "distribution too concentrated for an informative test"
        index = {o: i for i, o in enumerate(outcomes)}
        counts = np.zeros(len(outcomes))
        for o in res:
            counts[index[(tuple(o.token_ids), o.finished)]] += 1
        assert chi2_pvalue(counts, probs) > 1e-3


# ---------------------------------------------------------------- contract
@pytest.mark.parametrize("mode", ["simple", "batched"])
def test_budget_stop_and_finished_semantics(model, mode):
    stops = {0, 1, 2}  # frequent stops so every case occurs
    b = backend(model, mode=mode, stop=stops, max_batch=16, seed=5)
    for budget in (1, 2, 4):
        outs = b.generate([GenRequest([3, 4], [5], 300, budget), GenRequest([6], [], 300, budget)])
        lens_fin, lens_trunc = set(), set()
        for res in outs:
            assert len(res) == 300
            for o in res:
                assert not (set(o.token_ids) & stops), "stop token left in content"
                if o.finished:
                    assert len(o.token_ids) <= budget - 1
                    lens_fin.add(len(o.token_ids))
                else:
                    assert len(o.token_ids) == budget
                    lens_trunc.add(len(o.token_ids))
                _check_budget(len(o.token_ids), o.finished, budget, "test")
        assert budget - 1 in lens_fin, "a stop sampled at the last slot must count as finished"
        assert lens_trunc == {budget}


def test_rejects_bad_requests(model):
    with pytest.raises(ValueError):
        backend(model, SamplingConfig(temperature=0.0))
    b = backend(model, max_model_len=10)
    with pytest.raises(ValueError):
        b.generate([GenRequest([1] * 6, [2] * 2, 2, 3)])
    with pytest.raises(ValueError):
        b.generate([GenRequest([1], [], 2, 0)])


def test_continues_from_ids_not_text(model):
    """The context fed to the model is exactly prompt_ids + prefix_ids (no re-tokenisation)."""
    b = backend(model, max_batch=8)
    captured = []
    orig = b._prefill

    def spy(ids):
        captured.append(list(ids))
        return orig(ids)

    b._prefill = spy
    b.generate([GenRequest([1, 2], [7, 7, 3], 2, 2)])
    assert captured == [[1, 2, 7, 7, 3]]


# ---------------------------------------------------------------- real model
REAL = "Qwen/Qwen2.5-0.5B-Instruct"


def _real_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download
        from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

        snapshot_download(REAL, local_files_only=True, allow_patterns=DEFAULT_ALLOW_PATTERNS)
        return True
    except Exception:
        return False


real = pytest.mark.skipif(not _real_cached(), reason=f"{REAL} not in the local HF cache")


@pytest.fixture(scope="module")
def real_backend():
    return MLXBackend(REAL, SamplingConfig(temperature=1.0), seed=0, max_batch=32,
                      user_template="{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.")


@real
def test_real_model_is_unquantized_and_template_matches_hf(real_backend):
    from transformers import AutoTokenizer

    assert real_backend.model_dtype in ("bfloat16", "float16", "float32")
    assert real_backend.stop_ids >= {151645}
    hf = AutoTokenizer.from_pretrained(real_backend.model_path)
    p = {"problem": "What is 1+1?"}
    text = hf.apply_chat_template(
        [{"role": "user", "content": real_backend.user_template.format(problem=p["problem"])}],
        add_generation_prompt=True, tokenize=False,
    )
    assert real_backend.prompt_ids(p) == hf.encode(text, add_special_tokens=False)
    assert text.endswith("<|im_start|>assistant\n")


def _prefill_vs_incremental(backend, m):
    ids = backend.prompt_ids({"problem": "Compute 12 * 13."})
    ids = ids + backend.tok.encode(" Let us multiply: 12 * 13 = 12 * 10 + 12 * 3 = 120 + 36", add_special_tokens=False)
    ref = full_logits(m, ids)
    split = len(ids) - 20
    own, backend.model = backend.model, m
    try:
        cache, lg = backend._prefill(ids[:split])
    finally:
        backend.model = own
    rows = [np.array(lg[0].astype(mx.float32))]
    for t in ids[split:]:
        lg = m(mx.array([[t]]), cache=cache)[:, -1, :]
        rows.append(np.array(lg[0].astype(mx.float32)))
    got, want = np.stack(rows), ref[split - 1 :]
    return got, want, 0.5 * np.abs(_softmax(got) - _softmax(want)).sum(-1)


@real
def test_real_prefill_matches_incremental(real_backend):
    """fp32 (exact upcast of the stored weights): the code paths agree to fp tolerance.
    bf16: kernel rounding differs between prefill and decode. The bound is a regression guard set
    from measurement (max TV ~0.03, mean ~0.003 on this prompt), not a correctness criterion."""
    m = real_backend.model
    got, want, tv = _prefill_vs_incremental(real_backend, m)
    assert tv.max() < 0.06 and tv.mean() < 0.01, (tv.max(), tv.mean())
    import copy

    m32 = copy.deepcopy(m)
    m32.set_dtype(mx.float32)
    got, want, tv = _prefill_vs_incremental(real_backend, m32)
    assert np.abs(got - want).max() < 1e-3 and tv.max() < 1e-4, (np.abs(got - want).max(), tv.max())


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


@real
def test_real_two_prompt_measure_chunk_validates(real_backend):
    from tcredit.rewards import make_math_reward_fn

    prompts = [
        {"id": "a", "problem": "What is 7 times 8?", "answer": "56"},
        {"id": "b", "problem": "What is the sum of the first 5 positive odd integers?", "answer": "25"},
    ]
    cfg = Stage1Config(k0=4, m_base=1, k=2, max_new_tokens=160, interval=48)
    reward = make_math_reward_fn(real_backend.decode)
    trajs = measure_chunk(real_backend, prompts, reward, cfg)  # runs Trajectory.validate()
    assert len(trajs) == 2
    for t in trajs:
        assert t.checkpoints[0] == 0 and t.checkpoints[-1] == t.n_tokens
        assert t.prompt_token_ids == real_backend.prompt_ids(next(p for p in prompts if p["id"] == t.prompt_id))


@pytest.mark.parametrize("finished", [True, False])
def test_mlx_entropy_matches_direct(model, finished):
    from tcredit.entropy import token_entropies_mlx

    prompt, content, T = [1, 2, 3], [4, 5, 6, 7, 1, 2, 3, 4, 5], 0.7
    got = token_entropies_mlx(model, prompt, content, finished, T, chunk=4)
    lg = full_logits(model, prompt + content) / T
    lp = lg - lg.max(-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    ent = -(np.exp(lp) * lp).sum(-1)
    # response token t is sampled from row len(prompt) + t - 2; EOS (finished) from the last row
    n = len(content) + (1 if finished else 0)
    want = ent[len(prompt) - 1 : len(prompt) - 1 + n]
    assert len(got) == n
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_batched_outputs_are_in_sample_order_not_finish_order(model):
    """Regression test (corrections 21): outputs used to be appended in finish order, so each
    request's samples came back sorted by length. Within a request, length must not depend on the
    output index. Requests fit in one batch, as in measure_chunk's base request (n = k0 + M)."""
    b = backend(model, stop={0, 1}, max_batch=64, seed=3)
    outs = b.generate([GenRequest([3, 4], [5], 60, 12) for _ in range(5)])
    rs = []
    for out in outs:
        lens = np.array([len(o.token_ids) + o.finished for o in out], dtype=float)
        assert len(set(lens)) > 3
        rs.append(np.corrcoef(np.argsort(np.argsort(lens, kind="stable")), np.arange(len(lens)))[0, 1])
    # under correct ordering each rank correlation is ~N(0, 1/sqrt(59)); finish order gives ~ +0.9
    assert abs(np.mean(rs)) < 0.25, rs
