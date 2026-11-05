"""Unit tests: checkpoint rules, rewards, records, regression, and the vLLM backend plumbing."""

import sys
import types

import numpy as np
import pytest

from tcredit.analyze import fe_regression
from tcredit.checkpoints import delimiter_checkpoints, fixed_interval, n_positions
from tcredit.data import gsm8k_answer
from tcredit.estimator import front_loading_index, interval_overlap
from tcredit.records import Trajectory, mc_stats
from tcredit.rewards import answer_region, answers_match, last_boxed, math_reward

# ------------------------------------------------------------------ checkpoints


def test_n_positions():
    assert n_positions(5, True) == 6 and n_positions(5, False) == 5


def test_fixed_interval():
    assert fixed_interval(10, 3) == [3, 6, 9]
    assert fixed_interval(9, 3) == [3, 6]  # never at N
    assert fixed_interval(10, 3, max_checkpoints=2) == [3, 6]
    assert fixed_interval(2, 5) == []


def test_delimiter_split_across_tokens_and_gap():
    pieces = ["a", "\n", "\nb", "c", "\n\n", "d", "e", "\n\n", "f"]
    # completions of "\n\n": inside token 3 (t=3), token 5 (t=5), token 8 (t=8)
    assert delimiter_checkpoints(pieces, n_tokens=10, min_gap=1) == [3, 5, 8]
    assert delimiter_checkpoints(pieces, n_tokens=10, min_gap=3) == [3, 8]
    assert delimiter_checkpoints(pieces, n_tokens=8, min_gap=1) == [3, 5]  # t < N
    assert delimiter_checkpoints(pieces, n_tokens=10, min_gap=1, max_checkpoints=1) == [3]


def test_delimiter_no_double_count_inside_one_piece():
    assert delimiter_checkpoints(["x\n\n\n\ny", "z"], n_tokens=3) == [1]


# ------------------------------------------------------------------ rewards


def test_last_boxed_nested_and_truncated():
    assert last_boxed(r"so \boxed{\frac{1}{2}} and later \boxed{x^{2}+1}") == "x^{2}+1"
    assert last_boxed(r"\boxed{3") is None
    assert last_boxed("no box") is None
    assert last_boxed(r"\fbox{7}") == "7"


def test_answer_region():
    assert answer_region("think \\boxed{1}</think> final \\boxed{2}") == " final \\boxed{2}"
    assert answer_region("plain \\boxed{2}") == "plain \\boxed{2}"


@pytest.mark.parametrize(
    "gold,pred,ok",
    [("\\frac{1}{2}", "\\frac12", True), ("\\frac{1}{2}", "0.5", True), ("18", "18", True),
     ("18", "19", False), ("\\sqrt{2}", "\\sqrt 2", True), ("(1,2)", "(1, 2)", True)],
)
def test_answers_match(gold, pred, ok):
    assert answers_match(gold, pred) is ok


def test_math_reward_requires_finished_and_box():
    assert math_reward("4", "2+2=\\boxed{4}", True) == 1.0
    assert math_reward("4", "2+2=\\boxed{4}", False) == 0.0
    assert math_reward("4", "2+2=\\boxed{4}", False, require_finished=False) == 1.0
    assert math_reward("4", "2+2=4", True) == 0.0
    assert math_reward("4", "\\boxed{4}</think>\\boxed{5}", True) == 0.0  # graded after </think>


def test_gsm8k_answer():
    assert gsm8k_answer("blah\n#### 1,234") == "1234"


# ------------------------------------------------------------------ records / estimator pieces


def test_mc_stats_unbiased_variance():
    k, m, s2 = mc_stats([1, 0, 0, 1])
    assert (k, m) == (4, 0.5) and s2 == pytest.approx(1 / 3)


def test_trajectory_validation():
    ok = Trajectory("p", 0, 3, True, 1.0, [0, 2, 3], [0.5, 0.8, 1.0], [0.25, 0.1, 0.0], [8, 8, 0])
    ok.validate()
    bad = Trajectory("p", 0, 3, True, 1.0, [0, 2, 3], [0.5, 0.8, 0.0], [0.25, 0.1, 0.0], [8, 8, 0])
    with pytest.raises(ValueError):
        bad.validate()
    with pytest.raises(ValueError):
        Trajectory("p", 0, 3, True, 1.0, [0, 3, 3], [0.5, 0.8, 1.0], [0.2, 0.1, 0.0], [8, 8, 0]).validate()


def test_interval_overlap_counts_tokens():
    ov = interval_overlap(np.array([0.0, 3.0]), np.array([3.0, 7.0]), np.array([0.0, 2.0, 4.0, 8.0]))
    np.testing.assert_allclose(ov, [[2, 1, 0], [0, 1, 3]])


def test_front_loading_index_extremes():
    B = 10
    assert front_loading_index(np.full(B, 1 / B)) == pytest.approx(0.0)
    first = np.zeros(B)
    first[0] = 1
    last = np.zeros(B)
    last[-1] = 1
    assert front_loading_index(first) == pytest.approx(0.5 - 0.05)
    assert front_loading_index(last) == pytest.approx(-0.5 + 0.05)


def test_fe_regression_recovers_coefficients():
    rng = np.random.default_rng(0)
    P, n = 60, 40
    prompt = np.repeat(np.arange(P), n).astype(float)
    x1 = rng.normal(size=P * n)
    x2 = rng.normal(size=P * n) * 3
    fe = np.repeat(rng.normal(size=P) * 5, n)
    y = 2.0 * x1 - 0.5 * x2 + fe + rng.normal(size=P * n) * 0.1
    r = fe_regression({"prompt": prompt, "y": y, "x1": x1, "x2": x2}, ["x1", "x2"], n_boot=200)
    # coefficients are per SD of the covariate
    assert r["coef"][0] == pytest.approx(2.0 * x1.std(), rel=0.02)
    assert r["coef"][1] == pytest.approx(-0.5 * x2.std(), rel=0.02)
    assert r["ci_low"][0] < r["coef"][0] < r["ci_high"][0]


# ------------------------------------------------------------------ vLLM backend with a stand-in module


class _FakeTok:
    eos_token_id = 99

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        return "<s>" + "|".join(m["content"] for m in msgs) + "<a>"

    def encode(self, text, add_special_tokens):
        return [ord(c) % 50 for c in text] + ([1] if add_special_tokens else [])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + i % 26) for i in ids if not (skip_special_tokens and i == 99))


def _install_fake_vllm(monkeypatch, outputs):
    calls = {}

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class LLM:
        def __init__(self, **kw):
            calls["init"] = kw
            self.model_config = types.SimpleNamespace(max_model_len=4096)

        def get_tokenizer(self):
            return _FakeTok()

        def generate(self, prompts, params, use_tqdm=True):
            calls["prompts"] = prompts
            calls["params"] = params
            return outputs

    mod = types.ModuleType("vllm")
    mod.LLM = LLM
    mod.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return calls


def test_vllm_backend_plumbing(monkeypatch):
    from tcredit.backends import GenRequest, SamplingConfig, VLLMBackend

    C = lambda ids, fr, sr=None: types.SimpleNamespace(token_ids=ids, finish_reason=fr, stop_reason=sr)
    outs = [types.SimpleNamespace(outputs=[C([5, 6, 99], "stop"), C([5, 6, 7], "length")]),
            types.SimpleNamespace(outputs=[C([8, 42], "stop", 42), C([8], "stop")])]
    calls = _install_fake_vllm(monkeypatch, outs)
    be = VLLMBackend("m", SamplingConfig(temperature=0.7, top_p=0.95), max_model_len=4096)
    assert calls["init"]["enable_prefix_caching"] is True
    pid = be.prompt_ids({"problem": "hi"})
    res = be.generate([GenRequest(pid, [], 2, 3), GenRequest(pid, [5], 2, 2)])
    assert calls["prompts"][0] == {"prompt_token_ids": pid}
    assert calls["prompts"][1] == {"prompt_token_ids": pid + [5]}
    p0, p1 = calls["params"]
    assert (p0.n, p0.max_tokens, p0.temperature, p0.top_p) == (2, 3, 0.7, 0.95)
    assert p0.presence_penalty == 0.0 and p0.frequency_penalty == 0.0
    assert p1.max_tokens == 2
    assert [(o.token_ids, o.finished) for o in res[0]] == [([5, 6], True), ([5, 6, 7], False)]
    # stop token reported via stop_reason is stripped; a stop without a trailing stop id keeps tokens
    assert [(o.token_ids, o.finished) for o in res[1]] == [([8], True), ([8], True)]


def test_vllm_backend_rejects_stop_strings_and_overlong(monkeypatch):
    from tcredit.backends import GenRequest, SamplingConfig, VLLMBackend

    C = types.SimpleNamespace(token_ids=[1], finish_reason="stop", stop_reason="</s>")
    _install_fake_vllm(monkeypatch, [types.SimpleNamespace(outputs=[C])])
    be = VLLMBackend("m", SamplingConfig())
    with pytest.raises(RuntimeError):
        be.generate([GenRequest([1], [], 1, 2)])
    with pytest.raises(ValueError):
        be.generate([GenRequest([1] * 4000, [], 1, 200)])
    with pytest.raises(ValueError):
        VLLMBackend("m", SamplingConfig(temperature=0.0))


# ------------------------------------------------------------------ entropy index alignment (no torch needed)


def _fake_causal_logits(tokens, V=11):
    """Row i depends only on tokens[: i + 1], like a causal LM."""
    rows = []
    for i in range(len(tokens)):
        h = sum((j + 1) * (tok + 3) for j, tok in enumerate(tokens[: i + 1]))
        rows.append(np.array([np.sin(h * (v + 1)) * 3 for v in range(V)]))
    return np.stack(rows)


def _entropy(z):
    z = z - z.max()
    p = np.exp(z) / np.exp(z).sum()
    return float(-(p * np.log(p)).sum())


@pytest.mark.parametrize("finished", [True, False])
def test_distribution_rows_align_with_stepwise_sampling(finished):
    from tcredit.entropy import distribution_rows

    prompt, content = [4, 1, 7], [2, 9, 5, 5, 0]
    start, stop = distribution_rows(len(prompt), len(content), finished)
    full = _fake_causal_logits(prompt + content)
    teacher_forced = [_entropy(r) for r in full[start:stop]]
    n = len(content) + (1 if finished else 0)
    stepwise = [_entropy(_fake_causal_logits(prompt + content[: t - 1])[-1]) for t in range(1, n + 1)]
    assert len(teacher_forced) == n
    np.testing.assert_allclose(teacher_forced, stepwise, atol=1e-12)
