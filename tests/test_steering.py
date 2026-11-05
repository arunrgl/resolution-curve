"""Steering influence (methods §11): batched prefix-sharing computation vs naive forward passes."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tcredit.steering import binned_means, naive_window_kl, steering_for_trajectory, steering_rows  # noqa: E402

V = 8


@pytest.fixture(scope="module")
def model():
    from mlx_lm.models.qwen2 import Model, ModelArgs

    mx.random.seed(0)
    return Model(ModelArgs(model_type="qwen2", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                           num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=V, num_key_value_heads=2))


def test_matches_naive_and_identity_swap_is_zero(model):
    prompt, content = [1, 2, 3], [4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 3, 2]
    T, W, stride = 0.7, 4, 3
    alts_by_t = {}

    def alt_fn(p, t):
        a = [content[t - 1], (content[t - 1] % (V - 1)) + 1, 7 if content[t - 1] != 7 else 6]
        alts_by_t[t] = a
        return a

    out = steering_for_trajectory(model, prompt, content, stop_ids={0}, temperature=T, stride=stride,
                                  window=W, prefill_step=2, alt_fn=alt_fn)
    assert out["t"] == list(range(1, len(content), stride))
    for t, w, S, lag, ns in zip(out["t"], out["w"], out["S"], out["lag"], out["n_same"]):
        assert w == min(W, len(content) - t)
        ref = naive_window_kl(model, prompt + content[: t - 1], content[t - 1], alts_by_t[t], content[t : t + w - 1], T)
        assert abs(ref[0]).max() < 1e-5  # swapping in the real token changes nothing
        assert ns >= 1
        np.testing.assert_allclose(lag, ref.mean(0), atol=1e-5)
        np.testing.assert_allclose(S, ref.sum(-1).mean(), atol=1e-5)


def test_sampled_alternatives_never_stop(model):
    out = steering_for_trajectory(model, [1, 2], [3, 4, 5, 6, 7, 3], stop_ids={0, 1, 2, 3, 4, 5, 6},
                                  temperature=1.0, stride=1, window=2, swaps=6)
    # only token 7 is allowed, so every alternative is 7
    assert out["n_same"] == [0, 0, 0, 0, 6]


def test_rows_and_bins():
    recs = [{"prompt_id": "a", "n_tokens": 100, "window": 10, "t": [1, 50, 95], "w": [10, 10, 4],
             "entropy": [1.0, 2.0, 3.0], "S": [5.0, 2.0, 1.0]},
            {"prompt_id": "b", "n_tokens": 40, "window": 10, "t": [1, 17], "w": [10, 10],
             "entropy": [0.5, 0.5], "S": [3.0, 1.0]}]
    rows = steering_rows(recs)
    assert len(rows["y"]) == 4  # the truncated window is dropped
    np.testing.assert_allclose(sorted(rows["y"]), [0.1, 0.2, 0.3, 0.5])
    b = binned_means(rows, "u", np.array([0, 0.5, 1.0]), n_boot=50)
    np.testing.assert_allclose(b["mean"], [(0.5 + 0.3 + 0.1) / 3, 0.2])
