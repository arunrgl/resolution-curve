"""Torch steering implementation vs naive per-sequence forward passes (tiny random model, CPU)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tcredit.steering import steering_for_trajectory_torch  # noqa: E402

V = 11


@pytest.fixture(scope="module")
def model():
    cfg = transformers.LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128)
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).eval()


def _lp(model, seq, T):
    with torch.no_grad():
        z = model(input_ids=torch.tensor([seq])).logits[0].double() / T
    return torch.log_softmax(z, -1).numpy()


def test_matches_naive_and_identity_swap_is_zero(model):
    prompt, content, T, W = [1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 3, 2, 1, 4, 5], 0.7, 4
    alts = {}

    def alt_fn(p, t):
        a = [content[t - 1], (content[t - 1] % (V - 1)) + 1, 9]
        alts[t] = a
        return a

    out = steering_for_trajectory_torch(model, prompt, content, stop_ids={0}, temperature=T, stride=3, window=W, alt_fn=alt_fn)
    for t, w, S, lag, ent in zip(out["t"], out["w"], out["S"], out["lag"], out["entropy"]):
        prefix = prompt + content[: t - 1]
        cont = content[t : t + w - 1]
        ref = _lp(model, prefix + [content[t - 1]] + cont, T)[len(prefix):]
        D = []
        for a in alts[t]:
            alt = _lp(model, prefix + [a] + cont, T)[len(prefix):]
            D.append((np.exp(ref) * (ref - alt)).sum(-1))
        D = np.array(D)
        assert abs(D[0]).max() < 1e-6
        np.testing.assert_allclose(lag, D.mean(0), atol=1e-5)
        np.testing.assert_allclose(S, D.sum(-1).mean(), atol=1e-5)
        p0 = _lp(model, prefix, T)[-1]
        assert abs(ent - float(-(np.exp(p0) * p0).sum())) < 1e-5


def test_sampled_alternatives_never_stop(model):
    g = torch.Generator().manual_seed(0)
    out = steering_for_trajectory_torch(model, [1, 2], [3, 4, 5, 6], stop_ids=set(range(V - 1)),
                                        temperature=1.0, stride=1, window=2, swaps=5, generator=g)
    assert out["n_same"] == [0, 0, 0]  # only token V-1 is allowed and it never appears in content


def test_micro_batching_gives_identical_results(model):
    prompt, content = [1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 3, 2, 1, 4, 5]
    alt_fn = lambda p, t: [content[t - 1], 9, 2]
    a = steering_for_trajectory_torch(model, prompt, content, stop_ids={0}, temperature=0.7, stride=3, window=4, alt_fn=alt_fn)
    b = steering_for_trajectory_torch(model, prompt, content, stop_ids={0}, temperature=0.7, stride=3, window=4, alt_fn=alt_fn,
                                      max_batch_tokens=1)  # one row per forward
    np.testing.assert_allclose(a["S"], b["S"], atol=1e-6)
    np.testing.assert_allclose(np.concatenate(a["lag"]), np.concatenate(b["lag"]), atol=1e-6)
