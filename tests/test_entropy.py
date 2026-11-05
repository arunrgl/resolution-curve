"""Index alignment of the teacher-forced entropy pass, on a tiny randomly initialised model.

Teacher-forced entropies must equal the entropy of the next-token distribution computed one
prefix at a time. Skipped when torch/transformers are not installed.
"""

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tcredit.entropy import token_entropies  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    cfg = transformers.LlamaConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
    )
    torch.manual_seed(0)
    m = transformers.LlamaForCausalLM(cfg)
    m.eval()
    return m


def _stepwise(model, prompt, content, finished, temperature):
    out = []
    n = len(content) + (1 if finished else 0)
    for t in range(1, n + 1):
        prefix = prompt + content[: t - 1]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([prefix])).logits[0, -1].double() / temperature
        p = torch.softmax(logits, -1)
        out.append(float(-(p * torch.log(p)).sum()))
    return out


@pytest.mark.parametrize("finished", [True, False])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_teacher_forced_matches_stepwise(tiny_model, finished, temperature):
    prompt = [3, 14, 15, 9]
    content = [26, 5, 35, 89, 79, 32]
    tf = token_entropies(tiny_model, prompt, content, finished, temperature, chunk=4)
    sw = _stepwise(tiny_model, prompt, content, finished, temperature)
    assert len(tf) == len(content) + (1 if finished else 0)
    for a, b in zip(tf, sw):
        assert a == pytest.approx(b, rel=1e-4, abs=1e-5)
    assert all(0 <= e <= math.log(97) + 1e-6 for e in tf)
