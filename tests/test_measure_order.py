"""measure_chunk must split a request's samples into V0 rollouts and base trajectories at random,
not by the order the backend returns them (corrections 21)."""

from tcredit.backends import GenOutput
from tcredit.measure import Stage1Config, measure_chunk


class OrderedBackend:
    """Returns samples tagged by output index: sample j has the single token j (+ EOS)."""

    def prompt_ids(self, prompt):
        return [1]

    def token_pieces(self, ids):
        return [str(i) for i in ids]

    def generate(self, requests):
        return [[GenOutput([j], True) for j in range(r.n)] for r in requests]


def test_base_trajectories_are_not_the_last_outputs():
    cfg = Stage1Config(k0=8, m_base=2, k=2, max_new_tokens=4, interval=1, seed=0)
    prompts = [{"id": f"p{i}", "answer": ""} for i in range(60)]
    late = lambda p, ids, fin: float(ids[0] >= cfg.k0)  # 1 if the sample came from the last m_base slots
    trajs = measure_chunk(OrderedBackend(), prompts, late, cfg)
    frac = sum(t.reward for t in trajs) / len(trajs)
    # random split: P(base sample is from the last slots) = m_base / (k0 + m_base) = 0.2; no shuffle: 1.0
    assert frac < 0.4, frac
