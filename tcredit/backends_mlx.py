"""MLX (Apple silicon) generation backend, same contract as backends.VLLMBackend.

API checked against the installed mlx 0.32.4 / mlx-lm 0.32.2 source:
  * mlx_lm.load(repo) on an original Hugging Face checkpoint keeps the stored dtype and only
    quantizes if config.json carries a quantization config (utils.load_model). We refuse any model
    that ends up with quantized layers: a quantized model is a different policy.
  * mlx_lm.sample_utils.make_sampler applies top-p / min-p / top-k to the *untempered*
    log-probs and divides by the temperature only in the final categorical draw. vLLM and HF apply
    the temperature first. We therefore use our own sampler (`sample_logits`), temperature first.
    With top_p=1, top_k=0, min_p=0 (the recommended settings, methods §6) the two coincide.
  * TokenizerWrapper.apply_chat_template injects a thinking kwarg; we call the underlying HF
    tokenizer's template directly so the prompt is what HF/vLLM would build.
  * models.cache.BatchKVCache stores left-padded rows with per-row RoPE offsets and masks the
    padding (base.create_causal_mask); merge/extend/filter are used for the batched path.

Numerics: in bf16, prefill (many tokens per kernel call) and incremental decoding give slightly
different next-token distributions (measured on Qwen2.5-0.5B-Instruct: total variation up to ~0.03,
mean ~0.003; in fp32 the same comparison gives ~4e-6). Continuations start with a prefill, base
trajectories decode incrementally, so in bf16 they are very slightly off-policy relative to each
other. dtype="float32" upcasts the stored weights exactly and removes this.

Two generation paths, which must sample from the same distribution:
  * "simple": every sample gets its own full prefill and its own single-row decode loop.
  * "batched" (default): each request's prompt+prefix is prefilled once, its KV cache is copied
    into n rows, and rows from many requests decode together in one left-padded batch.
tests/test_mlx_backend.py checks both against exact enumeration on a tiny random model, and
checks that batched-path logits match a single full forward pass.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .backends import GenOutput, GenRequest, SamplingConfig


def sample_logits(logits, sampling: SamplingConfig):
    """One token per row of `logits` ([B, V]) under temperature -> top-k -> top-p -> min-p.

    Filters act on the tempered distribution (vLLM/HF convention). top-k keeps every token tied
    with the k-th largest logit; top-p keeps the smallest prefix of the sorted distribution whose
    mass reaches top_p (the token that crosses the threshold is kept); min-p keeps tokens with
    p >= min_p * p_max.
    """
    import mlx.core as mx

    x = logits.astype(mx.float32) / sampling.temperature
    neg_inf = mx.array(-float("inf"), dtype=mx.float32)
    V = x.shape[-1]
    if sampling.top_k and 0 < sampling.top_k < V:
        kth = mx.min(mx.topk(x, sampling.top_k, axis=-1), axis=-1, keepdims=True)
        x = mx.where(x < kth, neg_inf, x)
    if sampling.top_p < 1.0:
        order = mx.argsort(-x, axis=-1)
        xs = mx.take_along_axis(x, order, axis=-1)
        ps = mx.softmax(xs, axis=-1)
        mass_before = mx.cumsum(ps, axis=-1) - ps
        keep = mass_before < sampling.top_p
        thr = mx.min(mx.where(keep, xs, mx.array(float("inf"), dtype=mx.float32)), axis=-1, keepdims=True)
        x = mx.where(x < thr, neg_inf, x)
    if sampling.min_p > 0.0:
        p = mx.softmax(x, axis=-1)
        x = mx.where(p < sampling.min_p * mx.max(p, axis=-1, keepdims=True), neg_inf, x)
    return mx.random.categorical(x, axis=-1)


@dataclass
class _Row:
    req: int
    slot: int  # sample index within the request: outputs are returned in slot order, not finish order
    tokens: list
    budget: int
    dead: bool = False


class MLXBackend:
    """mlx-lm backend. Construct with a Hugging Face repo id or a local path of the *original*
    (fp16/bf16/fp32) checkpoint; use `from_loaded` to wrap an already-built model (tests)."""

    def __init__(
        self,
        model: str,
        sampling: SamplingConfig,
        *,
        revision: str | None = None,
        chat_template: bool = True,
        system_prompt: str | None = None,
        user_template: str = "{problem}",
        seed: int = 0,
        mode: str = "batched",
        max_batch: int = 96,
        prefill_step: int = 512,
        max_model_len: int | None = None,
        dtype: str = "auto",
    ) -> None:
        import mlx.core as mx
        from mlx_lm.utils import _download, load_model, load_tokenizer

        if dtype not in ("auto", "float32"):
            raise ValueError("dtype must be 'auto' (stored weights) or 'float32' (lossless upcast); "
                             "down-casting would change the policy")
        path = _download(model, revision=revision)
        mlx_model, config = load_model(path)
        if dtype == "float32":
            mlx_model.set_dtype(mx.float32)  # bf16/fp16 -> fp32 is exact; only the arithmetic changes
            mx.eval(mlx_model.parameters())
        hf_tok = load_tokenizer(path)._tokenizer
        stop_ids = _collect_stop_ids(path, hf_tok)
        limit = max_model_len or config.get("max_position_embeddings")
        self._init(mlx_model, hf_tok, stop_ids, sampling, chat_template, system_prompt, user_template,
                   seed, mode, max_batch, prefill_step, limit)
        self.model_path = str(path)
        self.model_name = model
        # snapshot directories are .../snapshots/<commit>; record it for the manifest
        self.model_revision = os.path.basename(os.path.normpath(path)) if "snapshots" in str(path) else revision
        self.model_dtype = _check_unquantized(mlx_model)

    @classmethod
    def from_loaded(cls, mlx_model, hf_tokenizer, stop_ids, sampling: SamplingConfig, **kw) -> "MLXBackend":
        self = cls.__new__(cls)
        self._init(mlx_model, hf_tokenizer, set(stop_ids), sampling, kw.get("chat_template", True),
                   kw.get("system_prompt"), kw.get("user_template", "{problem}"), kw.get("seed", 0),
                   kw.get("mode", "batched"), kw.get("max_batch", 96), kw.get("prefill_step", 512),
                   kw.get("max_model_len"))
        self.model_path = self.model_name = self.model_revision = None
        self.model_dtype = _check_unquantized(mlx_model)
        return self

    def _init(self, mlx_model, tok, stop_ids, sampling, chat_template, system_prompt, user_template,
              seed, mode, max_batch, prefill_step, max_model_len) -> None:
        import mlx.core as mx

        if sampling.temperature <= 0:
            raise ValueError("temperature must be > 0: the measurement needs a stochastic policy")
        if sampling.repetition_penalty != 1.0:
            raise ValueError("repetition_penalty is not supported by the MLX backend")
        if mode not in ("simple", "batched"):
            raise ValueError("mode must be 'simple' or 'batched'")
        if not stop_ids:
            raise ValueError("no stop token ids found")
        self.model = mlx_model
        self.tok = tok
        self.stop_ids = set(int(i) for i in stop_ids)
        self.sampling = sampling
        self.chat_template = chat_template
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.mode = mode
        self.max_batch = max_batch
        self.prefill_step = prefill_step
        self.slack = None  # dead rows tolerated before compaction; None -> max_batch // 4
        self.max_model_len = max_model_len
        self.stats = {"generated_tokens": 0, "prefill_tokens": 0, "prefill_seconds": 0.0, "requests": 0, "samples": 0}
        mx.random.seed(seed)
        # MLX keeps freed buffers in its allocator cache (default limit: the whole memory limit).
        # Compaction allocates fresh multi-GB cache arrays, so an uncapped cache grows until the
        # machine swaps (observed: 24 GB resident for a 1.5B model). Cap it.
        mx.set_cache_limit(int(float(os.environ.get("TCREDIT_MLX_CACHE_GB", "4")) * 2**30))
        self._stop_arr = mx.array(sorted(self.stop_ids))

    # ------------------------------------------------------------------ text <-> ids
    def prompt_ids(self, prompt: dict) -> list[int]:
        user = self.user_template.format(problem=prompt["problem"])
        if self.chat_template:
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": user})
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            return list(self.tok.encode(text, add_special_tokens=False))
        return list(self.tok.encode(user, add_special_tokens=True))

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def token_pieces(self, ids: list[int]) -> list[str]:
        return [self.tok.decode([i], skip_special_tokens=False) for i in ids]

    # ------------------------------------------------------------------ model primitives
    def _prefill(self, ids: list[int]):
        """Run ids through a fresh single-row cache; return (cache, logits at the last position [1, V])."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        if not ids:
            raise ValueError("empty prompt")
        t0 = time.perf_counter()
        cache = make_prompt_cache(self.model)
        x = mx.array(ids)[None]
        logits = None
        for a in range(0, len(ids), self.prefill_step):
            out = self.model(x[:, a : a + self.prefill_step], cache=cache)
            logits = out[:, -1, :]
            mx.eval(logits, [c.state for c in cache])
        self.stats["prefill_tokens"] += len(ids)
        self.stats["prefill_seconds"] += time.perf_counter() - t0
        return cache, logits

    def _check(self, r: GenRequest) -> list[int]:
        if r.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if r.n < 1:
            raise ValueError("n must be >= 1")
        ids = list(r.prompt_ids) + list(r.prefix_ids)
        if self.max_model_len is not None and len(ids) + r.max_new_tokens > self.max_model_len:
            raise ValueError(
                "prompt + budget exceeds max_model_len; continuations would be capped differently "
                "from base trajectories. Lower --max-new-tokens."
            )
        return ids

    # ------------------------------------------------------------------ generation
    def generate(self, requests: Sequence[GenRequest]) -> list[list[GenOutput]]:
        reqs = [(self._check(r), r.n, r.max_new_tokens) for r in requests]
        self.stats["requests"] += len(reqs)
        self.stats["samples"] += sum(n for _, n, _ in reqs)
        if self.mode == "simple":
            return [[self._sample_one(ids, budget) for _ in range(n)] for ids, n, budget in reqs]
        return self._generate_batched(reqs, lambda logits, rows: sample_logits(logits, self.sampling))

    def _sample_one(self, ids: list[int], budget: int) -> GenOutput:
        """Reference path: one sample, own prefill, single-row decoding."""
        import mlx.core as mx

        cache, logits = self._prefill(ids)
        out: list[int] = []
        while True:
            t = int(sample_logits(logits, self.sampling).item())
            if t in self.stop_ids:
                self.stats["generated_tokens"] += len(out) + 1
                return GenOutput(out, True)
            out.append(t)
            if len(out) == budget:
                self.stats["generated_tokens"] += len(out)
                return GenOutput(out, False)
            logits = self.model(mx.array([[t]]), cache=cache)[:, -1, :]

    def _generate_batched(
        self,
        reqs: list[tuple[list[int], int, int]],
        choose: Callable,
    ) -> list[list[GenOutput]]:
        """Batched path. `choose(logits [B, V], rows) -> tokens [B]` is the sampler; tests pass a
        function that forces tokens and records logits, exercising exactly this loop."""
        import mlx.core as mx
        from mlx_lm.models.cache import BatchKVCache

        # Outputs go to fixed slots. Appending in finish order would sort each request's samples by
        # length/finish time, and callers that split a request's samples by index (measure_chunk:
        # first k0 for V0, the rest as base trajectories) would get a length-biased split.
        results: list[list] = [[None] * n for _, n, _ in reqs]
        # work items (request, first slot, number of rows), each at most max_batch rows; pop() takes them in order
        queue = [(q, a, min(self.max_batch, n - a)) for q, (_, n, _) in enumerate(reqs) for a in range(0, n, self.max_batch)]
        queue.reverse()
        # Finished rows stay in the batch as dead slots (their outputs are ignored) until `slack`
        # of them accumulate: BatchKVCache.filter/extend copy the whole batch cache, so compacting
        # and admitting in bulk is much cheaper than doing it every time a row finishes.
        slack = self.slack or max(1, self.max_batch // 4)
        rows: list[_Row] = []
        bcache = None
        logits = None
        while True:
            live = sum(not r.dead for r in rows)
            dead = len(rows) - live
            free = self.max_batch - live
            admit = bool(queue) and queue[-1][2] <= free and (live == 0 or free >= min(slack, queue[-1][2]))
            if dead and (admit or dead >= slack or live == 0):
                if live == 0:
                    rows, bcache, logits = [], None, None
                else:
                    k = mx.array([i for i, r in enumerate(rows) if not r.dead])
                    for c in bcache:
                        c.filter(k)
                    logits = logits[k]
                    rows = [r for r in rows if not r.dead]
                mx.clear_cache()
            if admit:
                caches, lgs = [], []
                while queue and queue[-1][2] <= self.max_batch - len(rows):
                    q, a0, n = queue.pop()
                    ids, _, budget = reqs[q]
                    cache, lg = self._prefill(ids)
                    caches.extend([cache] * n)
                    lgs.append(mx.repeat(lg, n, axis=0))
                    rows.extend(_Row(q, a0 + i, [], budget) for i in range(n))
                new = [BatchKVCache.merge([c[layer] for c in caches]) for layer in range(len(caches[0]))]
                lg = mx.concatenate(lgs, axis=0)
                if bcache is None:
                    bcache, logits = new, lg
                else:
                    for x, y in zip(bcache, new):
                        x.extend(y)
                    logits = mx.concatenate([logits, lg], axis=0)
            if not rows:
                break
            toks = choose(logits, rows)
            mx.eval(toks)
            for row, t in zip(rows, toks.tolist()):
                if row.dead:
                    continue
                if t in self.stop_ids:
                    results[row.req][row.slot] = GenOutput(row.tokens, True)
                    self.stats["generated_tokens"] += len(row.tokens) + 1
                    row.dead = True
                    continue
                row.tokens.append(t)
                if len(row.tokens) == row.budget:
                    results[row.req][row.slot] = GenOutput(row.tokens, False)
                    self.stats["generated_tokens"] += len(row.tokens)
                    row.dead = True
            if all(r.dead for r in rows) and not queue:
                break
            logits = self.model(toks[:, None], cache=bcache)[:, -1, :]
        for res in results:
            if any(o is None for o in res):
                raise RuntimeError("internal error: missing samples")
        return results


def _collect_stop_ids(path, hf_tok) -> set[int]:
    """Union of the tokenizer's eos_token_id and eos_token_id from config.json and generation_config.json."""
    ids: set[int] = set()
    if hf_tok.eos_token_id is not None:
        ids.add(int(hf_tok.eos_token_id))
    for name in ("config.json", "generation_config.json"):
        p = os.path.join(str(path), name)
        if os.path.exists(p):
            with open(p) as f:
                e = json.load(f).get("eos_token_id")
            if isinstance(e, int):
                ids.add(e)
            elif e is not None:
                ids.update(int(x) for x in e)
    return ids


def _check_unquantized(mlx_model) -> str:
    """Refuse quantized models; return the parameter dtype (e.g. 'bfloat16')."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    for _, m in mlx_model.named_modules():
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            raise ValueError("quantized model: use the original fp16/bf16 checkpoint (a quantized model is a different policy)")
    dtypes = {str(v.dtype) for _, v in tree_flatten(mlx_model.parameters())}
    return ",".join(sorted(d.replace("mlx.core.", "") for d in dtypes))
