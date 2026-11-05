"""Generation backends.

A backend turns GenRequest(prompt_ids, prefix_ids, n, max_new_tokens) into n i.i.d. continuations
sampled from the *same* policy and decoding settings for every request. Outputs contain content
tokens only: a trailing stop/EOS token is stripped and reported through `finished` (see
tcredit/checkpoints.py for the position convention).

On-policy requirements (violations break the martingale and show up as a conservation ratio
far from 1 in the analysis):
  * identical sampling parameters for base trajectories and continuations;
  * continuation from token ids, never from re-tokenized text;
  * no penalties that treat prompt and output tokens differently (presence/frequency penalty);
  * no stop strings (they cut text mid-token); only stop *tokens*;
  * a fixed total budget: a continuation from prefix length t gets max_new_tokens - t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class GenRequest:
    prompt_ids: list[int]
    prefix_ids: list[int]
    n: int
    max_new_tokens: int


@dataclass
class GenOutput:
    token_ids: list[int]  # continuation content tokens (stop token stripped)
    finished: bool  # True if generation ended with a stop token, False if it hit the budget


class Backend(Protocol):
    def prompt_ids(self, prompt: dict) -> list[int]: ...

    def token_pieces(self, ids: list[int]) -> list[str]: ...

    def generate(self, requests: Sequence[GenRequest]) -> list[list[GenOutput]]: ...


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables (vLLM >= 0.10 convention; checked against vLLM 0.33.0 source)
    min_p: float = 0.0
    repetition_penalty: float = 1.0


class VLLMBackend:
    """vLLM offline-inference backend (API checked against the vLLM 0.33.0 source).

    Notes verified in vLLM 0.33.0:
      * LLM.generate returns outputs in the same order as the input prompts;
      * a dict {"prompt_token_ids": [...]} is a valid TokensPrompt;
      * a finished-by-stop-token CompletionOutput.token_ids *includes* the stop token
        (v1/engine/detokenizer.py re-appends it); we strip it here.
    """

    def __init__(
        self,
        model: str,
        sampling: SamplingConfig,
        *,
        tokenizer: str | None = None,
        chat_template: bool = True,
        system_prompt: str | None = None,
        user_template: str = "{problem}",
        seed: int = 0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
    ) -> None:
        from vllm import LLM  # imported lazily so the rest of the package works without vLLM

        if sampling.temperature <= 0:
            raise ValueError("temperature must be > 0: the measurement needs a stochastic policy")
        self.sampling = sampling
        self.chat_template = chat_template
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer or model,
            seed=seed,
            enable_prefix_caching=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
        )
        self.tok = self.llm.get_tokenizer()
        self.stop_ids = self._collect_stop_ids(model)
        mc = getattr(self.llm, "model_config", None)
        self.max_model_len = getattr(mc, "max_model_len", None)
        self.stats = {"generated_tokens": 0, "requests": 0, "samples": 0}

    def _collect_stop_ids(self, model: str) -> set[int]:
        ids: set[int] = set()
        if self.tok.eos_token_id is not None:
            ids.add(int(self.tok.eos_token_id))
        try:
            from transformers import GenerationConfig

            gc = GenerationConfig.from_pretrained(model)
            e = gc.eos_token_id
            if isinstance(e, int):
                ids.add(e)
            elif e is not None:
                ids.update(int(x) for x in e)
        except Exception:
            pass
        return ids

    # ------------------------------------------------------------------ text <-> ids
    def prompt_ids(self, prompt: dict) -> list[int]:
        user = self.user_template.format(problem=prompt["problem"])
        if self.chat_template:
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": user})
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            # the template already contains BOS/special tokens where the model expects them
            return list(self.tok.encode(text, add_special_tokens=False))
        return list(self.tok.encode(user, add_special_tokens=True))

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def token_pieces(self, ids: list[int]) -> list[str]:
        return [self.tok.decode([i], skip_special_tokens=False) for i in ids]

    # ------------------------------------------------------------------ generation
    def generate(self, requests: Sequence[GenRequest]) -> list[list[GenOutput]]:
        from vllm import SamplingParams

        s = self.sampling
        prompts, params = [], []
        for r in requests:
            ids = list(r.prompt_ids) + list(r.prefix_ids)
            if r.max_new_tokens < 1:
                raise ValueError("max_new_tokens must be >= 1")
            if self.max_model_len is not None and len(r.prompt_ids) + len(r.prefix_ids) + r.max_new_tokens > self.max_model_len:
                raise ValueError(
                    "prompt + budget exceeds max_model_len; continuations would be capped differently "
                    "from base trajectories. Raise --max-model-len or lower --max-new-tokens."
                )
            prompts.append({"prompt_token_ids": ids})
            params.append(
                SamplingParams(
                    n=r.n,
                    temperature=s.temperature,
                    top_p=s.top_p,
                    top_k=s.top_k,
                    min_p=s.min_p,
                    repetition_penalty=s.repetition_penalty,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    max_tokens=r.max_new_tokens,
                    skip_special_tokens=True,
                )
            )
        self.stats["requests"] += len(requests)
        self.stats["samples"] += sum(r.n for r in requests)
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        results: list[list[GenOutput]] = []
        for o in outs:
            row = []
            for c in o.outputs:
                ids = list(c.token_ids)
                if c.finish_reason == "stop":
                    if isinstance(c.stop_reason, str):
                        raise RuntimeError("stop strings are not supported (they cut text mid-token)")
                    self.stats["generated_tokens"] += len(ids)
                    if ids and (ids[-1] in self.stop_ids or ids[-1] == c.stop_reason):
                        ids = ids[:-1]
                    row.append(GenOutput(ids, True))
                elif c.finish_reason == "length":
                    self.stats["generated_tokens"] += len(ids)
                    row.append(GenOutput(ids, False))
                else:
                    raise RuntimeError(f"unexpected finish_reason {c.finish_reason!r}")
            results.append(row)
        return results
