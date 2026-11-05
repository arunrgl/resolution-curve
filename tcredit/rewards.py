"""Deterministic verifiable rewards for math answers.

R must be a deterministic function of (prompt, finished response); otherwise V_N != R and the
variance decomposition does not hold. Everything here is deterministic.
"""

from __future__ import annotations

import re


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}) with balanced braces, else None."""
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None
    i = starts[-1]
    depth = 1
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i:j].strip()
        j += 1
    return None  # unbalanced (e.g. truncated)


def answer_region(text: str, after_think: bool = True) -> str:
    """For reasoning models, grade only what follows the last </think> if one is present."""
    if after_think and "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def _normalize(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(?:left|right|displaystyle|,|;|!|\s)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = s.replace(" ", "").rstrip(".")
    return s


def answers_match(gold: str, pred: str) -> bool:
    """Math-Verify equivalence if available, else normalized string equality (documented fallback)."""
    try:
        from math_verify import parse, verify

        g = parse(f"${gold}$")
        p = parse(f"${pred}$")
        if g and p:
            return bool(verify(g, p))
    except Exception:
        pass
    return _normalize(gold) == _normalize(pred)


def math_reward(
    gold: str, response_text: str, finished: bool, require_finished: bool = True, after_think: bool = True
) -> float:
    if require_finished and not finished:
        return 0.0
    pred = last_boxed(answer_region(response_text, after_think))
    if pred is None:
        return 0.0
    return 1.0 if answers_match(gold, pred) else 0.0


def make_math_reward_fn(decode, require_finished: bool = True, after_think: bool = True):
    """Build reward(prompt, content_ids, finished) for a backend's `decode`."""

    def reward(prompt: dict, content_ids: list[int], finished: bool) -> float:
        return math_reward(prompt["answer"], decode(list(content_ids)), finished, require_finished, after_think)

    return reward
