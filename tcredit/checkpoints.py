"""Checkpoint rules. Every rule here places checkpoints at *stopping times*.

Why it matters (docs/methods.md §2): the variance decomposition sum_j E[D_j^2] = Var(R|x) needs
checkpoint times tau_j such that the event {tau_j <= t} is decided by the prefix y_<=t alone.
Rules that use the final length N (e.g. "cut at 10%, 20%, ... of the response") look into the
future and break the decomposition; tests/test_exact.py shows the bias on a toy process.

Token positions: a response has content tokens 1..L and, if it finished, an EOS token at
position N = L + 1 (if it was truncated, N = L). A checkpoint t means "after token t"; valid
interior checkpoints satisfy 1 <= t < N, i.e. t <= L for finished responses, t < L if truncated.
Both conditions depend only on y_<=t (whether any of y_1..y_t is EOS, and the fixed budget).
"""

from __future__ import annotations


def n_positions(n_content: int, finished: bool) -> int:
    """N: number of token positions including the EOS position for finished responses."""
    return n_content + (1 if finished else 0)


def fixed_interval(n_tokens: int, interval: int, max_checkpoints: int | None = None) -> list[int]:
    """Interior checkpoints at interval, 2*interval, ... strictly before N."""
    if interval < 1:
        raise ValueError("interval must be >= 1")
    ts = list(range(interval, n_tokens, interval))
    if max_checkpoints is not None:
        ts = ts[:max_checkpoints]
    return ts


def delimiter_checkpoints(
    pieces: list[str],
    n_tokens: int,
    delimiter: str = "\n\n",
    min_gap: int = 1,
    max_checkpoints: int | None = None,
) -> list[int]:
    """Checkpoint after any token that completes an occurrence of `delimiter`, at least `min_gap`
    tokens after the previous checkpoint. Decided online from the prefix, so a stopping-time rule.

    pieces: decoded text of each content token (len(pieces) == number of content tokens).
    """
    if not delimiter:
        raise ValueError("empty delimiter")
    out: list[int] = []
    last = 0
    tail = ""  # last len(delimiter)-1 characters seen before the current piece
    keep = len(delimiter) - 1
    for i, piece in enumerate(pieces):
        window = tail + piece
        hit = delimiter in window  # any match here ends inside `piece` (tail is too short to hold one)
        tail = window[-keep:] if keep > 0 else ""
        t = i + 1
        if t >= n_tokens:
            break
        if hit and t - last >= min_gap:
            out.append(t)
            last = t
            if max_checkpoints is not None and len(out) >= max_checkpoints:
                break
    return out
