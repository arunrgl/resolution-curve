"""Stage 1 analysis: resolution curves, controls, and the positional-credit table.

All inferential statements use a cluster bootstrap over prompts (the sampling unit).
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from .estimator import (
    CurveResult,
    aggregate,
    bootstrap_counts,
    group_by_prompt,
    positional_credit,
    resolution_curve,
    segment_table,
)
from .records import Trajectory


# ----------------------------------------------------------------------------------------------
# Segment-level regression with prompt fixed effects
# ----------------------------------------------------------------------------------------------


def segment_rows(trajs: list[Trajectory]) -> dict[str, np.ndarray]:
    """One row per segment: outcome = debiased per-token resolution density d2 / length."""
    groups = group_by_prompt(trajs)
    cols: dict[str, list] = {k: [] for k in ("prompt", "y", "u_mid", "log_len", "is_last", "ent_mean", "ent_max", "abs_mid")}
    for p_idx, ts in enumerate(groups.values()):
        for t in ts:
            seg = segment_table(t)
            N = float(t.n_tokens)
            length = seg.end - seg.start
            for j in range(len(seg.d2)):
                cols["prompt"].append(p_idx)
                cols["y"].append(seg.d2[j] / length[j])
                cols["u_mid"].append((seg.start[j] + seg.end[j]) / 2.0 / N)
                cols["abs_mid"].append((seg.start[j] + seg.end[j]) / 2.0)
                cols["log_len"].append(math.log(length[j]))
                cols["is_last"].append(float(seg.is_last[j]))
                if t.entropy is not None:
                    e = np.asarray(t.entropy[int(seg.start[j]) : int(seg.end[j])], dtype=float)
                    cols["ent_mean"].append(float(e.mean()))
                    cols["ent_max"].append(float(e.max()))
                else:
                    cols["ent_mean"].append(np.nan)
                    cols["ent_max"].append(np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def fe_regression(
    rows: dict[str, np.ndarray], covariates: list[str], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """OLS of y on z-scored covariates with prompt fixed effects (within-prompt demeaning).

    CIs: cluster bootstrap over prompts. Demeaning is invariant to duplicating a prompt, so each
    replicate is a weighted least-squares fit with weights = prompt multiplicities.
    """
    mask = np.all(np.isfinite(np.column_stack([rows[c] for c in covariates] + [rows["y"]])), axis=1)
    if mask.sum() < len(covariates) + 2:
        return {"error": "not enough complete rows", "n_rows": int(mask.sum())}
    prompt = rows["prompt"][mask].astype(int)
    X = np.column_stack([rows[c][mask] for c in covariates])
    sd = X.std(axis=0)
    binary = [set(np.unique(X[:, i])) <= {0.0, 1.0} for i in range(X.shape[1])]
    scale = np.where(np.array(binary) | (sd == 0), 1.0, sd)
    X = (X - X.mean(axis=0)) / scale
    y = rows["y"][mask].copy()
    uniq, inv = np.unique(prompt, return_inverse=True)
    P, d = len(uniq), X.shape[1]
    for p in range(P):
        m = inv == p
        X[m] -= X[m].mean(axis=0)
        y[m] -= y[m].mean()
    XtX = np.zeros((P, d, d))
    Xty = np.zeros((P, d))
    for p in range(P):
        m = inv == p
        XtX[p] = X[m].T @ X[m]
        Xty[p] = X[m].T @ y[m]
    beta = np.linalg.lstsq(XtX.sum(0), Xty.sum(0), rcond=None)[0]
    W = bootstrap_counts(P, n_boot, np.random.default_rng(seed))
    bs = np.array([np.linalg.lstsq(np.tensordot(w, XtX, 1), w @ Xty, rcond=None)[0] for w in W])
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    lo, hi = np.percentile(bs, q, axis=0)
    return {
        "covariates": covariates,
        "units": ["per SD" if not b else "indicator" for b in binary],
        "coef": beta.tolist(),
        "ci_low": lo.tolist(),
        "ci_high": hi.tolist(),
        "n_rows": int(mask.sum()),
        "n_prompts": int(P),
    }


# ----------------------------------------------------------------------------------------------
# Robustness: length strata (descriptive; conditioning on N looks into the future)
# ----------------------------------------------------------------------------------------------


def length_strata(trajs: list[Trajectory], n_bins: int, n_strata: int = 3, n_boot: int = 1000, seed: int = 0) -> list[dict]:
    Ns = np.array([t.n_tokens for t in trajs])
    cuts = np.quantile(Ns, np.linspace(0, 1, n_strata + 1))
    out = []
    for s in range(n_strata):
        lo, hi = cuts[s], cuts[s + 1]
        sub = [t for t in trajs if (lo <= t.n_tokens <= hi if s == n_strata - 1 else lo <= t.n_tokens < hi)]
        if len({t.prompt_id for t in sub}) < 2:
            continue
        r = resolution_curve(aggregate(sub, n_bins=n_bins), n_boot=n_boot, seed=seed)
        out.append({"length_range": [float(lo), float(hi)], "n_traj": r.n_traj, "n_prompts": r.n_prompts,
                    "shares": r.shares.tolist(), "F": r.F, "F_ci": list(r.F_ci)})
    return out


# ----------------------------------------------------------------------------------------------
# Full report
# ----------------------------------------------------------------------------------------------


def _curve_dict(r: CurveResult) -> dict:
    d = asdict(r)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else (list(v) if isinstance(v, tuple) else v)) for k, v in d.items()}


def build_report(
    trajs: list[Trajectory], n_bins: int = 10, abs_bin_width: int = 256, n_boot: int = 2000, seed: int = 0
) -> dict:
    rep: dict = {"n_prompts": len(group_by_prompt(trajs)), "n_traj": len(trajs)}
    main = resolution_curve(aggregate(trajs, n_bins=n_bins), n_boot=n_boot, seed=seed)
    rep["curve_all_segments"] = _curve_dict(main)
    excl = _curve_dict(
        resolution_curve(aggregate(trajs, n_bins=n_bins, exclude_last_segment=True), n_boot=n_boot, seed=seed)
    )
    for k in ("conservation_ratio", "conservation_ci", "last_segment_share", "last_segment_share_ci"):
        excl.pop(k)  # not meaningful once the final segment is excluded
    rep["curve_excluding_last_segment"] = excl
    lo, hi = main.conservation_ci
    rep["conservation_ok"] = bool(lo <= 1.0 <= hi)
    rep["length_strata"] = length_strata(trajs, n_bins, n_boot=min(n_boot, 1000), seed=seed)
    pc = positional_credit(trajs, bin_width=abs_bin_width, n_boot=n_boot, seed=seed)
    rep["positional_credit"] = {
        "bin_edges": pc.edges.tolist(),
        "beta": pc.beta.tolist(),
        "beta_ci_low": pc.beta_ci[0].tolist(),
        "beta_ci_high": pc.beta_ci[1].tolist(),
        "mean_alive_tokens": pc.alive_tokens.tolist(),
    }
    rows = segment_rows(trajs)
    rep["regression_position_only"] = fe_regression(rows, ["u_mid", "log_len", "is_last"], n_boot=n_boot, seed=seed)
    if np.isfinite(rows["ent_mean"]).any():
        rep["regression_with_entropy"] = fe_regression(
            rows, ["u_mid", "log_len", "is_last", "ent_mean", "ent_max"], n_boot=n_boot, seed=seed
        )
    else:
        rep["regression_with_entropy"] = {"error": "no entropy in records; run scripts/add_entropy.py"}
    return rep


def summarize(rep: dict) -> str:
    c = rep["curve_all_segments"]
    cx = rep["curve_excluding_last_segment"]
    lines = [
        f"prompts={rep['n_prompts']}  trajectories={rep['n_traj']}",
        f"conservation ratio (should be ~1): {c['conservation_ratio']:.3f}  95% CI [{c['conservation_ci'][0]:.3f}, {c['conservation_ci'][1]:.3f}]"
        + ("" if rep["conservation_ok"] else "   <-- CHECK: continuations may be off-policy or rewards non-deterministic"),
        f"front-loading index F (all segments):      {c['F']:+.4f}  95% CI [{c['F_ci'][0]:+.4f}, {c['F_ci'][1]:+.4f}]",
        f"front-loading index F (excl. last segment): {cx['F']:+.4f}  95% CI [{cx['F_ci'][0]:+.4f}, {cx['F_ci'][1]:+.4f}]",
        f"share resolved in final segment: {c['last_segment_share']:.3f}  95% CI [{c['last_segment_share_ci'][0]:.3f}, {c['last_segment_share_ci'][1]:.3f}]",
        "shares by relative-position bin: " + " ".join(f"{s:.3f}" for s in c["shares"]),
    ]
    for name in ("regression_position_only", "regression_with_entropy"):
        r = rep[name]
        if "error" in r:
            lines.append(f"{name}: {r['error']}")
            continue
        i = r["covariates"].index("u_mid")
        lines.append(f"{name}: coef(u_mid) = {r['coef'][i]:+.3e}  95% CI [{r['ci_low'][i]:+.3e}, {r['ci_high'][i]:+.3e}]")
    return "\n".join(lines)
