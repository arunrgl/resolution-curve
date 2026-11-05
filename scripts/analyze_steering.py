#!/usr/bin/env python3
"""Analysis of H1 steering influence (docs/methods.md §11): binned curves, lag profile, FE regression.

  python scripts/analyze_steering.py --steering runs/pilotA_q05b/steering.jsonl --out runs/pilotA_q05b/steering_analysis
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tcredit.analyze import fe_regression  # noqa: E402
from tcredit.steering import binned_means, steering_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steering", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    with open(args.steering) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    os.makedirs(args.out, exist_ok=True)
    rows = steering_rows(recs, full_window_only=True)
    W = recs[0]["window"]
    rep = {
        "n_trajectories": len(recs),
        "n_prompts": len({r["prompt_id"] for r in recs}),
        "n_positions_full_window": int(len(rows["y"])),
        "by_rel_position": binned_means(rows, "u", np.linspace(0, 1, args.bins + 1), args.n_boot),
        "by_abs_position": binned_means(rows, "t", np.arange(0, rows["t"].max() + 129, 128), args.n_boot),
        "regression_position_only": fe_regression(rows, ["u", "log_t"], n_boot=args.n_boot),
        "regression_with_entropy (H1-primary)": fe_regression(rows, ["u", "entropy", "log_t"], n_boot=args.n_boot),
    }
    lag = np.zeros(W)
    cnt = np.zeros(W)
    for r in recs:
        for l in r["lag"]:
            lag[: len(l)] += l
            cnt[: len(l)] += 1
    rep["lag_profile_mean_KL"] = (lag / np.maximum(cnt, 1)).tolist()
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    b = rep["by_rel_position"]
    lines = [f"trajectories={rep['n_trajectories']} prompts={rep['n_prompts']} positions(full window)={rep['n_positions_full_window']}",
             "mean S/W by relative position bin (95% CI):"]
    for i in range(len(b["mean"])):
        lines.append(f"  [{b['edges'][i]:.1f},{b['edges'][i+1]:.1f})  {b['mean'][i]:.4f}  [{b['ci_low'][i]:.4f}, {b['ci_high'][i]:.4f}]  n={b['n'][i]}")
    for k in ("regression_position_only", "regression_with_entropy (H1-primary)"):
        g = rep[k]
        if "error" in g:
            lines.append(f"{k}: {g['error']}")
            continue
        lines.append(f"{k}:")
        for c, v, lo, hi in zip(g["covariates"], g["coef"], g["ci_low"], g["ci_high"]):
            lines.append(f"  {c:8s} {v:+.4e}  [{lo:+.4e}, {hi:+.4e}]")
    lp = rep["lag_profile_mean_KL"]
    lines.append("lag profile mean KL at d=1,2,4,8,16,32,64: " + " ".join(f"{lp[d-1]:.4f}" for d in (1, 2, 4, 8, 16, 32, 64) if d <= len(lp)))
    txt = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
