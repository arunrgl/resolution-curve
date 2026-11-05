#!/usr/bin/env python3
"""Analyze Stage 1 measurements: resolution curve, controls, positional-credit table, figure.

Usage:
  python scripts/analyze_stage1.py --trajectories runs/x/trajectories.jsonl --out runs/x/analysis
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tcredit.analyze import build_report, summarize  # noqa: E402
from tcredit.records import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--bins", type=int, default=10, help="relative-position bins (even number)")
    ap.add_argument("--abs-bin-width", type=int, default=256, help="token bin width for the positional-credit table")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    trajs = list(read_jsonl(args.trajectories))
    if not trajs:
        sys.exit("no trajectories found")
    os.makedirs(args.out, exist_ok=True)
    rep = build_report(trajs, n_bins=args.bins, abs_bin_width=args.abs_bin_width, n_boot=args.n_boot, seed=args.seed)
    rep["inputs"] = {"trajectories": os.path.abspath(args.trajectories), **{k: v for k, v in vars(args).items() if k != "trajectories"}}
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    text = summarize(rep)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    if not args.no_plot:
        from tcredit.plots import plot_report

        plot_report(rep, os.path.join(args.out, "resolution_curve.png"), title=args.title)
        print(f"figure: {os.path.join(args.out, 'resolution_curve.png')}")


if __name__ == "__main__":
    main()
