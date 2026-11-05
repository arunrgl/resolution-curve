#!/bin/zsh
# Rebuild colab/resolution-curve-code.zip (code + data needed on Colab; no runs, no venv).
set -e
cd "$(dirname "$0")/.."
rm -f colab/resolution-curve-code.zip
zip -q -r colab/resolution-curve-code.zip tcredit scripts tests docs requirements.txt README.md HANDOFF.md \
  data/math500.jsonl data/math500_pool100_seed20261103.jsonl data/math500_pool150_seed20261103.jsonl \
  data/pilotB_selected_prompts.jsonl -x "*/__pycache__/*" "*.pyc"
ls -la colab/resolution-curve-code.zip
