# Data

Prompt files are generated, not stored in the repository:

```bash
python scripts/prepare_data.py --preset math500 --out data/math500.jsonl
python scripts/make_pools.py
python scripts/make_pools.py --selected results/stage1/qwen2.5-1.5b-instruct/selected_ids.json data/pilotB_selected_prompts.jsonl
```

`make_pools.py` writes `math500_pool100_seed20261103.jsonl` and `math500_pool150_seed20261103.jsonl`
(the first 100 / 150 prompts of a fixed permutation of MATH-500), byte-identical to the files used
in the runs. The last line rebuilds the 50-prompt selection used by the Colab notebook's step A.
