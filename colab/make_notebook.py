import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""
# Resolution Curve — Stage 1 on a Colab A100 (vLLM backend)

**Before running:** choose *Runtime → Change runtime type → A100 GPU*. Then run the cells in order; the
first cell asks you to upload `resolution-curve-code.zip` (or drag it into the Files pane at `/content/`).

No Google Drive is used. Results are written to `/content/runs/` on the Colab machine, which is wiped
if the session disconnects or is recycled. After each step the notebook writes
`/content/results_<step>.zip` and offers it for download: **download each one as it appears**, then send
the files to Claude. If a session is lost, upload the last results zip together with the code zip and
the notebook resumes from it.

| Step | What | Default |
|---|---|---|
| A | **Validation**: vLLM backend on the same 50 prompts as the Mac 1.5B pilot (first-ever vLLM run), entropy, analysis, steering | on |
| B | **R1-Distill length probe**: 20 prompts × 4 samples, to size step C | on |
| C | **R1-Distill Stage 1 pilot** (30 prompts; ~50M tokens, roughly 2–4 h on an A100) | on |
| E | **Base model** Qwen2.5-1.5B, raw prompt (~15–30 min), runs before C | on |
| D | Zip all results for download | on |

Each step stops with an error message if a command fails. If a cell fails right after installation with an import error, use *Runtime → Restart session* and run again from the top (the install is skipped when already done).
""")
code(r"""
# ---- switches and settings -----------------------------------------------------------------
RUN_A_VALIDATION = False  # done 2026-11-04 (results_A1/A2)
RUN_B_R1_PROBE   = False  # done 2026-11-04 (results_B)
RUN_E_BASE       = False  # done 2026-11-04 (results_E1/E2)
# Step C resumes: generation and analysis are done (results_C2); re-running finishes steering only.
RUN_C_R1_PILOT   = True   # ~2 h per 10-prompt chunk at ~2k tok/s; resumes from an uploaded results_C1_*.zip

WORK      = "/content"
CODE_ZIP  = f"{WORK}/resolution-curve-code.zip"
RUNS      = f"{WORK}/runs"
CODE      = f"{WORK}/code"
VLLM_VERSION = "0.33.0"   # the version the backend's API was checked against

# R1-Distill settings (model card: temperature 0.6 recommended, top-p 0.95; we default to no top-p
# filter so that sampling matches the other runs and the entropy control is exact).
R1_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
R1_TEMPERATURE = 0.6
R1_TOP_P = 1.0
R1_BUDGET = 16384
""")
code(r"""
import os, glob, subprocess
os.chdir(WORK)
if not os.path.exists(CODE_ZIP):
    try:
        from google.colab import files
        print("Upload resolution-curve-code.zip (and, to resume a lost session, the latest results_*.zip):")
        files.upload()
    except Exception as e:
        raise SystemExit(f"Upload resolution-curve-code.zip to {WORK}/ via the Files pane, then re-run this cell ({e})")
assert os.path.exists(CODE_ZIP), f"{CODE_ZIP} not found"
# resume: unpack the newest results zip if one was uploaded and no runs/ exist yet
prev = sorted(glob.glob(f"{WORK}/results_*.zip"), key=os.path.getmtime)
if prev and not os.path.exists(RUNS):
    subprocess.run(["unzip", "-q", "-o", prev[-1], "-d", WORK], check=True)
    print("resumed from", prev[-1])
os.makedirs(RUNS, exist_ok=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
""")
code(r"""
# sh(): run a shell command, stream its output (minus progress bars), stop the notebook on failure.
import subprocess, sys, re
NOISE = re.compile(r"it/s\]|s/it\]|Processed prompts|^\s*$")
def sh(cmd, log=None, ok=(0,)):
    print("$", cmd[:300] + (" ..." if len(cmd) > 300 else ""), flush=True)
    p = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], cwd=CODE if os.path.exists(CODE) else None,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lf = open(log, "a") if log else None
    for line in p.stdout:
        if lf: lf.write(line); lf.flush()
        if not NOISE.search(line): print(line, end="", flush=True)
    rc = p.wait()
    if lf: lf.close()
    if rc not in ok:
        raise RuntimeError(f"command failed with exit code {rc}")
    return rc

def save_results(step):
    # zip /content/runs to /content/results_<step>.zip and offer it for download
    z = f"{WORK}/results_{step}.zip"
    sh(f'cd {WORK} && rm -f "{z}" && zip -q -r "{z}" runs && ls -la "{z}"')
    try:
        from google.colab import files
        files.download(z)
    except Exception:
        print(f"download {z} from the Files pane")
""")
code(r"""
# unpack code (fresh copy each session; results live on Drive, not here)
sh(f'rm -rf {CODE} && mkdir -p {CODE} && unzip -q -o "{CODE_ZIP}" -d {CODE} && ls {CODE}')
os.chdir(CODE)
""")
code(r"""
# install (skipped if this session already has it)
import importlib.util
if importlib.util.find_spec("vllm") is None:
    sh(f'pip install -q "vllm=={VLLM_VERSION}" "math-verify[antlr4_13_2]>=0.9.0" datasets matplotlib pytest')
import vllm, torch, transformers
print("vllm", vllm.__version__, "| torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__)
""")
code(r"""
# unit tests (MLX tests are skipped on Linux; the torch entropy and steering tests run here)
sh("python -m pytest -q tests 2>&1 | tail -5")
""")
md(r"""
## Step A — validation of the vLLM backend

Same model, prompts and design as the Mac pilot B (Qwen2.5-1.5B-Instruct, the same 50 selected prompts,
T = 0.7 with no top-p/top-k/min-p, budget 2048, k0 = 48, M = 4, k = 8, checkpoints every 128 tokens).
It passes only if the conservation ratio's 95% CI contains 1. The Mac and GPU results are then compared.
""")
code(r"""
if RUN_A_VALIDATION:
    OUT = f"{RUNS}/gpuA_q15b_instruct"
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(f"{OUT}/selected_prompts.jsonl"):
        sh(f'cp data/pilotB_selected_prompts.jsonl "{OUT}/selected_prompts.jsonl"')
    sh(f'python -u scripts/run_stage1.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct '
       f'--prompts data/math500_pool150_seed20261103.jsonl --out "{OUT}" '
       f'--temperature 0.7 --top-p 1.0 --top-k 0 --min-p 0.0 --max-new-tokens 2048 --max-model-len 4096 '
       f'--interval 128 --k0 48 --m-base 4 --k 8 --n-select 8 --max-prompts 50 --chunk-prompts 25 --seed 0',
       log=f"{OUT}/log.txt")
""")
code(r"""
if RUN_A_VALIDATION:
    sh(f'python scripts/add_entropy.py --backend torch --device cuda --dtype bfloat16 --model Qwen/Qwen2.5-1.5B-Instruct '
       f'--temperature 0.7 --in "{OUT}/trajectories.jsonl" --out "{OUT}/trajectories_entropy.jsonl"', log=f"{OUT}/log.txt")
    sh(f'python scripts/analyze_stage1.py --trajectories "{OUT}/trajectories_entropy.jsonl" --out "{OUT}/analysis" --abs-bin-width 128')
    save_results("A1_stage1")
""")
code(r"""
if RUN_A_VALIDATION:
    sh(f'python -u scripts/run_steering.py --backend torch --model Qwen/Qwen2.5-1.5B-Instruct --temperature 0.7 '
       f'--trajectories "{OUT}/trajectories.jsonl" --out "{OUT}/steering.jsonl" --stride 16 --swaps 4 --window 64',
       log=f"{OUT}/log.txt")
    sh(f'python scripts/analyze_steering.py --steering "{OUT}/steering.jsonl" --out "{OUT}/steering_analysis"')
    save_results("A2_steering")
""")
md(r"""
## Step B — R1-Distill length probe

Long-reasoning models write thousands of tokens and Stage 1 cost grows with the square of the length,
so the pilot is sized from measured lengths. This prints the cost terms used in methods §6.
""")
code(r"""
if RUN_B_R1_PROBE:
    PROBE = f"{RUNS}/gpuB_r1_probe.json"
    sh(f'python scripts/probe_lengths.py --backend vllm --model {R1_MODEL} --prompts data/math500_pool150_seed20261103.jsonl '
       f'--n-prompts 20 --n 4 --temperature {R1_TEMPERATURE} --top-p {R1_TOP_P} '
       f'--max-new-tokens {R1_BUDGET} --max-model-len {R1_BUDGET + 1024} --out "{PROBE}" | tail -25')
    import json
    r = json.load(open(PROBE))
    tps = r["gen_tok_per_s"]
    for P, M, k, L, k0 in [(30, 4, 8, 512, 32), (50, 4, 8, 512, 32)]:
        cont = P * M * k * r["EN2_sqrt"] ** 2 / (2 * L)
        base = P * (k0 + M) * r["N_mean"]
        sel = 3 * P * 8 * r["N_mean"]  # selection pool ~3x the kept prompts
        tot = cont + base + sel
        print(f"P={P} M={M} k={k} L={L}: ~{tot/1e6:.1f}M tokens -> ~{tot/tps/3600:.1f} h at the probe's {tps:.0f} tok/s "
              "(the probe has low concurrency, so this is pessimistic)")
    save_results("B_probe")
""")
md(r"""
## Step E — base model (Qwen2.5-1.5B, raw prompt, no chat template)

Same design as step A, but prompt selection is redone (the base model's pass rates differ).
""")
code(r"""
if RUN_E_BASE:
    OUT = f"{RUNS}/gpuE_q15b_base"
    os.makedirs(OUT, exist_ok=True)
    sh(f'python -u scripts/run_stage1.py --backend vllm --model Qwen/Qwen2.5-1.5B --no-chat-template '
       f'--prompts data/math500_pool150_seed20261103.jsonl --out "{OUT}" '
       f'--temperature 0.7 --top-p 1.0 --top-k 0 --min-p 0.0 --max-new-tokens 2048 --max-model-len 4096 '
       f'--interval 128 --k0 48 --m-base 4 --k 8 --n-select 8 --max-prompts 50 --chunk-prompts 25 --seed 0',
       log=f"{OUT}/log.txt")
    sh(f'python scripts/add_entropy.py --backend torch --device cuda --dtype bfloat16 --model Qwen/Qwen2.5-1.5B '
       f'--temperature 0.7 --in "{OUT}/trajectories.jsonl" --out "{OUT}/trajectories_entropy.jsonl"', log=f"{OUT}/log.txt")
    sh(f'python scripts/analyze_stage1.py --trajectories "{OUT}/trajectories_entropy.jsonl" --out "{OUT}/analysis" --abs-bin-width 128')
    save_results("E1_base_stage1")
    sh(f'python -u scripts/run_steering.py --backend torch --model Qwen/Qwen2.5-1.5B --temperature 0.7 '
       f'--trajectories "{OUT}/trajectories.jsonl" --out "{OUT}/steering.jsonl" --stride 16 --swaps 4 --window 64',
       log=f"{OUT}/log.txt")
    sh(f'python scripts/analyze_steering.py --steering "{OUT}/steering.jsonl" --out "{OUT}/steering_analysis"')
    save_results("E2_base_steering")
""")
md(r"""
## Step C — R1-Distill Stage 1 pilot

30 prompts selected from the 150-prompt pool; k0 = 32, M = 4, k = 8, checkpoints every 512 tokens,
budget 16,384. Measured on 2026-11-04: selection ~5.5M tokens, then ~16M tokens and ~2 h per
10-prompt chunk (~2k tok/s at these lengths). **A results zip is offered after every chunk:
download each one.** To resume after a disconnect, upload the code zip and the latest
`results_C1_*.zip` (or the zip Claude prepares from downloaded files), then run all cells.
Steering for R1 uses stride 128 (not 16) because responses average ~7,600 tokens.
""")
code(r"""
if RUN_C_R1_PILOT:
    OUT = f"{RUNS}/gpuC_r1d15b"
    os.makedirs(OUT, exist_ok=True)
    # one chunk (10 prompts, ~1.5-2.5 h) per call, then a downloadable zip: a disconnect loses at most one chunk
    import json
    while True:
        rc = sh(f'python -u scripts/run_stage1.py --backend vllm --model {R1_MODEL} '
                f'--prompts data/math500_pool150_seed20261103.jsonl --out "{OUT}" '
                f'--temperature {R1_TEMPERATURE} --top-p {R1_TOP_P} --top-k 0 --min-p 0.0 '
                f'--max-new-tokens {R1_BUDGET} --max-model-len {R1_BUDGET + 1024} '
                f'--interval 512 --k0 32 --m-base 4 --k 8 --n-select 8 --max-prompts 30 --chunk-prompts 10 --seed 0 '
                f'--max-chunks 1', log=f"{OUT}/log.txt", ok=(0, 3))
        done = len({json.loads(l)["prompt_id"] for l in open(f"{OUT}/trajectories.jsonl")})
        save_results(f"C1_r1_{done:02d}of30")
        if rc == 0:
            break
    sh(f'python scripts/add_entropy.py --backend torch --device cuda --dtype bfloat16 --model {R1_MODEL} '
       f'--temperature {R1_TEMPERATURE} --in "{OUT}/trajectories.jsonl" --out "{OUT}/trajectories_entropy.jsonl"', log=f"{OUT}/log.txt")
    sh(f'python scripts/analyze_stage1.py --trajectories "{OUT}/trajectories_entropy.jsonl" --out "{OUT}/analysis" --abs-bin-width 512')
    save_results("C2_r1_analysis")
    sh(f'python -u scripts/run_steering.py --backend torch --model {R1_MODEL} --temperature {R1_TEMPERATURE} '
       f'--trajectories "{OUT}/trajectories.jsonl" --out "{OUT}/steering.jsonl" --stride 128 --swaps 4 --window 64',
       log=f"{OUT}/log.txt")
    sh(f'python scripts/analyze_steering.py --steering "{OUT}/steering.jsonl" --out "{OUT}/steering_analysis"')
    save_results("C3_r1_steering")
""")
md(r"""
## Step D — final results

Download `results_final.zip` and send it to Claude (or put it in the project folder on the Mac).
""")
code(r"""
save_results("final")
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("cells:", len(cells))
