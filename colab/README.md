# Running on a Colab A100 (no Google Drive)

1. Open `stage1_a100.ipynb` in Colab (File → Upload notebook, or from VS Code with the Colab
   kernel). Choose an A100 runtime.
2. Run the first cells: they ask you to upload `resolution-curve-code.zip` (rebuild it with
   `colab/build_zip.sh` after any code change). Or drag it into the Files pane at `/content/`.
3. Steps A (vLLM validation) and B (R1-Distill length probe) are on; step C (R1-Distill pilot)
   is off until its size is agreed from step B's output.
4. After each step the notebook writes `/content/results_<step>.zip` and offers it for download.
   Download each one and send it to Claude. Colab's disk is wiped on disconnect; to resume,
   upload the code zip plus the latest results zip and run again from the top.

`make_notebook.py` regenerates the notebook.
