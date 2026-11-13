# Environment Setup

> **English translation.** AI-assisted translation of the [Korean original](SETUP.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

Compute tier: 🟢 Mac (CPU/MPS) · 🟡 Colab (GPU) · 🔴 institutional GPU cluster

## 1. Local Mac (🟢)
```bash
# Check whether uv is installed (if not, see https://docs.astral.sh/uv/)
uv --version

uv venv --python 3.12 .venv      # base.txt install and module 01 notebooks verified on 3.12 (system Python 3.15 not tried)
source .venv/bin/activate
uv pip install -r requirements/base.txt
python -m ipykernel install --user --name bio-fm
```

Check whether MPS is available:
```python
import torch
print(torch.backends.mps.is_available())
```
- If some operations are unsupported on MPS, set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to allow a CPU fallback.
- 🟢 notebooks are designed to use only small checkpoints (e.g., ESM-2 at the 8M–35M scale).

## 2. Google Colab (🟡)
In the first cell of the notebook:
```python
!pip install -q -r https://raw.githubusercontent.com/<your-account>/<repo>/main/requirements/colab.txt
```
Or clone the repository and install `requirements/colab.txt`. Set the runtime type to GPU (T4 or better).

**What was confirmed on a Colab T4** (2026-11-06, GPU 15.6 GB, 2 vCPUs, about 12 GB RAM, Python 3.13, torch 2.14.1, transformers 5.22.1):
- The image does not include `Bio`, `rdkit` or `peft`. Install them with `pip install biopython==1.88 rdkit==2026.9.1 peft==0.21.2`.
- The image's `torchao 0.10.0` conflicts with `peft 0.21.2` (`ImportError: ... torchao`). Run `pip uninstall -y torchao`, then restart the kernel.
- There is no `.cache/`, so you must first download or upload the caches that the module 05, 06 and 07 notebooks assume (ProteinGym v1.3, the UniProt table of module 05, and the ESM-2 150M and 650M `config.json` files of module 06/03). Module 02 notebook 01 downloads ProteinGym, and module 05 notebook 02 downloads the UniProt table.
- Memory: if ESMFold v1 loads its weights onto the CPU first, it can exceed the 12 GB of RAM and kill the kernel (module 02 notebook 04 puts them straight onto the GPU with `device_map`).
- Colab run times per notebook and the numeric differences from the Mac are in each module README (05 §9, 06 §8, 07 §5).

## 3. Institutional cluster (🔴)
Assumption: a SLURM scheduler and the `module` system are used. Partition and module names differ between institutions, so check and edit them.
```bash
#!/bin/bash
#SBATCH --job-name=bio-fm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

source .venv/bin/activate
python train.py
```
Install CUDA-dependent packages such as `flash-attn` individually, to match the cluster's CUDA and compiler versions.

## 4. Reproducibility
- `requirements/base.lock.txt`: the `uv pip freeze` output of the environment (Python 3.12.13, macOS arm64) in which the module 01–06 notebooks were confirmed to run. Module 04 added 22 packages, such as `harmonypy` and `scvi-tools`, module 05 added `rdkit`, and module 06 added `peft`; the versions of existing packages did not change. To reproduce it exactly, run `uv pip install -r requirements/base.lock.txt`.
- If you add packages in later modules, update the lock file after confirming that the notebooks run.
- Record the ID and the verification date of each model/dataset at the top of the notebook.

## 5. Isolated environment for module 03 notebook 03 (Enformer)
`enformer-pytorch` pins `transformers==4.56.2`, so installing it into the default `.venv` (transformers 5.x) can break other notebooks. Create a separate environment.
```bash
uv venv --python 3.12 .venv-enformer
uv pip install --python .venv-enformer/bin/python -r requirements/enformer.txt
.venv-enformer/bin/jupyter lab   # run 03_enformer_tracks.ipynb with this environment's kernel
```
- `.venv-enformer/` is excluded from git. The verified environment is Python 3.12, torch 2.15.0 and transformers 4.56.2.
- Module 03 notebooks 01 and 02 run in the default `.venv`.
- Among the models and data downloaded in module 03, the **Nucleotide Transformer v1 weights are CC BY-NC-SA 4.0 (non-commercial)**.

## 6. Viewing it as a web book (optional)
This turns the Markdown and notebooks, without changing their content, into a web book (Jupyter Book 2 / MyST) with a table of contents and search. The originals are not touched; they are copied to `.book_build/` (excluded from git) and built there.
```bash
uv venv --python 3.12 .venv-book
uv pip install --python .venv-book/bin/python jupyter-book   # verified: jupyter-book 2.1.7, Node required
bash book/build.sh             # build the documents only (notebooks are included without output)
EXEC=1 bash book/build.sh      # run the 17 notebooks that can run in the Mac .venv and include their output and figures (only changed ones are re-run)
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory .book_build/_build/html   # http://localhost:8765
```
- The first build downloads the site theme (`book-theme`) from GitHub (a network connection is required).
- When you add a document, add it to the `toc` in `book/myst.yml` as well. A document that is not in the toc does not appear in the book.
- The notebooks that need Colab or an isolated environment (02/04 ESMFold, 03/03 Enformer) are not executed and are included without output.
- `book/stage_fix_links.py` fixes the folder links and the Korean heading anchors in the copy so that they work in the book (MyST drops Hangul from heading ids).

## 7. Publishing to GitHub Pages (optional)
Use this to publish the web book as a project page (`https://<account>.github.io/<repository name>/`). The notebooks take a long time to execute, so the book is not built with GitHub Actions. Build the HTML locally and push it to the `gh-pages` branch.
```bash
bash book/pages.sh <repository name>
```
- It builds for a subpath (`BASE_URL=/<repository name>`) and inspects the output. It stops if any local path, account name or secret pattern appears, or if the inspection itself ends in an error. It also stops when the page count differs from the toc, when the executed copy of a notebook whose code changed is out of date, and when the git email is not a noreply address. Re-run an outdated executed copy with `EXEC=1 bash book/build.sh`.
- If everything passes, it collects the output into `.pages_site/` (excluded from git) together with `.nojekyll` and the license file, and commits it to the gh-pages branch. The `robots.txt` and the sitemap, which are generated with a placeholder domain (`http://pre.render`), are left out.
- The script does not push. After you create the repository, run the `git -C .pages_site push --force …` command printed at the end yourself, and in the repository's Settings > Pages choose the `gh-pages` branch as the Source.
- Local absolute paths left in notebook outputs are replaced with `<repo>` and `~` by `book/redact_paths.py`. This step is part of `book/build.sh`, so it also applies to an ordinary build.
- To view the book locally, rebuild with the command in §6, without `BASE_URL`. A build for a subpath does not open correctly at the root address (`http://localhost:8765`).
- After you change a document, run `bash book/pages.sh <repository name>` again and push with the same command. Because of `--force`, the earlier history of gh-pages is not kept.

<!-- i18n-source: SETUP.md blob:55be4a71460ae19f5502dd86f2c81ac83837293b -->
