#!/bin/bash
# SLURM template for multi-GPU fine-tuning with torchrun (module 06).
#
# STATUS: NOT RUN on a cluster. This repository has no cluster access; the script was only
# syntax-checked (`bash -n`). Partition/account/module names are placeholders: ask your cluster's docs.
# The Python side (train_ddp.py) was smoke-tested with 2 CPU processes (notebook 03).
#
# Submit:  sbatch slurm_ddp.sbatch
#SBATCH --job-name=esm2-lora-ddp
#SBATCH --partition=gpu                # PLACEHOLDER
#SBATCH --account=YOUR_ACCOUNT         # PLACEHOLDER
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # one torchrun launcher per node; it spawns one process per GPU
#SBATCH --gpus-per-node=4              # PLACEHOLDER: match --nproc_per_node below
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# --- environment (site-specific; PLACEHOLDERS) ---
# module load cuda/12.x
# source /path/to/venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}          # keep model caches off $HOME quotas
export HF_HUB_OFFLINE=1                               # compute nodes often have no internet: pre-download on a login node
export TOKENIZERS_PARALLELISM=false

# Rendezvous: the first node of the allocation
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR MASTER_PORT

# Pin the model revision (supply-chain hygiene; see module 05 section 8)
REVISION="REPLACE_WITH_COMMIT_SHA"

# train_ddp.py is looked up relative to the submit directory
cd "${SLURM_SUBMIT_DIR:-.}"

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train_ddp.py \
  --model facebook/esm2_t6_8M_UR50D --revision "$REVISION" \
  --steps 100 --per-rank-batch 8 --out "results-$SLURM_JOB_ID.json"

# Notes
# * Effective batch = per-rank batch x number of processes. Scale the learning rate deliberately and report it.
# * For multi-node jobs raise --nnodes/--nodes; the c10d rendezvous above is the usual pattern, but check your site.
# * Checkpoint on rank 0 only and write to $SCRATCH; make the job resumable (requeue) if the partition preempts.
