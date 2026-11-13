# Cheatsheet 04 — Compute, Memory and Environment

> **English translation.** AI-assisted translation of the [Korean original](04_compute_and_memory.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original. In the code blocks only the comments and message strings were translated; the executable code is unchanged.

This collects, separately, what was **measured or confirmed** in module 06 (notebook 03) and SETUP.md and what was estimated with formulas (approximations) without measuring. Notation: **measured** = a value from a run in this repository, **formula** = a calculation that is a rough estimate without activations, etc., † = a statement that could not be confirmed. The `python` blocks were run and confirmed in the repository `.venv` (2026-11-06).

## 1. Memory budget (without activations, formula)
| Situation | Bytes/parameter | Explanation |
|---|---|---|
| Inference, fp32 | 4 | Weights |
| Inference, bf16/fp16 | 2 | Weights |
| Full fine-tuning, pure fp32 AdamW | 16 | Weights 4 + gradients 4 + Adam 1st·2nd moments 8 |
| Mixed precision full fine-tuning | 16–20† | An fp32 master copy is needed, so the total bytes can be similar to or larger than pure fp32 (depending on how the half-precision copy and gradients are kept)†. The gain is mainly in compute speed and activation memory |
| LoRA (frozen weights fp32) | frozen 4 + trained part 16 | If the trainable parameters are under 1% of the total, it is nearly 4P |
| LoRA (frozen weights bf16) | frozen 2 + trained part 16 | Only the frozen weights are bf16, and the LoRA parameters are fp32 |

```python
def budget_gb(n_params, mode="full_fp32", n_trainable=None):
    """A memory estimate without activations (GB, 1 GB = 1e9 bytes). A formula, not a measurement."""
    if mode == "infer_fp32": b = 4 * n_params
    elif mode == "infer_bf16": b = 2 * n_params
    elif mode == "full_fp32": b = 16 * n_params
    elif mode in ("lora_fp32", "lora_bf16"):
        frozen = (4 if mode == "lora_fp32" else 2) * (n_params - n_trainable)
        b = frozen + 16 * n_trainable
    else: raise ValueError(mode)
    return b / 1e9

assert abs(budget_gb(650e6, "full_fp32") - 10.4) < 0.01                   # module 06 §5: full fine-tuning of 650M ≈ 10.4 GB
assert abs(budget_gb(650e6, "lora_fp32", 1.35e6) - 2.6) < 0.1             # LoRA (q,v r=8) ≈ 2.6 GB
assert abs(budget_gb(650e6, "lora_bf16", 1.35e6) - 1.3) < 0.1             # frozen weights in bf16 ≈ 1.3 GB
print({m: round(budget_gb(650e6, m, 1.35e6), 2) for m in ["infer_fp32", "infer_bf16", "full_fp32", "lora_fp32", "lora_bf16"]})
```
- **Activations** vary greatly with batch, sequence length and the number of layers, so they are not in the formula above. They also vary with the settings even for small models and short sequences, so **measure them in your own setting**, and they can dominate for long sequences (§3 below).
- At 8M and 35M, the difference between methods is 0.1–0.5 GB and is buried in the differences between equipment (module 06).

## 2. Estimating compute
- Training compute of a transformer $C \approx 6ND$ (N parameters, D tokens)†, forward pass $\approx 2ND$†. Module 06 estimated the time from this formula and the throughput measured on the Mac MPS.
- Estimate time by dividing the compute by **the throughput measured on the same equipment and the same implementation** (FLOP/s or token/s), matching the units. Re-measure when moving to other equipment. In module 04 we measured that one cannot assume the Colab T4 is faster than the Mac MPS. Geneformer V2-104M inference (fp32) took 788 seconds on the T4 and 673 seconds on the Mac, and scVI, which trains on the CPU, took 757 seconds on Colab's 2 cores and 134 seconds on the Mac. Conversely, notebook 01 of module 06 (LoRA and full fine-tuning, mostly GPU) was faster on the T4, at 721 seconds, than the Mac MPS's about 1,156 seconds, and notebook 01 of module 05 (sklearn cross-validation, mostly CPU) was much slower on Colab (2 vCPUs), at about 60 minutes, than the Mac's about 10 minutes. These are single runs and not repeated measurements.

## 3. Measured values (Mac MPS, ESM-2 35M, batch 4, median of 3 runs of forward+backward)
The memory is the MPS **driver allocation**, so it is a *proxy* and not a peak. The time of the same setting fluctuated by about 10% from run to run.

| Sequence length | eager fp32 | sdpa fp32 | sdpa bf16 autocast | sdpa fp32 + checkpointing |
|---|---|---|---|---|
| 128 | 0.11 s · 0.63 GB | 0.11 s · 0.63 GB | 0.12 s · 0.68 GB | 0.13 s · 0.41 GB |
| 512 | 0.41 s · 3.50 GB | 0.41 s · 3.50 GB | 0.47 s · 2.45 GB | 0.51 s · 1.35 GB |
| 1024 | 1.00 s · 7.84 GB | 0.98 s · 7.85 GB | 1.11 s · 7.90 GB | 1.28 s · 2.48 GB |

(The same as the table in module 06 README §5-C.) How to read it:
- When the sequence length went from 128 to 1024, the memory grew about 12 times. Activations dominate. At length 1024, checkpointing changed the memory by −68% and the time by +31%.
- On this Mac, the proxy memory of `sdpa` and `eager` was the same. **MPS has no FlashAttention kernel**†. It was different on the Colab T4 (CUDA). With the same setting (35M, batch 4, length 1024), the maximum allocated memory was 7.34 GB for eager fp32, 2.58 GB for sdpa fp32 and 0.61 GB with checkpointing applied as well (it is `max_memory_allocated`, so it is a different metric from the driver proxy of MPS. Module 06 README §8).
- bf16 autocast gave no time gain on this equipment (+9–20%).

```python
import gc
import numpy as np, torch

def mps_mem_gb():
    """MPS driver allocation (GB). Not a peak, but a proxy that includes the cache."""
    if not torch.backends.mps.is_available():
        return float("nan")
    return torch.mps.driver_allocated_memory() / 1e9

gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()             # if you do not clean up before measuring, the buffers of the previous work get mixed in (experienced in module 06)
before = mps_mem_gb()
x = torch.zeros(256, 1024, 1024, device="mps" if torch.backends.mps.is_available() else "cpu")   # 1 GiB (=1.07 GB in decimal GB) fp32
after = mps_mem_gb()
print("change of the MPS proxy (GB):", None if np.isnan(after) else round(after - before, 2))
del x; gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

## 4. What works and what does not per environment (within the range confirmed in this repository)
| Environment | What was confirmed | What was not confirmed or does not work |
|---|---|---|
| 🟢 Mac (MPS/CPU) | ESM-2 8M–650M embeddings and zero-shot, NT 500M embeddings, Geneformer, small LoRA/full fine-tuning, SAE, DDP equivalence (CPU, 2 processes) | FlashAttention (not on MPS†), large models (Evo 2 7B and above), CUDA-only packages |
| 🟡 Colab T4 | Ran notebooks 01–03 of module 04 without errors (about 56 minutes in total, about 1 compute unit). Ran notebook 04 of module 02 (ESMFold v1) without errors (about 1.5–3 minutes). Modules 05 (01·02) and 06 (01–03) and the module 07 template also ran without errors (1 run each; 05/01 about 60 minutes, 05/02 about 25 minutes, 06/01 721 seconds, 06/02 792 seconds, 06/03 131 seconds, 07 93 seconds) | No guarantee that the T4 is faster than the Mac (§2 above). Some numbers of 05 and 06 (position-block fine-tuning, PCA, individual SAE features) differed from the Mac (module 06 README §8). Stochastic training such as scVI can differ by about ±0.01 in the numbers. The free-tier runtime has about 12 GB of RAM and 15.6 GB of GPU, so the whole ESMFold fp32 (about 14 GB) must be put straight onto the GPU without going through the CPU (§5) |
| 🟡 Colab A100 | Full fine-tuning of Geneformer V2-104M (512 tokens, batch 32, fp32): 0.580 seconds/step, maximum 9.2 GB, about 9,600 cells, 2 epochs, 351 seconds per fold (about 8 seconds/step on Mac MPS). 20 folds took 7,087 seconds and 13.26 compute units (6.73 CU/h). It is a **measurement**, run 1 time (module 07 [C③-V2](../../07_capstone/examples/dry_runs/second_halves.en.md)) | fp16 autocast was 0.116 seconds/step in the benchmark, but the equivalence of the results was not confirmed. The Drive mount of the CLI failed (the results were downloaded periodically) |
| 🔴 SLURM | Only a syntax check of `slurm_ddp.sbatch` with `bash -n` | A real cluster run, NCCL and multiple nodes **could not be run** |
| Models with large weights | Evo 2: `evo2_1b_base` and 40B need FP8 (Transformer Engine) and a Hopper GPU, and `evo2_7b` can use bf16 (official README, checked 2026-11). ESMFold2 weights are about 27 GB (model card) | This repository did not directly run either |

## 5. Equipment quirks (what we ran into)
| Symptom | Cause/fix |
|---|---|
| `torchrun --standalone` times out after 300 seconds (macOS) | The host name resolves to `1.0.0.127.in-addr.arpa`. Give `--local-addr=127.0.0.1` |
| The memory measurement is affected by the previous run | Measure after `gc.collect()` + `torch.mps.empty_cache()` |
| The time measurement fluctuates | Run alone without overlap with other GPU/CPU jobs, take the median of 3 or more runs |
| DDP finishes in 2 steps on short data | Use an epoch loop and `sampler.set_epoch(epoch)` |
| The results differ because the package versions differ | Reproduce with `requirements/base.lock.txt`, and use revision pinning for the model |
| Large files get into git | `.cache/` (weights·data) is in `.gitignore`. Check with `git status` before committing |
| On Colab, importing `peft` fails with `Found an incompatible version of torchao` | The image's `torchao 0.10.0` conflicts with `peft 0.21.2` (which requires torchao 0.16 or later). The notebooks do not use it, so run `pip uninstall -y torchao` and restart the kernel |
| `torchrun --nproc_per_node=2` DDP fails on 1 GPU | Rank 1 cannot open `cuda:1`. To check on CPU (gloo), set `CUDA_VISIBLE_DEVICES=""` for the child processes (module 06 notebook 03) |
| On Colab, the kernel suddenly dies while loading the model (ESMFold, RAM 12 GB) | The weights are first loaded in fp32 onto the CPU RAM, requiring about 14 GB, and a cgroup OOM occurs. It failed in 2 of 3 runs. With `from_pretrained(..., device_map={"": 0})` they go straight onto the GPU, the CPU memory is under 1 GB (excluding the file cache) and the GPU instantaneous peak is about 14.1 GB |
| The kernel died during `colab exec` but the CLI did not finish | `--timeout` did not apply to the disconnection (waited 4.6 hours, about 5 CU consumed). Wrap it with the shell's `timeout 600 colab exec ...` and run `colab stop` when it ends |

## 6. Checks before running
- [ ] You wrote down the needed downloads (file, source, size) and got approval
- [ ] You estimated with the memory formula, and separately checked the activations (sequence length, batch)
- [ ] When reporting time and memory, you wrote that they are *proxies* and wrote the equipment
- [ ] You wrote the environments you could not run (cluster, other GPUs) as "unverified"

<!-- i18n-source: resources/cheatsheets/04_compute_and_memory.md blob:3328106581a7b653e803c2a81323e8757bbbb00c -->
