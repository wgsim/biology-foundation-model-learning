# 06. Advanced Research — Adaptation, Interpretation, Efficiency, Benchmarks, Accountability

> **English translation.** AI-assisted translation of the [Korean original](README.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

**Duration**: Weeks 15–16 (2 weeks, about 16–20 hours) · **Compute tier**: 🟢 3 local Mac (MPS/CPU) notebooks · 🔴 1 SLURM template (**could not be run** on a cluster)
**Prerequisites**: [module 01](../01_foundations/) (scaling, pitfalls of evaluation), [module 02](../02_protein/) (ProteinGym, probe), [module 04](../04_single_cell/)·[05](../05_multimodal_other/) (simple baselines, random controls, paired comparisons, provenance checks).

> Notation: † marks a statement that must be checked directly in the paper text or the documents. In this repository, only the abstract, model-card and repository-metadata level was checked. The cited papers are in section 06 of the [reading_list](../resources/reading_list.en.md). This is information as of 2026-11.

---

## Learning goals
1. Plan small-scale pretraining **backward, from the compute budget and the evaluation design**, and explain the paths by which downstream can get worse even as the loss improves.
2. Compare frozen probes, head training, LoRA and full fine-tuning on **the same split and the same pre-fixed settings**, and confirm that the conclusion can differ by split and assay type.
3. Distinguish by measurement what LoRA reduces and what it reduces less. The trainable parameters and the gradient/optimizer memory shrink, but it **saves less computation**. This is because the gradient computation of the frozen weights is skipped but backpropagation through the activations remains.
4. Implement the structure and evaluation metrics of a sparse autoencoder (SAE), and check feature–concept correspondence against a **control that removes the selection effect** (random dictionary) and on **held-out proteins**. Distinguish correlation from causation.
5. Measure how the attention implementation, precision and checkpointing affect time and memory. Also verify the **meaning of the behavior** — what data splitting and gradient averaging mean in distributed training. Separate out what was not measured, such as the cluster, NCCL and FlashAttention.
6. Design benchmarks, **check the licenses of data and models with grounds**, and build the habit of reviewing dual-use risk.

## Two-week schedule
| Week | Topic | Notebook/template | Reading (mainly ★) |
|---|---|---|---|
| 15 | Comparing adaptation methods (LoRA, etc.), interpretation (SAE) | [01](notebooks/01_lora_vs_frozen_vs_full.ipynb) · [02](notebooks/02_sae_esm2_features.ipynb) | LoRA, Biderman 2024, InterPLM, Cunningham 2023, Gao 2024 |
| 16 | Efficiency and distribution, benchmark design, license and accountability | [03](notebooks/03_efficiency_and_ddp_check.ipynb) · [templates/](templates/) | FlashAttention, Micikevicius 2017, Baker & Church 2024, Sandbrink 2023 |

**The module deliverable** is one of two. One is a 1-page report **comparing frozen representations with fine-tuning**, which extends the comparison of notebook 01 to another assay of your choice and applies validation-based selection (exercise 2). The other is an intervention experiment on the SAE features of notebook 02 (exercise 4). Write the design **before seeing the results**.

---

## 1. Designing small-scale pretraining: backward from the evaluation
Large-scale pretraining is outside the scope of this repository. But **how to plan** small-scale pretraining, like the small MLM of module 01 notebook 02, is knowledge needed to read every pretraining paper.

1. **Decide the evaluation and the split first.** Decide the downstream task and the held-out split, and only then build the pretraining data. The homology-cluster-level split seen in module 01 is an example. If you reverse the order, the evaluation sequences get mixed into the pretraining. It is a problem seen repeatedly in modules 04 and 05.
2. **Reduce data redundancy.** Sequences in public DBs over-represent close relatives. If you sample in a balanced way from the clusters made with an identity threshold, you can see more diverse sequences with the same compute†. Which threshold favors downstream differs by task, so decide it by experiment.
3. **Calculate the compute budget.** The training compute of a transformer is approximated as roughly $C\approx 6\,N D$ for N parameters and D tokens†. Let us estimate the effective throughput from this repository's Mac MPS measurement (notebook 03 §1, ESM-2 35M, batch 4, forward+backward). $6\cdot 3.33{\times}10^7\cdot(4L)/t$ is about 0.9–1.0 TFLOP/s at lengths 128–512 and about 0.75 TFLOP/s at length 1024. Training a 35M model on 1 billion tokens needs $6\cdot3.33{\times}10^7\cdot10^9\approx2{\times}10^{17}$ FLOP, so at this throughput it takes about 55–74 hours. This is a **rough estimate** and varies greatly with batch, length, precision and optimizer-step time. It does not apply to CUDA GPUs.
4. **Decide the model size and the number of tokens together.** Kaplan et al. (2020) report that the loss follows a power law in size, data and compute, and Hoffmann et al. (2022) report that, for a given compute, size and number of tokens should grow together. Both are results for **text language models**, and whether the same exponents hold for proteins could not be confirmed in this repository†.
5. **Do not look at the loss alone.** Along with the cluster-level validation loss, periodically measure downstream performance such as a probe or zero-shot. Even as the loss goes down, downstream may not go up, or the model may just be memorizing homologs (module 01 notebook 03).
6. **Secure reproducibility.** Record the tokenizer, masking ratio, seeds and data filters, and justify the design with ablations that change one thing at a time at a small scale.

## 2. A map of adaptation methods
| Method | What is trained | Memory (including optimizer) | Risk |
|---|---|---|---|
| Frozen + linear probe | head | Very small | A limit if the representation does not hold task information linearly |
| Adapters / **LoRA** | A low-rank matrix $BA$ ($W+\frac{\alpha}{r}BA$) | Frozen weights + a small training state | Sensitive to rank, target layers and LR; the time saving is limited (11–12% at 8M, §5) |
| Full fine-tuning | All the weights | weights + gradients + Adam state ≈ 16P bytes (fp32) | Overfitting, forgetting, memory |
| Prompt/instruction | The input | No change to the model | Rare for protein models |

- **The LoRA paper** (Hu et al. 2021) presents in the abstract a method that freezes the pretrained weights and injects trainable low-rank decomposition matrices into each layer. It reports that on GPT-3 175B the trainable parameters were reduced to 1/10,000 and the GPU memory to 1/3, while the quality was comparable or better.
- **Biderman et al. (2024)** report in the abstract that, for LLMs in the programming and mathematics domains, LoRA with a standard low-rank setting falls well short of full fine-tuning. In return, they say it better preserves performance outside the target domain. It is a result in an **LLM and large-data setting**, so do not transfer it as it is to small protein data.
- Sledzieski et al. (PNAS 2024), which deals with **protein PEFT**, was checked only up to the title level.

## 3. Interpretation: sparse autoencoders (SAE)
- **Polysemanticity and superposition** are assumptions that explain the phenomenon in which an individual neuron responds to several concepts. The network is thought to represent more features than the number of neurons by superposing them on an overcomplete set of directions (Cunningham et al. 2023 abstract). An SAE is an attempt to find those directions by reconstructing the hidden state **sparsely** from a wider dictionary.
- **TopK SAE** (Gao et al. 2024) reports in the abstract that directly controlling sparsity with a k-sparse autoencoder makes tuning easier and reduces dead latents. It also proposes scaling laws for dictionary size and sparsity, and metrics for feature quality.
- **InterPLM** (Simon & Zou, Nat Methods 2025) was checked only up to the title level. It is a study that applies an SAE to ESM-2 to find interpretable features. Notebook 02 of this module is **an imitation at a much smaller scale**, not a reproduction.
- There are four **common pitfalls in evaluation**. ① If you pick the feature that best matches a concept from among thousands, it looks good even by chance. So a **random-dictionary control** and **held-out proteins** are needed (notebook 02). ② A correlation between a feature and a concept is not evidence that the model *uses* that concept. It must be confirmed by an intervention experiment (exercise 4). ③ You should also look at how much the dictionary directions correspond when the seed is changed, that is, the stability. ④ When the dictionary size is increased, features can split†.

## 4. Efficiency and distributed training
- Let us look at **attention memory**. Standard attention makes an $L\times L$ score matrix for length L, so time and memory grow as $L^2$. **FlashAttention** (Dao et al., NeurIPS 2022) is a CUDA kernel that computes exact attention tile by tile, reducing memory reads and writes†. **It is not available on Mac MPS.**
- **Mixed precision** (Micikevicius et al. 2017) is, per the abstract, a method that trains in half precision while keeping a single-precision copy of the weights and scaling the loss. bf16 often does not need loss scaling†. If you do full fine-tuning with mixed precision, a separate fp32 master copy is needed, so the total bytes are around 16P, similar to pure fp32†. The gain comes mainly from compute speed and activation memory, not from optimizer memory†. The `LoRA_bf16동결` (LoRA with low-precision frozen weights) column in §5-C is a configuration that keeps only the *frozen* weights in bf16 and trains the LoRA parameters in fp32. The measured values depend on the equipment (§5).
- **Gradient checkpointing**: some activations are not stored and are recomputed in the backward pass, trading time for memory.
- **Data parallel (DDP)** replicates the model, splits the data by rank, and then **averages the gradients**. For the update with batch b per rank and W processes to be equal to the update of a single process with batch bW, five conditions are needed. The global batch must be the same and must be split **evenly** across ranks, the initial parameters must be the same, and stochastic operations such as dropout must be off or matched. Also, the loss must be a sample mean, and there must be no layers that depend on batch statistics, like BatchNorm. Notebook 03 checked this only in a synthetic setting designed to satisfy these conditions (dropout 0, SGD, even shards).
- **ZeRO and FSDP** (Rajbhandari 2019, Zhao 2023) were checked only up to the title level. They reduce the memory of one device by storing the parameters, gradients and optimizer state split across ranks†. They could not be run in this repository.

## 5. What we saw in this data — summary of the results of notebooks 01–03
All numbers are values from one run on a Mac, so **check the full notebook outputs**. Because there are 3 or fewer seeds, in some places per-seed values were reported instead of intervals.

### 5-A. Frozen probe vs LoRA vs full fine-tuning (notebook 01)
ESM-2 **8M**, ProteinGym single substitutions, test Spearman, **mean of 3 seeds** (per-seed values and standard deviations are in the notebook output). About 800–1,300 training variants, 5 epochs, LR fixed in advance. Columns: `random | position-block` for each assay.

| Method | DN7A (stability) | IF1 (growth) | A0A247D711 (activity) |
|---|---|---|---|
| B1 aa one-hot+BLOSUM | 0.68 \| 0.61 | 0.40 \| 0.22 | 0.50 \| 0.40 |
| B2 B1+position one-hot | 0.81 \| 0.59 | 0.71 \| 0.25 | 0.66 \| 0.40 |
| Frozen probe (Ridge) | 0.82 \| 0.56 | 0.69 \| 0.43 | 0.62 \| 0.28 |
| LoRA (q,v, r=8) | 0.79 \| 0.41 | 0.70 \| 0.22 | 0.61 \| 0.14 |
| Full fine-tuning | 0.82 \| 0.54 | 0.74 \| 0.38 | 0.63 \| 0.24 |

Scale and sensitivity (DN7A, mean of seeds 0–1): **35M** position-block — probe 0.49, LoRA 0.57, full 0.53 (random: 0.85 / 0.85 / 0.87). **8M LoRA sensitivity** (position-block): default (r=8, lr 1e-3) 0.385, r=2 0.501, r=32 0.503, all linear layers r=8 0.492, lr 3e-4 0.432, lr 3e-3 0.294; **full**: default (lr 5e-5) 0.554, lr 1e-5 0.523, lr 2e-4 0.452 (probe of the same seeds 0.56).

**What we learned**
1. **Fine-tuning does not always lead** (§4). In the position-block split, the mean order in all three assays was probe ≥ full > LoRA. DN7A is 0.56/0.54/0.41, IF1 is 0.43/0.38/0.22, A0A is 0.28/0.24/0.14, and the number of seeds in which LoRA was higher than the probe is 0 of 9. In the random split, full fine-tuning was the same as or slightly higher than the probe (in IF1, in all 3 seeds, by about +0.05). This is a result obtained with about 1 thousand training variants, an 8M model and fixed settings.
2. **Simple baselines are strong** (§4). In the random split, B2, which uses the position one-hot, is the same as or higher than fine-tuning and the probe, at 0.81 on DN7A and 0.66 on A0A. In the position-block split, B1 is higher than the ESM-2 probe's 0.56 and 0.28, at 0.61 on DN7A and 0.40 on A0A. Only on IF1 does the probe lead B1 (0.43 vs 0.22), but the seed variation is large. Before saying "ESM is better than the baseline", always include a baseline that uses the position effect (module 02).
3. **The ranking and the size change with the split** (§4). The random scores of the ESM-based methods are 0.61–0.82, much higher than the 0.14–0.56 of position-block. Every method fluctuates a lot in the problem of predicting new positions. The per-seed values of IF1 and A0A are spread from −0.10 to 0.60. With 3 seeds it is hard to settle the ranking between methods.
4. **The LoRA result depends strongly on the setting** (§6). The default setting fixed in advance (r=8, lr 1e-3) was 0.385, the second lowest of the 6 LoRA settings, and the full range was 0.29–0.50. In seeds 0–1, every LoRA setting was lower than the probe's 0.56. However, this is the result of a limited search over settings, and since validation-based selection (exercise 2) was not done, we do not conclude that LoRA is worse. For full fine-tuning, the default lr 5e-5 was the best, and it got worse, 0.452, at lr 2e-4. When it was re-run on a Colab T4, this ranking changed: the default setting was the lowest of the 6, and lr 3e-3 was 0.430 (§8).
5. **The order can flip with scale** (§6). The result of looking at DN7A with the 35M model and the position-block split (2 seeds) was LoRA 0.57 ≥ full 0.53 ≥ probe 0.49, a different order from 8M. Even so, it did not exceed the B1 of the same seeds, 0.59 and 0.61. There are only 2 seeds, so we do not generalize to "LoRA is better at 35M". Also, the 0.49 of the 35M frozen probe was lower than the 0.56 of the 8M probe of the same seeds. A larger model does not always mean a better frozen representation.
6. **There are things LoRA reduces and things it fails to reduce** (§5). The trainable parameters are 61,761, only 0.83% of the full 7,409,402. However, the training time for the same number of steps fell by only 11–12% at 8M and 24–25% at 35M. This is because the gradient computation of the frozen weights is skipped but backpropagation through the activations remains as it is. How much it shrinks depends on the model structure and the implementation. With 5 epochs fixed, the last-epoch training loss on DN7A was 0.278 for LoRA, higher than the 0.163 of full fine-tuning. In this setting it may be underfitting. That the output of the initial LoRA model is exactly the same as the original, with a maximum absolute difference of 0, was confirmed in the implementation check.

> **There are limitations.** There are only two models, 8M and 35M, and 35M was used on only one assay. It uses 3 short proteins and trains a randomly initialized linear head **at the same time** as the backbone. LP-FT, which trains only the head first, or initializing the head with a zero-shot score, was not tried†. The large gradients of the initial head may have damaged the pretrained representation and worked against fine-tuning. There are 3 seeds, and 2 for 35M and the sensitivity experiments. The training data is about 1 thousand; the epochs and LR were fixed in advance and only a small sensitivity analysis was looked at, with no validation-based tuning and no intervals computed. It does not represent the results of larger models, more data and well-tuned fine-tuning. Biderman 2024's "LoRA learns less and forgets less" is a claim in an LLM setting, so these results can neither verify nor refute it.

### 5-B. ESM-2 hidden states seen through an SAE (notebook 02)
The 7,397 human Swiss-Prot proteins are split into 5,917 train and 1,480 Pfam-group test, and 83 keywords are used. The model is ESM-2 35M and the SAE is a TopK SAE with dictionary 3,840 and k=32. The values in the table are the **median** of the per-keyword test AUROC, and the brackets are the Pfam-group bootstrap 95% interval. The features are chosen on train and measured on test.

| Method | Layer 6 | Layer 12 |
|---|---|---|
| Original dimensions 480 (both signs) | 0.631 [0.623, 0.679] | 0.664 [0.647, 0.693] |
| PCA 480 | 0.645 [0.623, 0.696] | 0.665 [0.644, 0.692] |
| **SAE** (positive direction) | **0.678** [0.636, 0.720] | **0.701** [0.676, 0.721] |
| Untrained random dictionary | 0.657 [0.627, 0.697] | 0.663 [0.653, 0.693] |
| Logistic regression (supervised, for reference) | 0.821 | 0.828 |

The pre-specified primary comparisons are written as the difference of the medians with the 95% interval, and the number of keywords where the SAE was higher. At layer 6, SAE − original dimensions is +0.022 [+0.016, +0.053] and it was higher on 53 of the 83 keywords. SAE − random dictionary is +0.022 [+0.015, +0.049] and 58. At layer 12, SAE − original dimensions is +0.037 [+0.016, +0.056] with 52, and SAE − random dictionary is +0.028 [+0.014, +0.042] with 52. The reconstruction error FVU of the SAE is 0.145 at layer 6 and 0.195 at layer 12, and for the random dictionary 0.745 and 0.747, respectively.

**What we learned**
1. **The SAE reconstructs well, but reconstruction is not interpretability** (§3). It explains 80–85% of the variance, while the random dictionary reaches only 25%. But concept alignment must be measured separately.
2. **There is a gain, but it is small** (§5). Looking at the *difference between the medians* in the table, SAE − original dimensions is +0.047 at layer 6 and +0.037 at layer 12, and SAE − random dictionary is +0.021 and +0.038, respectively. The pre-specified comparison that first computes the per-keyword differences and then takes their *median* is +0.02–0.04 as in the paragraph above, and the interval does not include 0. The SAE is higher on 63–70% of the keywords. But it is far from the 0.83 of the supervised probe. A single unsupervised feature cannot be the whole representation of a concept.
3. **The selection effect is large** (§5). The untrained random dictionary is similar to or slightly higher than the original dimensions (0.657 and 0.631 at layer 6), and PCA is similar too, at 0.645 and 0.665. When it was re-run on Colab, PCA at layer 12 was 0.694, 0.03 higher than the random dictionary, so "PCA ≈ random dictionary" was not reproduced (§8). If you pick the best on train out of thousands of candidates, the median train AUROC of 0.74–0.78 falls to 0.63–0.70 on test. The drop is 0.08–0.14 depending on the method and layer. The SAE went down from 0.784 to 0.678 at layer 6, and from 0.782 to 0.701 at layer 12. However, the rules for choosing the candidates differ. For the SAE the candidates are 3,840 positive directions, and for the original dimensions and PCA they are 480 directions in both signs. Only the random dictionary is a control with the same number of candidates and the same rule as the SAE. Hard keywords fell below chance level on test. Deafness fell from 0.66 on train to 0.45 on test, and Lipid-binding from 0.69 to 0.42.
4. **For easy concepts, it fits well even without training** (§6). At layer 12, the best SAE features are GTP-binding 0.99, Signal 0.99 and Transmembrane 0.98. But the random dictionary also gets 0.97, 0.94 and 0.93, and the original dimensions 0.95, 0.80 and 0.85. Even when all of the top 5 test proteins have the keyword, if they are one family, like 5 RHO GTPases or 5 SLC35 transporters, they are not 5 independent pieces of evidence.
5. **Individual features hardly corresponded across seeds** (§6). Comparing the decoder directions of two SAEs that differ only in seed on the same data, the median nearest cosine was 0.312. Between random dictionaries it is 0.162, and only 0.8% of the features had a cosine of 0.9 or more. Yet the median keyword AUROC was similar for the two seeds, 0.713 and 0.701. With 2 seeds and the nearest cosine alone, you cannot assert either "the aggregate performance is stable" or "the features are reproduced". This is because the nearest cosine is not evidence of a one-to-one correspondence or of functional equivalence. The reconstruction FVU is 0.195 and it is a small-scale 12-epoch setting, so the effect of undertraining may also be there.
6. **There were no dead features, and the activation density is skewed** (§4). If counted by the protein-level max, every feature looks on and the metric becomes meaningless, so we recounted with 1,893,297 residues as the unit. Both the SAE and the random dictionary had 0% dead features. The mean density of TopK is k/F, that is 0.83%, but for the SAE the median density is 0.29% at layer 6 and 0.43% at layer 12, and the top 1% of features are on in 11% and 8% of the residues, so the distribution is skewed. The random dictionary had medians of 0.69% and 0.66%, and top 1% of 2.9% and 3.3%. This means that "high-density" features that are on very often are mixed in with the concept features.

> **Limitations**: ESM-2 35M, human proteins, UniProt keywords (some are annotations transferred on the basis of sequence similarity; a test positive count ≥ 20 was used for choosing the evaluation keywords, but it is the same for all methods), 1 Pfam-group split, one setting of dictionary size, k and epochs, and the second seed used only for the stability check. The bootstrap is conditional on the fixed feature selection and one split, and the (keyword, round) pairs excluded as incomputable in resampling were 0.18%. The implementation was simplified: ReLU is applied after TopK, and the projection that removes the decoder-direction component of the gradient at decoder column normalization was not done (it may differ from the implementation of the reference paper†). Layer 12 (`hidden_states[12]`) is a value after the final LayerNorm and layer 6 is not. A feature–keyword correlation is not causal evidence that the model *uses* the concept (exercise 4). It does not represent the results of large-scale SAEs such as InterPLM.

### 5-C. Checks of efficiency and distributed training (notebook 03)
On Mac MPS, ESM-2 35M was run with batch 4 and the median forward+backward time was measured 3 times. The memory is the MPS **driver allocation**, a proxy for actual usage.

| Length | eager fp32 | sdpa fp32 | sdpa bf16 autocast | sdpa fp32 + checkpointing |
|---|---|---|---|---|
| 128 | 0.11 s · 0.63 GB | 0.11 s · 0.63 GB | 0.12 s · 0.68 GB | 0.13 s · 0.41 GB |
| 512 | 0.41 s · 3.50 GB | 0.41 s · 3.50 GB | 0.47 s · 2.45 GB | 0.51 s · 1.35 GB |
| 1024 | 1.00 s · 7.84 GB | 0.98 s · 7.85 GB | 1.11 s · 7.90 GB | 1.28 s · 2.48 GB |

In the first run, the sdpa at length 128 jumped to 1.63 GB, which was because of buffers left over from the previous numerical check. After adding cleanup code and re-running, it returned to 0.63 GB. Even with the same setting, the time fluctuated by about 10% from run to run. The eager at length 1024 going from 1.09 seconds to 1.00 seconds is an example.

**What we learned**
1. **`sdpa` gives the same numbers as `eager`, but on this Mac the memory proxy was also the same** (§1). The maximum absolute difference of the last-layer output was 6.2e-6, and for lengths of 256 and above the memory proxy was the same, 3.50 GB and 7.84 GB. MPS has no FlashAttention kernel, and which implementation `sdpa` picked could not be confirmed. It was different on the Colab T4, which uses CUDA. The maximum allocated memory at length 1024 fell from 7.34 GB for eager to 2.58 GB for sdpa (§8). It differs by device, so we do not extrapolate.
2. **In the observed range, the time was nearly linear in length** (§1). When the length was increased 8 times from 128 to 1024, the time became about 9 times, from 0.11 seconds to about 1.0 second. This is only an observation that the $L^2$ term did not show up clearly in this range. An explanation such as the linear-layer cost dominating because the hidden dimension is 480 is a hypothesis.
3. **The bf16 autocast gave no time gain on this equipment, and its memory effect differed by length** (§1). Compared with sdpa fp32, the time rose by 9–20%. The memory proxy fell from 3.50 GB to 2.45 GB at length 512, but it rose from 0.63 GB to 0.68 GB at length 128 and from 7.85 GB to 7.90 GB at length 1024. The gain on CUDA tensor cores† cannot be judged from this result.
4. **Checkpointing greatly reduces the memory proxy but increases the time** (§1). At length 1024, the memory fell by 68%, from 7.85 GB to 2.48 GB, and the time rose by 31%, from 0.98 seconds to 1.28 seconds. At lengths 128–512 the time rose by 15–24%.
5. **You can estimate with the memory budget formula** (§2). It is a calculation without the activations, not a measurement, and is based on pure fp32 AdamW. Full fine-tuning a 650M model with fp32 AdamW needs about 10.4 GB. Applying LoRA to q,v with r=8 gives 1.35M trainable parameters and needs about 2.6 GB, and keeping the frozen weights in bf16 makes it 1.3 GB. At 8M and 35M the difference is 0.1–0.5 GB, so it matters little. The parameters of the "35M" model are 33.27M, unlike the name.
6. **DDP produced the same update as a single process** (§3). Comparing, after 4 steps, a run with 2 processes of 8 samples per rank against a run with 16 samples in 1 process, the maximum absolute difference of the trainable parameters was 3.7e-9. That is about 3×10⁻⁷ times the maximum amount moved during the steps, 1.32e-2. The hashes of the two ranks were the same, and the initial parameters at step 0 were also exactly the same, with a maximum absolute difference of 0, in the single process and in the 2 processes. The negative control run without the sampler had a difference of 4.7e-4, about 120,000 times larger. This equivalence was confirmed only in a synthetic setting that satisfies the conditions of §4, that is, even shards and dropout 0. **That the rank hashes are the same alone is not evidence that the data split was right.** The hashes were the same even when run without the sampler.
7. **macOS has a pitfall.** With `torchrun --standalone` alone, the local host name resolves to `1.0.0.127.in-addr.arpa`, and it failed with a rendezvous timeout (`DistNetworkError`) after 300 seconds. Giving `--local-addr=127.0.0.1` solved it.

> **There are limitations.** It was measured on one Mac, and running the same setting twice makes the time fluctuate by about 10%. The memory is a proxy that includes the cache and is not a peak. The CUDA branch was run on a Colab T4 and measured with `max_memory_allocated` (§8). The effects of bf16 and checkpointing may differ on CUDA. DDP was checked only with CPU gloo, 2 processes, and NCCL, multiple nodes and FSDP/ZeRO could not be run. The DDP time includes model loading.

## 6. Benchmark design
A good benchmark defines not a score but **what can be said and what cannot be said**.
- Write down the **task and the labels**. State whether the labels come from experiments, annotation or prediction, how large the uncertainty is, and what the known errors are. *Annotations transferred on the basis of sequence similarity*, like the keywords and text of module 05, are a leakage path.
- When designing the **split**, first write down the leakage paths you want to block. Homology, the same position, the same donor or institution, the same scaffold, near-duplicate text and time are such paths. Then, after splitting, **measure leakage in numbers**. The test→train nearest similarity is an example (module 05).
- Decide the **metrics and the uncertainty** too. Have several metrics that fit the nature of the task, and use group-level bootstrap and several seeds. Do not rank by a single average score. The per-assay-type reporting seen in module 02 is an example.
- Have three kinds of **baselines**. A method that learns no parameters (nearest-neighbor transfer, position mean. It uses the train labels but estimates only on train), simple features such as one-hot, composition and fingerprints, and a random-initialization control.
- Check for **contamination**. Check whether the pretraining data and the evaluation data overlap, and if you could not confirm it, write it as a limitation.
- Prevent **saturation and overfitting**. If you keep looking at the same test and choosing models, you get fitted to the test. You need a hidden test, version pinning and an update policy.
- Leave what is needed for **reproduction**. Record the data version and hash, the code, the environment and the seeds. Be especially careful with material that *changes each time you download it*, like the UniProt release problem of module 05.
- Read critically. The criticism of the "do-everything benchmark" by Raji et al. (2021, title level) and the reporting items of DOME seen in module 01 are useful references.

## 7. Data and model licenses, and research accountability
We organize **on what grounds we confirmed** the licenses of the materials used in this course. We wrote separately what was confirmed on 2026-11-05 and what could not be confirmed. This is not legal advice.

| Material | License | Grounds | Notes |
|---|---|---|---|
| UniProtKB data | CC BY 4.0 | The text of the `LICENSE` file on the UniProt FTP | Attribution required |
| ESM-2 weights (8M·35M) | MIT | The `license` of the Hugging Face model metadata | Derivatives (LoRA, etc.) also need the original license notice† |
| ProteinGym **repository code** | MIT | GitHub API `license` | **The DMS data itself** has an original source per assay and is **unverified** |
| Geneformer | Apache-2.0 | HF metadata (module 04) | |
| all-MiniLM-L6-v2 | Apache-2.0 | HF metadata | |
| ChemBERTa-77M | **Not stated** | No license in the HF metadata, no model card | Needs checking before redistribution |
| MoleculeNet CSVs | **Unverified** | — | The original source of each dataset needs checking |
| Pathology FMs such as UNI·CONCH | cc-by-nc-nd-4.0 stated, access approval required | HF metadata | Non-commercial and no-derivatives conditions†; this repository does not use them |

- You also have to decide how to treat **derivatives**. LoRA weights cannot be used without the base model and are *derived* from it, so it is safer to consider that they follow the terms of the base model†. The licenses of training data such as tables and labels may also apply separately.
- Also review the **dual-use** issue. Baker & Church (Science 2024) deal with protein design and biosecurity (title level). Sandbrink (2023) distinguishes, in the abstract, that LLMs lower the barrier for non-experts, while biological design tools (BDTs) can expand the capabilities of skilled actors. In practice, **concretely evaluate the paths of misuse** before release. Decide whether the scope of release is full, restricted or on request, and also decide the usage restrictions in the model card and the access logs. Also check the biosafety and ethics review requirements of your institution†.
- **The attitude of this repository** is as follows. A gated model is used only when the user has personally agreed to the terms, and an agent does not agree or authenticate on the user's behalf (module 05 §8).

## 8. Cluster and Colab
- The **SLURM template** (`templates/slurm_ddp.sbatch`) was checked only for syntax with `bash -n`. The partition, account and module names are **placeholders**, and it could not be run on an actual SLURM cluster. It is in an **unverified** state.
- **`templates/train_ddp.py`** was run with `torchrun --local-addr=127.0.0.1 --nproc_per_node=2` on CPU gloo with 2 processes to check that it is equivalent to a single process (notebook 03 §3). GPUs with NCCL, multiple nodes and FSDP/ZeRO could not be run.
- **Colab T4** (2026-11-06, 1 run each; Python 3.13, torch 2.14.1+cu130, transformers 5.22.1, peft 0.21.2, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.4.6 — the Mac has scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4): all three notebooks finished without errors, and the code cells that were run are the same as in the repository. What we ran into while preparing the environment: a new VM has no `Bio` or `peft`, so they must be installed, and the image's `torchao 0.10.0` conflicts with `peft 0.21.2` (`ImportError: ... torchao`), so it was removed. 03 reads the `config.json` of ESM-2 150M·650M from the HF cache (`local_files_only=True`), so in a new environment you must put those files there first. In the DDP check of 03, when there is a GPU, rank 1 failed because it could not open `cuda:1`, so we passed `CUDA_VISIBLE_DEVICES=""` to the child processes and fixed it to CPU (gloo) with 2 processes.

| Notebook | Colab T4 | Mac | Comparison of results |
|---|---|---|---|
| 01 | 721 seconds (8M main experiment 438 · 35M check 159 · sensitivity 96) | about 1,156 seconds | The B1, B2 and probe in the random split were the same, and LoRA and full were within ±0.02. **The fine-tuning in position-block differed by up to ±0.08** (below) |
| 02 | 792 seconds (1st pass 55 · SAE training 74 · 2nd pass 93 · selection·evaluation 36 · bootstrap 229) | about 780 seconds | The reconstruction error (FVU) was the same, and the signs and intervals (not including 0) of the 4 pre-specified primary comparisons were the same. The sizes, and the PCA and individual features, differed (below) |
| 03 | 131 seconds | 76–94 seconds | The memory budget table and the DDP equivalence (maximum absolute difference 3.7e-9, negative control 4.7e-4) were the same. The attention measurements differ because they are CUDA (below) |

- **01**: In the position-block split, the order probe ≥ full > LoRA was kept in all three assays (DN7A 0.56/0.46/0.42, IF1 0.46/0.38/0.29, A0A 0.23/0.23/0.16). But the numbers differed from the Mac: DN7A full 0.54 → 0.46, IF1 LoRA 0.22 → 0.29. The number of seeds in which LoRA beat the probe was 1/9 rather than 0/9 (A0A seed 0). In the LoRA sensitivity (DN7A, position-block, seeds 0–1), the default setting was the lowest of the 6 (0.411; range 0.411–0.507) and lr 3e-3 was 0.430 (Mac: the second lowest, 0.385, with lr 3e-3 the worst at 0.294). For full fine-tuning, the default was 0.540, lr 1e-5 0.501 and lr 2e-4 0.503. The 35M (2 seeds) was probe 0.49 · LoRA 0.58 · full 0.52, in the same order as the Mac. The time saving of LoRA was 16–19% at 8M (Mac 11–12%) and 23–24% at 35M (Mac 24–25%). **That is, the fine-tuning numbers in the position-block split fluctuate this much with device and seed, and only the "order" conclusion of this section was reproduced.**
- **02**: SAE − original dimensions / SAE − random dictionary (median of the per-keyword differences): layer 6 +0.035 [+0.019, +0.054] / +0.037 [+0.029, +0.054], layer 12 +0.039 [+0.019, +0.051] / +0.033 [+0.020, +0.047] (Mac: +0.022 / +0.022 / +0.037 / +0.028). The signs and intervals are the same, but the sizes were different. **PCA was different**: at layer 12 it was 0.694, 0.03 higher than the random dictionary (0.663) (Mac 0.665 vs 0.663). The median keyword AUROC of the SAEs that differ only in seed was 0.705 and 0.703 (Mac 0.701 and 0.713). The individual features differed in each environment: the feature that fits GTP-binding was 2564 on the Mac and 2816 on Colab.
- **03 (CUDA)**: With ESM-2 35M, batch 4 and length 1024, the maximum allocated memory (`max_memory_allocated`; a different metric from the Mac's driver proxy) was 7.34 GB for eager fp32, 2.58 GB for sdpa fp32, 0.61 GB for sdpa + checkpointing and 6.67 GB for sdpa bf16 autocast, and the times were 0.65 / 0.56 / 0.71 / 0.74 seconds. On Mac MPS the proxy memory of sdpa and eager was the same, so **the memory effect of the attention implementation differs by device.** The T4 does not support bf16 in hardware†, so the autocast was the slowest. Running the same setting twice gave times within ±5% of each other.
- In module 04, the T4 was not faster than the Mac MPS (Geneformer inference, scVI), but for 01 of this module the T4 was about 1.6 times faster (mostly GPU training). 02 mixes GPU (passes, SAE) and CPU (bootstrap), so the total times were similar. It is a single run.

## 9. Comprehension check

**Q1.** When a result comes out in which the frozen probe beats LoRA and full fine-tuning, what should you check before concluding that fine-tuning is "useless"?
<details><summary>Answer</summary>
You must check five things. First, were the settings fixed in advance? In the LoRA sensitivity of notebook 01, results under the same conditions fluctuated between 0.29 and 0.50. Second, were early stopping or the LR chosen on a validation basis? That was not done here. Third, how large is the training data? It is small, about 1 thousand. Fourth, how many seeds? In the position-block split the per-seed values fluctuated from −0.10 to 0.60. Fifth, what is the split? In the random split, full fine-tuning was the same as or slightly higher than the probe. So "it did not lead in this fixed setting and this split" is the accurate statement.
</details>

**Q2.** What does LoRA reduce and what does it fail to reduce? Answer with the numbers of the notebook.
<details><summary>Answer</summary>
The trainable parameters shrink, in the 8M model, to 61,761, which is 0.83% of the full 7,409,402. The optimizer state and the gradient memory shrink accordingly (the formula of §2). However, only the gradient computation of the frozen weights is skipped, and backpropagation through the activations remains. So the training time fell by only 11–12% at 8M and 24–25% at 35M. How much is saved depends on the model structure and the implementation. Also, the last-epoch training loss was higher for the same number of steps, which may be underfitting.
</details>

**Q3.** Why does the position one-hot (B2) come out similar to or higher than the ESM-2 probe in the random split, and what does that imply?
<details><summary>Answer</summary>
In the random split, other substitutions at the same position are in train, so the per-position mean effect can be used. This is because the one-hot columns of B2 have learned values at the test positions too. In the position-block split that information is absent, so B2 becomes similar to B1 (DN7A from 0.81 to 0.61). There are two implications. The gain of ESM in the random split is overestimated unless you compare it with a baseline that uses the position effect. And to predict new positions, you need a split like position-block.
</details>

**Q4.** Why must you not look at the sensitivity table of notebook 01, pick the best LoRA setting and report its test score? How should that table be used?
<details><summary>Answer</summary>
If you choose the setting by the test score, you get an optimistic value fitted to the test set. It is the same principle as the layer choice in module 04. Choose the setting on a validation split inside the training data, split in the same way as the outer split. Use the sensitivity table only to report how much the conclusion depends on the setting (exercise 2).
</details>

**Q5.** Why is a random-dictionary control needed in the evaluation that matches SAE features with keywords? What did this module show?
<details><summary>Answer</summary>
Because if you pick the one that fits best on train from thousands of candidates, even a meaningless direction looks good by chance — a selection effect arises. The median test AUROC of the untrained random TopK dictionary at layer 12 was 0.663, similar to 0.664 for the original dimensions, and the SAE was +0.028 [+0.014, +0.042] higher than that. The empirical difference from this particular random control, +0.028, is small. And how much of the SAE's gain comes from the overcomplete sparse readout itself and how much from training cannot be decomposed by this experiment alone. There were also keywords such as Deafness and Lipid-binding in which the feature that was the best on train fell below chance level on test.
</details>

**Q6.** What can be said, and what cannot be said, from the observation that "feature 2564 turns on in GTP-binding proteins"? How would you see causation?
<details><summary>Answer</summary>
One thing can be said. The activity of this feature shows a high correlation with the keyword, an AUROC of 0.99, on held-out proteins. Three things cannot be said. That the model *uses* that feature for judging GTP binding, that the feature points to only one concept (the top 5 are one family, so the generality is also limited), and that the same feature is reproduced when the seed is changed (median nearest cosine 0.31). To see causation, you must measure how the downstream prediction changes when the contribution of the feature is removed (ablation), compared with the case where a random feature is removed (exercise 4).
</details>

**Q7.** The median keyword AUROCs of two SAEs that differ only in seed were similar, but the individual feature directions hardly corresponded. How should the interpretation be written?
<details><summary>Answer</summary>
The aggregate performance was similar in the two seeds, but with only 2 seeds the stability cannot be asserted. The individual directions of the dictionary hardly corresponded, and the nearest cosine itself does not guarantee a one-to-one correspondence or functional equivalence. So do not assert that "feature number N is concept X". Instead, report the correspondence across several seeds and the change with dictionary size and sparsity together, and write as limitations whether the reconstruction is sufficient (FVU 0.195, 12 epochs) and the possibility of undertraining. Do not generalize the result of this one setting.
</details>

**Q8.** Why can training be wrong even when the parameter hashes of all the ranks are the same in DDP? How do you check?
<details><summary>Answer</summary>
DDP averages the gradients so that all the ranks make the same update. This equivalence itself holds only under conditions such as even shards, the same initial values and stochastic operations turned off. However, even when the data is split wrongly — for example, when there is no sampler and all the ranks see the same samples — the parameters across the ranks are the same as one another. In notebook 03, the hash of the run without the sampler was also the same. But the maximum absolute difference from the single process that used the same global batch was 4.7e-4, about 120,000 times larger than the 3.7e-9 of normal DDP. To check, compare the parameters with a single process run with the same global batch and seed.
</details>

**Q9.** You want to release the LoRA weights fine-tuned in this module. List three or more things to check before releasing.
<details><summary>Answer</summary>
First, check the license of the base model and the license of the training data. ESM-2 is stated as MIT, but you must check the notice obligations†. ProteinGym's repository code is MIT, but for the DMS data you must check the original source per assay, and this repository could not confirm it. Second, write the training data and split, the limitations (about 1 thousand variants, 3 assays) and the usage restrictions in the model card. Third, review dual use. Look at whether the functions that these weights enable could be misused, and decide the scope of release and the access-control options (Baker & Church 2024, Sandbrink 2023). Fourth, check your institution's policy and review requirements. Fifth, record the revision, seeds and environment for reproduction. This is not legal advice.
</details>

---

## 10. Reading (reading_list section 06)
- Required: Hu et al. 2021 (LoRA) · Simon & Zou 2025 (InterPLM) · Baker & Church 2024
- Adaptation and critique: Biderman et al. 2024 · Sledzieski et al. 2024 (PEFT for pLMs)
- Interpretation: Cunningham et al. 2023 · Gao et al. 2024
- Efficiency: Dao et al. 2022 (FlashAttention) · Micikevicius et al. 2017 · Rajbhandari et al. 2019 (ZeRO) · Zhao et al. 2023 (FSDP)
- Benchmarks and accountability: Walsh et al. 2021 (DOME) · Raji et al. 2021 · Sandbrink 2023

## 11. Notebooks and templates
| # | File | Tier | Time (Mac) | Content |
|---|---|---|---|---|
| 01 | [01_lora_vs_frozen_vs_full](notebooks/01_lora_vs_frozen_vs_full.ipynb) | 🟢 | about 20 minutes (8M main experiment 719 seconds, 35M check 259 seconds, sensitivity 178 seconds) | B1·B2·probe·LoRA·full fine-tuning, random vs position block, 35M·sensitivity |
| 02 | [02_sae_esm2_features](notebooks/02_sae_esm2_features.ipynb) | 🟢 | about 13 minutes (1st pass about 130 seconds, SAE training 105 seconds, 2nd pass 179 seconds, bootstrap 111 seconds) | TopK SAE, keyword alignment, random-dictionary·PCA·original-dimension controls, seed stability |
| 03 | [03_efficiency_and_ddp_check](notebooks/03_efficiency_and_ddp_check.ipynb) | 🟢 | about 1–2 minutes (76 seconds in total) | Measuring attention implementations·precision·checkpointing, memory budget, DDP equivalence (CPU, 2 processes) |
| — | [templates/train_ddp.py](templates/train_ddp.py) · [templates/slurm_ddp.sbatch](templates/slurm_ddp.sbatch) | 🔴 | — | A DDP LoRA training script (verified on CPU) · a SLURM template (**not run**) |

- The notebooks are committed to the repository **without outputs**. Notebook 02 reads the UniProt table (`.cache/uniprot/`) that module 05 notebook 02 downloaded.
- Files downloaded: none (ProteinGym·ESM-2·UniProt come from the caches of modules 02 and 05; 03 also reads the `config.json` of ESM-2 150M·650M from the cache). Installation: `peft` (about 0.84 MB wheel). The times are single-run values with no overlap with other jobs.

## Next
In [module 07](../07_capstone/) you carry out, with the tools so far, 1 project or the reproduction of 1 paper. The habits of this module (not choosing settings on the test set, simple baselines and controls, checking for the selection effect, distinguishing correlation from causation, stating what was not measured) are used as they are in the capstone evaluation rubric.

<!-- i18n-source: 06_advanced_research/README.md blob:68ce9e9b79ef864693da061af5b1cece90a4498c -->
