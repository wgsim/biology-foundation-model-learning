# Module 02 Exercises

> **English translation.** AI-assisted translation of the [Korean original](exercises.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

No answers are provided. Check for yourself whether you meet the **evaluation criteria**, and record the results with the [experiment log template](../00_orientation/experiment_log_template.en.md). Report every metric together with seeds/repeats and confidence intervals or standard deviations.

## Exercise 1 — probe vs baseline vs split (module deliverable)  *(notebook 01)*
Add **one more assay of a different type** to notebook 01 (e.g., Expression or Activity), and change at least two of the following and report how the results change.
- Position block size 5 / 10 / 20
- Reduce the number of training samples to 1/4
- Choice of the ESM-2 embedding layer (last layer vs an intermediate layer). Use `embed(..., layer=k)` of notebook 01 (a `hidden_states` index, 0 to the number of layers)
- Evaluation criteria: ① a table that includes the baselines (B1, B2), ② the results of the two splits, random and position-block, ③ at least 2 hypotheses on the conditions under which the ESM features beat the baselines and those under which they do not, ④ a statement of limitations (number of assays, number of repeats), ⑤ interpret the rise of B2 in the random split without asserting that it is "memorization", in light of the **purpose of the split** (generalization to new positions vs predicting the unmeasured substitutions of already-measured positions).

## Exercise 2 — Model size and assay type  *(notebook 02)*
Add 3 or more assays to notebook 02 (criteria from the ProteinGym reference table: length ≤ 250, single substitutions ≥ 1500), and plot the change in Spearman with model size (from 8M to 650M).
- Evaluation criteria: ① a position-level bootstrap confidence interval, ② for assays that do not improve as size grows, give a hypothesis **if there are any**, and **if there are none, report that there are none** and interpret by the overlap of the confidence intervals (it may differ depending on the assays you add), ③ describe per assay whether there is an improvement over the BLOSUM62 baseline, ④ explain with your own results "why you must not judge a model by one average".

## Exercise 3 — Score variants and cost  *(notebook 02)*
Implement one more besides the masked marginal and the wild-type marginal (e.g., the pseudo-log-likelihood (PLL) ratio of the variant sequence, or the average over several masked positions).
- **Cost caution**: The PLL ratio needs masked forward passes of (L for the variant sequence + L for the wt sequence) per variant. In the measurements of this repository (Mac M1 Pro, batch 32, sequence length 100–250), the 8M model processed about 270 sequences per second and 150M about 44 sequences per second, so running all of a 2,500-variant assay takes tens of minutes even for 8M and several hours for 150M. **Randomly subsample about 100 variants** and start with 8M–35M.
- Evaluation criteria: ① a table of the Spearman of the three methods and the **forward time** (measured with device synchronization after warm-up), ② state the subsample size and the seed, and compare with the masked/wt marginal on the same variant set, ③ describe the performance-vs-cost trade-off, ④ compare your numbers with the official ProteinGym score table (web) and describe why they differ (number of assays, only single substitutions used, version, etc.).

## Exercise 4 — contact  *(notebook 03)*
Add one or more other PDB structures (length 100–250, your choice, with the source and license noted), and compare the results while changing `MIN_SEP` and the contact criterion (Cβ 8 Å).
- Evaluation criteria: ① a table of precision@L/5 for at least two settings (by model size), ② a per-layer precision curve or a per-head heatmap, ③ **compute the gap between the precision of the head chosen by leave-one-protein-out and the precision of the best head within the evaluated protein (the optimism gap)**, and state that with few proteins even the leave-one-out itself is uncertain, ④ report in a table the effect of each correction step — raw / symmetrization / APC / diagonal removal, ⑤ explain why the unsupervised attention-based score differs from a supervised-trained contact head.

## Exercise 5 — Does pLDDT tell us the effect of variants?  *(notebook 04, Colab 🟡)*
From a small ProteinGym assay (e.g., `DN7A_SACS2_Tsuboyama_2023_1JIC`, 55 aa), sample single variants and fold them with ESMFold, and compute the **Spearman between the pLDDT change (variant − wt) and the DMS score**. Compute both the overall average and the average over a **window around the variant (±5 residues)**.
- **Resource planning**: Notebook 04 was confirmed by running with the real weights on a Colab T4 (2026-11-06, a single environment). All the code cells take about 1.5 minutes (model load about 40 seconds), and the first run that downloads the weights takes about 3 minutes (measured with the code before the load order was changed). This time includes folding 12 sequences of length 56–76 aa (2 natural proteins, 2 control sequences, 8 point mutants), so the time per sequence was not measured separately. Also, right after the model is loaded, about 14.1 GB of GPU memory is used at the instantaneous peak (on a T4 with 15.6 GB), so the notebook does not run as it is on a smaller GPU. Therefore first measure the time per sequence and the memory with a **pilot of 20 variants**, then decide the target sample size (about 200), and if resources are short, reduce the sample but report that fact and the confidence interval.
- Evaluation criteria: ① sample size, sampling method and confidence interval, ② comparison with the ESM-2 zero-shot score on the same variants, ③ a conclusion, with grounds, on whether pLDDT is useful for variant-effect prediction.

## Exercise 6 — Paper note
Organize Lin et al. 2023 (ESM-2/ESMFold) **or** Notin et al. (ProteinGym) with the [paper note template](../00_orientation/paper_note_template.en.md).
- Evaluation criteria: ① fill in the pretraining–evaluation overlap, split and baseline items of the checklist with grounds from the main text, ② a summary of the limitations stated by the authors, ③ an account of the reasons (differences in settings) when the numbers of this module's notebooks differ from the paper's numbers.

## Exercise 7 (optional, 🔴 cluster or GPU) — Design self-consistency
Design 8 sequences with ProteinMPNN or ESM-IF1 from the backbone of 1UBQ (or another small structure), refold them with ESMFold, and report the **pLDDT and the backbone RMSD**. Follow the README of each official repository for tool installation and usage (this repository has no notebook for it).
- Evaluation criteria: ① the sequence identity of the 8 designed sequences (against wt), ② a table of refolding RMSD and pLDDT, ③ an explanation, in 3 or more sentences together with your own results, of "why in silico self-consistency does not guarantee experimental success".

## Exercise 8 (optional) — SAE features
Open the public InterPLM materials (paper and repository), pick three ESM-2 SAE features and summarize which concepts they correspond to, and write a design of **what experiment is needed to verify that correspondence causally**.
- Evaluation criteria: ① the grounds for the chosen features (examples of activating sequences, annotations), ② the design of the causal verification (an intervention experiment), ③ limitations.

<!-- i18n-source: 02_protein/exercises.md blob:011af0d2eea64e62029660043bc8940bb0085b51 -->
