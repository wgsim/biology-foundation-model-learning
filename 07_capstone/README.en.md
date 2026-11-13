# 07. Capstone — Project or Paper Reproduction

> **English translation.** AI-assisted translation of the [Korean original](README.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

**Duration**: Weeks 17–18 (2 weeks, about 16–20 hours) · **Compute tier**: 🟢 local Mac by default (the template notebook was also run and checked on a Colab T4) · 🔴 the cluster was not used in this module
**Prerequisites**: [modules 00–06](../). In particular, the evaluation habits of modules 04–06 (simple baselines, random-init controls, measuring leakage with numbers, settings fixed in advance, paired comparisons).

> Notation: † marks a statement that must be checked directly in the paper text or documents. This module does not teach a new model. It practices **answering one question honestly, to the end, with the tools learned so far**.

---

## Learning goals
1. Before seeing any results, write down a verifiable question and the **maximum scope you can claim and what cannot be said** (preregistration).
2. Design the data source, split, baselines, metrics and decision rule, and **measure leakage with numbers**.
3. Set the strength of your conclusion with paired comparisons, a group-level bootstrap (when there are enough clusters), sensitivity analysis, and negative diagnostics and random controls.
4. Write a report that includes negative results and what was not done, and **verify the points raised in the independent review** before accepting or rejecting them.
5. Self-assess with the rubric, and when you reproduce 1 paper, decide the reproduction level and the tolerance in advance.

## Two-week schedule
| Milestone | Content | Deliverable | Hours |
|---|---|---|---|
| M1 | Choose the track and question, write the proposal (preregistration), check downloads and licenses | `proposal` ([template](templates/proposal_template.en.md)) | 2–3 |
| M2 | Data inspection, split, **leakage measurement**, baselines | Leakage table, baseline results, negative control | 3–4 |
| M3 | Primary experiment (one pre-specified comparison) | Paired Δ + interval, verdict of the decision rule | 3–4 |
| M4 | Sensitivity, controls, extension (optional) | Tables | 2–3 |
| M5 | Report | `report` ([template](templates/report_template.en.md)) | 3–4 |
| M6 | Self-assessment with the checklist and rubric, independent review and how it was handled | Review-handling record, re-run | 2–3 |

**Deliverables**: the proposal, the notebook (runs from start to finish), the report, the rubric score sheet, and the record of how the independent-review points were handled. The evaluation criteria for each milestone are in [exercises.en.md](exercises.en.md).

---

## 1. How this module differs from the others
- **You choose the question, and there is no right answer.** The evaluation looks at the *quality of the procedure* (the rubric), not at the size or novelty of the result. A negative result such as "the baseline won" can also earn full marks.
- **Write the design first.** Fix the split, the metric, the primary comparison and the decision rule before the results, and record any change together with its date and reason. As evidence that you fixed them, **commit the proposal before producing results** (or store it somewhere that keeps a timestamp) and write that hash in the report. A document that merely carries a date cannot show that it was not written after the fact.
- **Start small.** Shrink the project to a size that can be finished in 16–20 hours. Start with one dataset, one question and one primary comparison, and extend only after the MVP is done.
- **Download or install only after the user's approval.** Write the file, source and size in the proposal and get approval. The templates and track starting points in this module use only caches that have already been downloaded.

## 2. Choosing a track
| Track | Starting point (caches and notebooks in this repository) | Example questions | Controls you must include | Leakage and pitfalls to watch for | Cost |
|---|---|---|---|---|---|
| **A Protein variants and adaptation** (modules 02, 06) | ProteinGym v1.3 · ESM-2 8M/35M/150M/650M caches · [06-01](../06_advanced_research/notebooks/01_lora_vs_frozen_vs_full.ipynb) · [**template notebook**](templates/capstone_notebook_template.ipynb) | If you use validation-based selection (06 exercise 2), does the probe vs fine-tuning ranking change? Does it also change for other assay types (activity, expression, binding)? What about model size and layer choice? | Position baseline (B2), baseline that learns no parameters (B0), random initialization | Random split with the same position in train, choosing settings on the test set, number of seeds | 🟢 (≤35M) |
| **B DNA/genome** (module 03) | NT-500M · NT promoter task · TraitGym feature cache · [03 notebooks](../03_dna_genome/notebooks/) | How much does the gain of the NT probe depend on layer choice and on the size of the 6-mer baseline? On TraitGym, how do LLR and a conservation score compare on the same variant set with a permutation null distribution? (To change the split unit, first check whether the data has that information, such as coordinates.) | 6-mer frequencies, conservation-based scores (phyloP/CADD), permutation null | Repeated/duplicate sequences, same gene/chromosome, mismatched variant sets | 🟢 |
| **C Single-cell** (module 04) | Wilk PBMC + Geneformer V1/V2 embedding caches · [04 notebooks](../04_single_cell/notebooks/) | Does the zero-shot ranking change with label granularity (coarse/fine)? How does a *supervised* head on top of the embeddings differ from PCA (04 does only zero-shot kNN)? (A new dataset only after approval.) | PCA (HVG), random initialization, donor-level split | Donor/disease confounding, fitting that includes the held-out data during training (transductive) | 🟢 (V2 inference about 11–14 minutes) |
| **D Molecules and multimodal** (module 05) | 3 MoleculeNet datasets · ChemBERTa-77M · UniProt table · ESM-2 35M · MiniLM caches · [05 notebooks](../05_multimodal_other/notebooks/) | Notation augmentation vs normalization (05 exercise 2), false-negative masking (05 exercise 5), a continuous sweep of the cluster-split threshold (05 exercise 1) | ECFP4 · descriptors, random initialization, nearest-neighbor transfer | Limits of the scaffold split, near-duplicate text, tie and gallery definitions | 🟢 |
| **E Reproduce 1 paper** | A paper you choose ([§3](#3-track-e-paper-reproduction-protocol)) | Is one table cell or one figure of the paper reproduced? What if you change the split or the baseline? | The paper's baselines + a simple baseline you add | Data versions and preprocessing that differ from the paper, differences in evaluation code | 🟢–🟡 |

**Verification level of the starting points**: The template notebook of track A runs as it is. Tracks B–D point to the notebooks and caches of modules 03–05; one question each (the MVP) was moved into the template structure and **run once as a dry run (the author's own rehearsal of the workflow) using the caches alone** ([records and scripts](examples/dry_runs/README.en.md): B about 16 seconds, C about 80 seconds, D about 14 minutes. They were revised after cross-review by Codex and agy, and they were not learner pilots). Most of the **other half of each track's questions** were then [dry-run as well](examples/dry_runs/second_halves.md) (A①②③, B①, C①②③④, D②③⑤). C② was run on a new dataset (CELLxGENE Cross-sectional), C③ with Geneformer V1-10M fine-tuning (Mac) and V2-104M fine-tuning (Colab A100; impractical on a Mac), and C④ with scGPT whole-human zero-shot embeddings. ChemBERTa fine-tuning was not done. All the needed columns were present, but the fine-grained label of track C (`cell.type.fine`) is not in `pbmc_sub.h5ad`, so the subsampling of notebook 04/01 had to be reproduced and the label attached. Check directly, at the proposal stage, whether the labels, coordinates and columns that your question needs are in the cache. For track E, 5 candidate papers were [surveyed](examples/track_E_candidates.en.md), and one paper, TraitGym, was [dry-run](examples/dry_runs/track_E_traitgym.en.md) at L1 (the level of computing metrics from scores) + L3. The 7 Mendelian zero-shot claims in the main text and some tables and figures (Table 1, S4, S8, S9; Figure 5; supplementary figures S7, S8; figures only at the level of ordering) were compared against the cache, but the score-generation step and the remaining tables and figures were not reproduced and could not be verified. A learner's track E uses a paper that the learner chooses, and checking the paper, the data and the compute cost comes first. The **MVP** of each track is *one* of the questions above reduced to a single pre-specified comparison. Extensions (other assays, larger models, new data) are done only after the MVP is finished. If new data or a new model is needed, write the file, source, size and license in the proposal and get approval.

## 3. Track E: paper reproduction protocol
Pineau et al. (arXiv:2003.12206, abstract) describe reproducibility as obtaining results similar to those reported in the paper using the same code and data (where possible). This protocol narrows that definition to the *claim unit*.
1. **Choose a paper** (using the [paper note template](../00_orientation/paper_note_template.en.md) and 3-pass reading): choose a paper that satisfies all 4 of the following conditions. ① The core claim is one sentence. ② Code, data and weights are public, and you have **checked the license**. ③ The computation fits within 16–20 hours (🟢 Mac / 🟡 Colab). ④ You can identify the **claim unit** to reproduce (one table cell or one figure). Choose from the ✅ papers in the [reading_list](../resources/reading_list.en.md), but check directly whether the code is public. There is a [candidate survey](examples/track_E_candidates.en.md) in which the author applied the 4 conditions to 5 candidate papers. It is an example, not the right answer.
2. **Decide the reproduction level**: there are **L1 numerical reproduction** (with public code and weights, the number is within the tolerance), **L2 re-implementation** (implement the core pipeline yourself and compare it with L1), and **L3 variant verification** (change the split, baseline or ablation and see whether the claim holds). The recommendation is L1 plus one L3.
3. **Define the tolerance before you see the primary reproduction result.** If the paper gives an interval, use it. If it does not, first measure and record the *variation from running the original code with only the seed changed* (before seeing the primary result), and set the tolerance from that variation. Do not choose a value such as "±0.02" after seeing the result.
4. **Deviation log**: record the data version, preprocessing, environment, seed and any details that were not in the paper, with dates.
5. **When it does not reproduce**: check the candidate causes (data version, preprocessing, evaluation code, possible leakage, seed) one by one, and compare with the authors' code. "Failed to reproduce" is a valid result, but **if you could not confirm the cause, write that.** Do not assert that the paper is wrong.
6. Use the [report_template](templates/report_template.en.md) for the report, but put the reproduction level and the tolerance in §1, and "the difference between the paper's claim and what I verified" in §5.

## 4. Procedure and templates
- [Proposal template](templates/proposal_template.en.md) → start from the [**template notebook**](templates/capstone_notebook_template.ipynb) → [report template](templates/report_template.en.md) → [pre-submission checklist](templates/checklist.en.md) → [rubric](templates/rubric.en.md) → [independent-review prompt](templates/review_prompt_template.txt).
- [**Example**](examples/example_track_A_proposal_and_report.en.md): a proposal and a report (track A, short form) filled in with the actual run results of the template notebook. It is meant to show the format, and its conclusions are not to be generalized.
- [**Dry-run records for B–D**](examples/dry_runs/README.en.md): records and scripts of running the MVPs of tracks B, C and D in the template structure (what had to be changed from the template, how the controls were defined, the results). Track E is covered in the [TraitGym dry run](examples/dry_runs/track_E_traitgym.en.md).
- [**Dry-run records for the other half of the questions**](examples/dry_runs/second_halves.md): records and 11 scripts of running, in the same way, the questions that were left out of the MVPs above (A①②③, B①, C①②③④, D②③⑤) (only the V2-104M fine-tuning of C③ was trained on a Colab A100). The only new downloads were 3 MoleculeNet CSV files, 1 CELLxGENE H5AD file (150 MB), 26 scGPT source `.py` files and the whole-human checkpoint (about 207 MB), all downloaded after approval (nothing was installed).
- The template notebook changes only the `✏️` cells (settings, decision rule, limitations) and keeps the rest of the structure. It is **dedicated to track A (ProteinGym variants)**, so for other tracks you must replace the data loading, the feature extraction, the split key (cluster unit) and the B0 function. Run records are written to `.cache/capstone_runs/<RUN_NAME>/` (excluded from git).

## 5. What the template notebook shows (demo results)
ProteinGym `DN7A_SACS2_Tsuboyama_2023_1JIC` (length 55, 1,008 single substitutions), ESM-2 8M, 5 seeds (the split and the random initialization change with each seed); it takes about 20 seconds. **These are the results of a demo, and not a generalization of any kind.**

| | random | position-block (10) |
|---|---|---|
| test fraction / fraction of test positions that are also in train / fraction of the same wt→mt pairs in train / median position distance to train / **number of test clusters (blocks)** | 0.20 / 1.00 / 0.88 / 0 / 6 | 0.31 / 0.00 / 0.77 / 4.6 / **2** |
| B0 (mean of the nearest train position; no parameters learned) | 0.69 | −0.04 (1 NaN excluded) |
| B1 (aa one-hot + BLOSUM) | 0.69 | 0.59 |
| B2 (B1 + position one-hot) | 0.80 | 0.57 |
| probe (frozen ESM-2 8M) | 0.81 | 0.50 |
| probe-randinit (random initialization, fresh for each seed) | 0.64 | 0.49 |
| probe-shuffled (negative diagnostic) | −0.02 (SD 0.09) | −0.08 (SD 0.19, largest \|ρ\| of a single seed 0.29) |

| Paired difference Δ (per-seed values, seeds won) | random | position-block |
|---|---|---|
| probe − B2 | +0.010 (+0.02, +0.01, +0.02, +0.01, −0.02; 4/5) | **−0.065 (−0.00, −0.01, −0.06, −0.19, −0.06; 0/5)** ← primary comparison (chosen for demonstration) |
| probe − B1 | +0.112 (5/5) | −0.083 (0/5) |
| probe − B0 | +0.117 (5/5) | +0.532 (valid 4/4) |
| probe − probe-randinit | +0.167 (5/5) | +0.012 (+0.08, +0.05, +0.12, −0.12, −0.07; 3/5) |

**The group bootstrap interval was not computed.** Splitting a length-55 protein into blocks of 10 gives only 6 blocks, and the position-block test set is 2 blocks. Resampling 2 clusters with replacement gives only 3 possible resamples, so an interval is meaningless. The notebook is guarded so that it computes an interval only when the number of test clusters of every seed is at least `MIN_TEST_GROUPS` (8), and the interval code was checked separately on synthetic data (40 clusters) (with equal accuracy the interval includes 0, and when one side is much more accurate it does not). The 5 seeds re-draw the split from the same 1,008 variants, so the test sets overlap. They measure **split sensitivity**, not independent repeats.

Under the decision rule fixed for demonstration purposes ("seed-mean Δ ≥ 0.02, seeds won ≥ 4/5, and, if an interval can be computed, lower bound > 0"), the verdict is **'not supported'** (mean Δ −0.065, 0/5, no interval). In the block-size sensitivity (3 / 5 / 10, minimum number of test clusters 4 / 3 / 2), Δ (probe − B2) is −0.064 / −0.013 / −0.065. How to read it:
- In the random split, the difference between probe and B2 is small (+0.010) and changes sign across seeds (4/5), so **the direction could not be determined.** In this split, 100% of the test positions are in train. In position-block, the probe was lower than B1 and B2, and it lost to B2 in all 5 seeds.
- The contribution of pretraining (probe − random initialization) is clear in the random split (+0.167), but in position-block it is +0.012 and **the direction could not be determined** (3/5). This value was obtained by generating the random initialization fresh for each seed. When the comparison used a single initialization (seed 0), it looked like +0.063. If there is only one control, the luck of that one initialization can become the conclusion.
- B0 (the mean of the nearest train position) does not work on new positions (−0.04). All positions within a block map to the same edge position, so the prediction is nearly constant, and for one seed the Spearman correlation was undefined (recorded as NaN and excluded).
- The seed mean of the negative diagnostic is near 0, but in position-block there are only 2 test clusters, so the |ρ| of a single seed reached 0.3. This diagnostic is not proof that there is no leakage. If the mean is far from 0, use it as a signal to investigate.
- This is the result for one assay, 8M, Ridge only, and 5 seeds. There is not even a conditional interval.

**Colab T4 re-run** (2026-11-06, 1 run, 93 seconds; Python 3.13, torch 2.14.1, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.4.6. The Mac has scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4): the same notebook finished without errors, and the 60 per-seed ρ values agreed with the Mac to within 2×10⁻⁴ (the 1 undefined value was also the same). The table above and the verdict are the same. What differed was block 3 of the block-size sensitivity. The seed mean of Δ (probe − B2) was −0.079 (seeds won 0/5), unlike the Mac's −0.064 (2/5). Block 5 was −0.011 (2/5), against the Mac's −0.013 (2/5), and block 10 was −0.065 (0/5), the same as the Mac. Only the library versions and the device differ, and we could not isolate the cause. Read the sensitivity numbers as values that can vary with the environment.

## 6. Evaluation rubric and checklist
The [rubric](templates/rubric.en.md) has 8 dimensions (question, preregistration, data and leakage, baselines and controls, evaluation and uncertainty, reproducibility, interpretation and limitations, accountability) × 0–3 points. **You write down the evidence and then assign the score, and there are automatic caps (red flags).** Examples are choosing a setting on the test set and then reporting the result evaluated on that same test set as the primary result (regardless of whether you wrote "fixed in advance"; if there is no separate test set, mark it as an exploratory result), writing "none" for leakage without measuring it, and a description that differs from the output. The pass mark (total ≥ 16, no 0-point dimension, no red flag) and the expert-level checkpoint (total ≥ 20, every dimension ≥ 2, no red flag) are **arbitrary criteria proposed by this course**. The [checklist](templates/checklist.en.md) collects what was actually wrong in modules 01–06.

## 7. Getting an independent review
This generalizes the method used in modules 04–06. Send the work read-only to two or more models ([prompt template](templates/review_prompt_template.txt)), and **verify each point one by one with code and output**. Merely sending the same bundle to two models does not make the review *independent*. Record to whom you sent what, and whether information was shared between the reviewers. If you have no model access, substitute a human review, or a **separated self-check** in which you look only at the checklist a day later, as if you were reading it for the first time (record the time and the items you found), and state in the report that the independence is weak. In this course the review found a real bug every time (e.g., duplicate sequences straddling the split, standardization mixed into the inner CV, the PCA sample being the front part of a length-sorted order), but some of the points were misunderstandings (e.g., removing constant columns using the whole data). Telling the reviewer **what you could not verify and what you got wrong and later fixed** reduces false positives. Keep the grounds for accepting or rejecting each point in a table (the report appendix), and after fixing, re-run the notebook from the start.

## 8. Comprehension check: finding flawed report sentences
Find the problem in each of the following sentences. All of them are based on cases that actually occurred in this course, or plausibly could.

**Q1.** "In 5-fold random cross-validation, the ESM-2 probe scored 0.10 higher than a simple baseline that does not use position, so ESM-2 predicts variant effects well."
<details><summary>Answer</summary>
In a random split, other substitutions at the same position are in train, so the difference from a baseline that uses the position effect (B2) can shrink (template demo: probe − B2 = +0.010, so the direction could not be determined, versus +0.112 against B1, which does not use position). If you compared only against B1, which does not use position, or did not use a split that predicts new positions, you cannot make the claim "it predicts new positions". You must show with numbers (the fraction of test positions that are in train) which paths the split blocked and which it did not.
</details>

**Q2.** "I compared layers 4, 8 and 12 and used layer 8, which had the highest test Spearman. The settings were fixed in advance."
<details><summary>Answer</summary>
Choosing the layer by test results and reporting that test score as the primary result is a red flag in the rubric, whether or not you wrote "fixed in advance" (dimensions 2 and 5 are capped at 1 point). If there is no separate test set, it must be marked as an *exploratory result*. If it really was fixed in advance, the layer should have been decided beforehand and the others reported as *sensitivity*; if a selection is necessary, choose with a validation split inside the training data (made the same way as the outer one). The layer selection in module 04 (a difference of 0.18) and the LoRA sensitivity in module 06 show the same problem.
</details>

**Q3.** "A scaffold split was used, so there is no train–test leakage."
<details><summary>Answer</summary>
The name of a split does not guarantee that there is no leakage. In module 05, even with a scaffold split, 26% of the Lipophilicity test set had a molecule in train with Tanimoto ≥ 0.7 (median nearest-neighbor similarity 0.57). Measure and report the test→train similarity after splitting, and check the paths that the split did not block with a stricter split or stratification.
</details>

**Q4.** "For BBBP, I pooled the predictions of all folds to compute the AUROC. The layer-0 `<s>` representation gives 0.46, which is slightly worse than chance."
<details><summary>Answer</summary>
If you mix the score scales of the different models of each fold into one, even a constant prediction deviates from 0.5 because of the differences in the fold-wise mean predictions. If you compute the AUROC per fold and average, this constant representation is exactly 0.500 (module 05). For the same reason, pooled metrics are sensitive to calibration differences between folds.
</details>

**Q5.** "In the mean of 3 seeds, LoRA (0.41) was lower than the probe (0.56), so LoRA is useless for this task."
<details><summary>Answer</summary>
This is only the result of one fixed setting (r=8, lr 1e-3, 5 epochs), 3 seeds, an 8M model, and no validation-based selection. In the sensitivity analysis of the same notebook, LoRA fluctuated between 0.29 and 0.50 depending on the setting, and with 35M and 2 seeds the order was different (module 06). "It did not win in this fixed setting" is the statement that the data support.
</details>

**Q6.** "I incorporated all 12 points raised by the review model."
<details><summary>Answer</summary>
Accepting every point without verification is a red flag in the rubric. Reviewers find real bugs, but they also misunderstand (module 06: removing constant columns using the whole data does not change any prediction, so it is not leakage). Check each point against the code and the output, keep the grounds for accepting or rejecting it in a table, and after fixing, re-run from the start to confirm that the executed copy and the submitted code are the same.
</details>

---

## 9. Reading (reading_list section 07)
- Required: Walsh et al. 2021 (DOME, module 01) · Pineau et al. (reproducibility report)
- Documentation: Mitchell et al. (Model Cards) · Gebru et al. 2021 (Datasheets for Datasets)
- Re-read the critique papers of each module (Kedzierska 2025, Praski 2025, Guo 2024, de Jong 2025, Biderman 2024), and use this module's checklist to evaluate the evaluation design of *those papers* (optional).

## 10. Files
| File | Contents |
|---|---|
| [exercises.en.md](exercises.en.md) | Exercises for milestones M1–M6 and the evaluation criteria |
| [templates/proposal_template.en.md](templates/proposal_template.en.md) | Preregistered proposal |
| [templates/capstone_notebook_template.ipynb](templates/capstone_notebook_template.ipynb) | A runnable skeleton + the track A demo (🟢, about 20 seconds) |
| [templates/report_template.en.md](templates/report_template.en.md) | Report structure |
| [templates/rubric.en.md](templates/rubric.en.md) · [templates/checklist.en.md](templates/checklist.en.md) | Rubric · pre-submission checklist |
| [templates/review_prompt_template.txt](templates/review_prompt_template.txt) | Independent-review prompt |
| [examples/example_track_A_proposal_and_report.en.md](examples/example_track_A_proposal_and_report.en.md) | A filled-in example |
| [examples/dry_runs/README.en.md](examples/dry_runs/README.en.md) | Dry-run records for tracks B, C and D + 3 scripts (caches only) |
| [examples/track_E_candidates.en.md](examples/track_E_candidates.en.md) | Survey of 5 candidate papers for track E (4 conditions applied) |
| [examples/dry_runs/track_E_traitgym.en.md](examples/dry_runs/track_E_traitgym.en.md) | Track E dry-run record (TraitGym L1+L3, check of the main-text claims, comparison of tables and figures) and 3 scripts. The 22 public metric CSV files (2,058 B) were downloaded after approval |
| [examples/dry_runs/second_halves.md](examples/dry_runs/second_halves.md) | Dry-run record of the other half of the questions and 11 scripts (A①②③ B① C①②③④·C③-V2 D②③⑤. For D③ the 3 MoleculeNet CSV files, for C② the 1 CELLxGENE H5AD file, and for C④ the 26 scGPT sources and the checkpoint (about 207 MB) were downloaded after approval) |

- Notebooks are committed to the repository **without outputs**. There are no files to download and nothing to install.

## Wrap-up
This module completes modules 00–07 of the 18-week curriculum. The [glossary](../resources/glossary.en.md), the [benchmark map](../resources/benchmarks.en.md) and the [cheatsheets](../resources/cheatsheets/) in `resources/`, and the full link and citation check, were added and run afterward. **What remains** is running on a cluster (SLURM), a learner pilot, and reproducing the rest of the tables and figures of track E (some were read and compared) and the score-generation step. The checklist so far applies to this repository itself as well. If something is missing, leave an issue so that it can be filled in.

<!-- i18n-source: 07_capstone/README.md blob:b3eba9b5553b0d25f695e47d1767239c73c11831 -->
