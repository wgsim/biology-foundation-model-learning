# Dry-Run Records for Tracks B, C and D

> **English translation.** AI-assisted translation of the [Korean original](README.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

> **What this record is**: Track A runs directly as the [template notebook](../../templates/capstone_notebook_template.ipynb), but B–D only pointed to starting points. This record is **what the people who made the course did: they chose an MVP (one pre-specified comparison) for each track, moved it into the template structure (preregistration → provenance → data inspection → unit and leakage → baselines → evaluation → controls → sensitivity), and actually ran it.**
> - **It is not a learner pilot.** The time a learner would need and where they would get stuck were not measured.
> - **It is not a real preregistration.** The design and the results were made at about the same time, and it cannot be proven that the decision rules were set before the results were seen. It is in the same position as the [track A example](../example_track_A_proposal_and_report.en.md), which is meant to show the format.
> - Each comparison is **a single run**. Do not generalize the results below to "DNA LMs / single-cell FMs / chemical LMs are …".
> - It received Codex and agy cross-review (see the "Cross-review" section at the bottom). Each point was checked against the code and incorporated, and for the points that were not incorporated the reason is given.
> - The other half of the questions that this record left as "not done" was run next in [second_halves.en.md](second_halves.en.md) (2026-11-09).

The environment is a Mac (CPU/MPS) and the repository `.venv`, and there are no new downloads (only the caches that modules 03, 04 and 05 had downloaded were used). The code is the 3 scripts in this folder, run from the repository root (`HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I <script>`). The run record (`provenance.json`) is written to `.cache/capstone_runs/<RUN_NAME>/` (excluded from git).

## Summary
| Track | MVP (one pre-specified comparison) | Time | Was anything needed in the cache | Primary comparison result (decision rule) |
|---|---|---|---|---|
| **B DNA** ([script](track_B_traitgym_llr_vs_conservation.py)) | On TraitGym non-coding variants, is the AUPRC of NT LLR higher than that of a conservation score (phyloP-241m)? | about 16 seconds | Yes (variant table, 17 score files). No model run | not supported (Δ −0.571, chromosomes 0/14) |
| **C single-cell** ([script](track_C_head_geneformer_vs_pca.py)) | Is a *supervised* head (logistic) on Geneformer V2 embeddings better than the same head on PCA, by donor? | 70 seconds | The embeddings were there. **The fine-grained labels were not, and had to be reconstructed** | not supported (macro-F1 Δ −0.018, donors 3/13) |
| **D molecules** ([script](track_D_lipo_cutoff_sweep.py)) | Does ChemBERTa-MLM − ECFP4 change across the similarity cutoffs {0.3, 0.4, 0.5, 0.6} of the Lipophilicity cluster split? | 842 seconds (about 14 minutes) | Yes (CSV, ChemBERTa cache) | Not supported at any cutoff (effectively a tie at 0.4) |
| **E paper reproduction** ([record](track_E_traitgym.en.md)) | Reproduction of the numbers of the 22 TraitGym public metric CSVs (L1: score → metric) + whether the ranking holds (L3) | 140 seconds | The scores were there. The 22 public metric CSVs (2,058 B) were downloaded after approval | All 22 reproduced (maximum absolute difference about 1e-16). The ranking holds when the metric definition is changed, and of the 4 of the 5 subsets that could be judged, it holds in 2 |

In all three tracks (B, C and D), the **positive control** (can the decision rule output 'supported'?) and the **negative control** (are shuffled scores and labels at chance level?) came out as expected. That the primary comparison is "not supported" is the result in this setting, and it does not mean that the rule always outputs a negative.

---
## Track B — NT LLR vs conservation score on TraitGym
- **Setup (fixed)**: TraitGym `mendelian_traits_matched_9` (3,380 variants, 338 positives, 9 matched controls per 1 positive = 338 `match_group`s, 19 chromosomes). The metric is AUPRC. The sign of each score was fixed before looking at the data (LLR −, conservation +). Δ = AUPRC(model) − AUPRC(baseline). The decision rule is 'supported' if all three of the following conditions hold: the mean of the per-chromosome Δ (the 14 chromosomes with ≥ 5 positives) is 0.02 or more; the number of chromosomes with Δ > 0 is 10/14 or more; and the lower bound of the 95% interval of the `match_group`-level paired bootstrap (500 resamples) is greater than 0. The per-score permutation null has 1,000 resamples.
- **Data inspection (assert)**: The score files and the variant table have the same number of rows, there is no NaN, each group has exactly 1 positive, each group lies within one chromosome, and there are no duplicate variants. All of them passed.
- **AUPRC**: The permutation p is, *for each score*, a comparison against the pooled-AUPRC distribution of 1,000 shuffles of that score's own values within `match_group`. The null mean is 0.100–0.102, the 95% quantile is 0.103–0.112, and the minimum of p is 0.001.

| Score | AUPRC | permutation p |
|---|---|---|
| NT LLR (provided by TraitGym) | 0.103 | 0.445 |
| Evo2-7B LLR | 0.381 | 0.001 |
| phyloP-241m | 0.673 | 0.001 |
| phyloP-100v / phastCons-43p | 0.653 / 0.497 | 0.001 / 0.001 |
| TSS distance only (matching variable) | 0.115 | 0.001 |

- **Primary comparison** NT LLR − phyloP-241m: pooled Δ −0.571 [−0.618, −0.519], chromosome-mean Δ −0.561, 0/14 chromosomes with Δ > 0, so it is **not supported.** The secondary comparison Evo2-7B LLR − phyloP-241m has Δ −0.293 [−0.348, −0.231], 1/14, and is not supported. Both comparisons were pre-specified, and no multiple-comparison correction was made.
- **Controls**: The negative control shuffles the model score within `match_group`, and its AUPRC is 0.096 (the null mean of NT LLR is 0.102). The positive control is Evo2-7B LLR vs TSS distance only, with pooled Δ +0.266 [+0.213, +0.314], chromosome-mean Δ +0.231, 13/14, so it is 'supported'.
- **Sensitivity (one at a time)**: Changing the baseline to phyloP-100v and to phastCons-43p is still not supported in both cases (Δ −0.550 / −0.394). Changing the minimum number of positives per chromosome to 3 and to 10 gives Δ > 0 in only 1 of 17 and 0 of 10 chromosomes, respectively.
- It matches the AUPRC of module 03 notebook 02 (NT 0.103, Evo2-7B 0.38, phyloP-241m 0.67, TSS 0.115). The permutation p did not match before the cross-review of this record. Notebook 02 used a single null distribution for all scores (the values obtained by shuffling the within-group ranks 0…9), giving NT 0.34 and TSS 0.006. The permutation distribution of the observed statistic (pooled AUPRC) is obtained by shuffling each score's own values, so notebook 02 was fixed in the same way (see "Cross-review" below), and it is now NT 0.41 (this script gives 0.445 because the seed is different; Monte Carlo standard error about 0.015) and TSS 0.001.
- **What was changed from the template**: This track has *no training (zero-shot)*. `make_split`, `fit_predict`, `PRED`, Ridge and the random-init model are not needed; the per-seed Δ slot was moved to the **per-chromosome Δ**, and `MIN_TEST_GROUPS` to `MIN_POS_CHROM` and the number of `match_group`s. `decide()` could be used as it is. The negative control is not "shuffling labels" but "shuffling scores within groups".
- **Where it got stuck / cautions**: ① The question said "to change the split unit, first check whether coordinates exist", and the variant table had `chrom`, `pos` and `match_group`, so the chromosome unit was possible. ② The AUPRC of a chromosome with only 2–8 positives fluctuates a lot (so the minimum number of positives was pre-specified). ③ The directly computed NT LLR covers only 210 chr1 variants (21 positives), which is unsuitable for an MVP, so the provided score was used.
- **What cannot be said**: It is only a comparison on this variant set and this metric, and it does not apply to DNA LMs in general or to other diseases and complex traits. The model variant and the window length of the NT score provided by TraitGym were not checked†.
- **What was not done**: The other half of the question ("how much does the gain of the NT probe depend on the layer choice and the size of the 6-mer baseline?") was not run in this record, because the NT embeddings were not in the cache. It was later run in [B① of second_halves.en.md](second_halves.en.md) by computing the embeddings anew (about 6 minutes).

## Track C — supervised head on Geneformer embeddings vs PCA head
- **Setup (fixed)**: A 10,400-cell subsample of Wilk 2020 PBMC, 13 donors, leave-one-donor-out by donor. The head is logistic regression (C = 1 fixed; centering, L2 normalization and standardization are fitted only on the training donors). **However, the PCA (HVG, 50) baseline of the primary comparison is a transductive setting in which HVG selection, scaling and PCA are fitted (without labels) on all cells including the held-out donor** (the same way as module 04 notebook 02). This favors PCA, and there is no such fitting for the Geneformer embeddings. Fold-wise fitting is examined separately in the sensitivity analysis. The primary comparison is V2-104M `<cls>` L-2 vs PCA (HVG, 50), with 13 coarse labels. The metrics are the pooled macro-F1 (donor-level bootstrap, 1,000 resamples) and the per-donor balanced accuracy. The decision rule is: mean of the per-donor balanced-accuracy Δ ≥ 0.02, donors with Δ > 0 ≥ 9/13, and the lower bound of the macro-F1 Δ interval > 0.
- **Where the cache blocked**: `pbmc_sub.h5ad` has **no** fine-grained label (`cell.type.fine`), and the cell names were also changed to `cell0…`, so they cannot be linked to the original (obs: donor, disease, label, cell_type_cxg, n_counts). `balanced_subsample` in notebook 04/01 (seed 0, 800 per donor) is deterministic, so it was applied again to `wilk2020_pbmc.h5ad`; after confirming that the donor, coarse label and cxg label of all 10,400 cells agree, the 20 fine-grained labels were attached (the script asserts this). There are no new downloads. PCA (HVG) was not in the cache, so it was built by the same procedure as notebook 04/02 (7 seconds).
- **Results** (13 coarse labels, logistic head):

| Representation | pooled macro-F1 | mean per-donor balanced accuracy |
|---|---|---|
| PCA(HVG, 50) | 0.856 | 0.841 |
| Geneformer V2 `<cls>` L-2 | 0.838 | 0.806 |
| Geneformer V1 mean L-2 | 0.724 | 0.691 |
| V1 random initialization | 0.526 | 0.508 |
| (negative control) V2, training labels shuffled | 0.068 | 0.075 (uniform guessing over 13 classes 0.077) |

- **Primary comparison** V2 − PCA: macro-F1 Δ −0.018 [−0.031, −0.009], per-donor balanced-accuracy Δ −0.036, donors 3/13, so it is **not supported.** The control, the pretraining effect (V1 − V1 random initialization), has pooled macro-F1 Δ +0.198 [+0.182, +0.223], mean per-donor balanced-accuracy Δ +0.183, 13/13, so it is 'supported'. Unless stated otherwise, the Δ below is the *pooled macro-F1* Δ, the same as the interval, and the decision rule looks together at the mean and the number of wins of the per-donor balanced-accuracy Δ and at this interval.
- **Sensitivity (one at a time)**: If PCA is fitted per fold, leaving out the held-out donor (centering and L2 normalization applied the same as in the primary comparison), Δ is −0.015 [−0.031, −0.004], balanced-accuracy Δ −0.030, 3/13. If the head is changed to kNN (k = 15), Δ is −0.036 [−0.064, −0.018], balanced-accuracy Δ −0.090, 0/13 (the same as the zero-shot numbers of module 04 notebook 02: PCA 0.847, V2 0.811). If the labels are changed to the 20 fine-grained ones, Δ is −0.079 [−0.097, −0.062], balanced-accuracy Δ −0.103, 0/13 (PCA 0.824, V2 0.744). All of them are not supported.
- **Reading**: The supervised head narrowed the gap (from kNN −0.036 to logistic −0.018) but stayed below PCA, and the gap grew as the labels became finer. It is the result for one dataset and 1 head hyperparameter (C = 1).
- **What was changed from the template**: The unit is the **donor** instead of the seed (the split is deterministic), and logistic regression and macro-F1 are used instead of Ridge and Spearman. The negative control is "shuffling the training labels", and for the random-init control the V1 random-init embeddings in the cache were used.
- **Cautions**: ⓪ The logistic head in this script has `max_iter=300`, while the C②③④ scripts of [second_halves.en.md](second_halves.en.md) written afterward use `max_iter=3000`. For that reason the balanced accuracy of the same PCA baseline is written as 0.841 here and 0.842 there (the pooled macro-F1 is 0.856 in both; no effect on the verdict). Also, the "PCA per fold" here (0.835) is the one in which only PCA is fitted per fold, while the "strict per fold" there (0.841) is a different procedure in which even the filter, HVG and scaling are fitted on the training folds. ① HVG selection was done with all the cells (a limitation that remains even when PCA is fitted per fold). ② The disease status overlaps completely with the donors (7 COVID, 6 normal). ③ With `max_iter = 300`, the logistic convergence warning appeared 13 times. When re-run with a copy in which only `max_iter` was raised to 3000, there were no warnings and the numbers above were the same (only the negative control changed from 0.068 to 0.069). The warnings come from joblib worker processes, so they are printed regardless of the `warnings.filterwarnings("ignore")` at the top of the script (also 13 times in the re-run after the review). The "y_pred contains classes not in y_true" warning of `balanced_accuracy_score` also appears. It occurs when a cell type that is not in one donor's test set is predicted, and the balanced accuracy averages only over the types that actually exist in that donor. ④ The `per-fold` PCA also did HVG selection with all the cells.
- **Not done (in the MVP of this document)**: A new dataset, fine-tuning, and other models such as scGPT. Later, in [second_halves.en.md](second_halves.en.md), label granularity (C①), a new dataset (C②), Geneformer V1-10M fine-tuning (C③) and scGPT zero-shot embeddings (C④) were run, and V2-104M fine-tuning was run separately on a Colab A100 (C③-V2).

## Track D — Cutoff sweep of the Lipophilicity cluster split (module 05 exercise 1)
- **Setup (fixed)**: 4,200 Lipophilicity molecules (0 invalid SMILES, 0 duplicates), ECFP4 count (2048) vs ChemBERTa-77M-MLM mean-pooled last layer (revision pinned), a RidgeCV linear head, 5-fold × 5 repeats, the cutoffs {0.3, 0.4, 0.5, 0.6} and a random split. The metric is RMSE, and Δ = RMSE(ECFP4) − RMSE(MLM) (positive = MLM is better). The bootstrap is done 300 times in units of Butina clusters. The decision rule is: mean of the per-repeat Δ ≥ 0.02, repeats with Δ > 0 ≥ 4/5, and interval lower bound > 0.
- **Leakage in numbers** (test→train nearest Tanimoto of the first repeat):

| Split | Number of clusters | Median nearest similarity | p90 | Fraction of test molecules that still have a neighbor at or above the cutoff in train |
|---|---|---|---|---|
| random | — | 0.697 | 0.865 | — |
| cutoff 0.3 | 816 | 0.358 | 0.687 | 0.74 |
| cutoff 0.4 | 1,269 | 0.366 | 0.620 | 0.36 |
| cutoff 0.5 | 1,652 | 0.420 | 0.653 | 0.25 |
| cutoff 0.6 | 2,067 | 0.486 | 0.700 | 0.20 |

  As the cutoff is raised, the clusters split more finely and the nearest similarity goes up. Because the clusters are not transitive, pairs at or above the cutoff remain across folds (74% of the test set at 0.3).
- **RMSE (mean ± sd over 5 repeats)**:

| Split | ECFP4 | ChemBERTa-MLM | MLM random initialization | (negative control) RMSE of ECFP4 trained on shuffled labels, compared with the *original* labels |
|---|---|---|---|---|
| random | 0.750 ± 0.002 | 0.828 ± 0.002 | 1.033 ± 0.002 | 1.204 |
| 0.3 | 0.906 ± 0.006 | 0.930 ± 0.007 | 1.096 ± 0.005 | 1.204 |
| 0.4 | 0.918 ± 0.008 | 0.917 ± 0.010 | 1.080 ± 0.006 | 1.204 |
| 0.5 | 0.868 ± 0.007 | 0.902 ± 0.004 | 1.082 ± 0.005 | 1.205 |
| 0.6 | 0.827 ± 0.004 | 0.883 ± 0.006 | 1.068 ± 0.005 | 1.204 |

  The standard deviation of y is 1.203, and when the labels are shuffled the value should converge to this, and it did.
- **Primary comparison** Δ(ECFP4 − MLM) [95% interval], repeats with Δ > 0: random −0.079 [−0.102, −0.054] 0/5 · 0.3: −0.024 [−0.076, +0.022] 0/5 · 0.4: +0.002 [−0.050, +0.048] 3/5 · 0.5: −0.034 [−0.065, +0.002] 0/5 · 0.6: −0.057 [−0.086, −0.027] 0/5, and it is **not supported in any split.** The intervals include 0 at 0.3, 0.4 and 0.5, and the gap is not monotonic in the cutoff (smallest at 0.4 and larger again at 0.6). The control, the pretraining effect (MLM vs random initialization), has Δ +0.16 to +0.21 and 5/5 in all 5 splits, so it is 'supported'.
- **Observation on the question of exercise 1**: The hypothesis that "raising the cutoff always makes it harder" did not hold in this setting. The ECFP4 RMSE mostly went down (it became easier), from 0.906 to 0.827, going from cutoff 0.3 to 0.6. The rise from 0.3 to 0.4 (0.012) is at the level of the repeat sd of 0.006–0.008. It is in the same range as the values in the module 05 README (random 0.750, about 0.923 near cluster 0.4; 3 repeats).
- **What was changed from the template**: Ridge and Spearman were replaced by RidgeCV and RMSE, and position blocks by Butina clusters. The random split has no cluster unit, so the bootstrap unit was arbitrarily set to the cutoff-0.4 clusters (the meaning of the interval is not exactly the same as in the other cells). The functions of notebook 05/01 are inside cells and cannot be imported, so the `balanced_group_folds`, Butina and ChemBERTa pooling code was copied into the script.
- **What cannot be said**: It is the result for Lipophilicity only, frozen embeddings + a linear head, and a fixed layer and pooling (last-layer mean). Other tasks, fine-tuning and MTR variants were not covered.
- **What was not done**: Exercise 2 (notation augmentation vs normalization), exercise 5 (false-negative masking) and exercise 3 (other MoleculeNet tasks) were run in [D②, D⑤ and D③ of second_halves.en.md](second_halves.en.md). ChemBERTa fine-tuning was not done.
- **Cost caution**: The README rated the cost of track D as 🟢, but this one MVP took about 14 minutes (mostly the 2048-dimensional ECFP4 RidgeCV).

---
## What was confirmed in common
1. **All three tracks could run the MVP to the end using only the cache, with no new downloads.** There was one blocker (the fine-grained labels of track C), and it was resolved by deterministic reproduction.
2. **The template notebook does not work as it is, but its structure can be moved over.** What changed was only the data loading, the features, the split and unit (from seeds to chromosomes, donors and repeats), and the metric. `decide()` and the order "preregistration → provenance → inspection → controls → sensitivity" were used as they were. For comparisons without training, as in track B, the `make_split`/`fit_predict` interface does not fit.
3. **The controls had to be redefined for each track.** The negative control was "shuffling scores within groups" in B, "shuffling the training labels" in C and "shuffling labels" in D, and the positive control that shows the decision rule can output 'supported' was "Evo2-7B vs TSS distance only" in B and "pretraining vs random initialization" in C and D. Without these positive controls we could not have told whether the "not supported" of the three tracks was a bias of the rule.
4. That all three primary comparisons came out "not supported" is partly because of the *choice* of question (asking whether an FM beats a strong baseline). It is not a new conclusion, but in the same direction as modules 03, 04 and 05.

## Cross-review (2026-11-08)
The full diffs of commit 1999904 (Module 02 ESMFold Colab), f406cbd (modules 05–07 Colab) and df8e368 (this folder) were shown read-only to Codex (gpt-6.1-terra) and agy (gemini-4-argon-low) for review (Codex 6 points, agy 8 points). Each point was checked against the code.

**Incorporated**
- **B permutation p (Codex, incorporated)**: The null distribution was built from ranks (0…9) unrelated to the score and was used in common for all scores. It was changed to shuffle *each score's own values* within the group (the table above). The p values that changed are NT LLR 0.338 → 0.445 and TSS distance only 0.006 → 0.001, and the other results (AUPRC, primary and secondary comparisons, controls, sensitivity) are unchanged. **Module 03 notebook 02 had the same rank-based null distribution, so it was fixed in the same way and re-run.** The p-values over all variants went NT 0.34 → 0.41, HyenaDNA 0.53 → 0.63, Caduceus 0.92 → 0.93 and TSS 0.006 → 0.001. The chr1 subset differed greatly. For the directly computed NT LLR, the 1-frame p went 0.035 → 0.109, the 6-frame mean 0.138 → 0.289, and the 6-frame × forward/reverse 0.106 → 0.273, and for the public NT LLR 0.007 → 0.063. On chr1 all the earlier p-values were too small (less conservative). The AUPRC and the bootstrap intervals are unchanged.
- **Transductive fitting of the primary PCA baseline in C (Codex, agy; disclosure strengthened)**: The code already did this and the comments said so, but the README's setup paragraph did not. It is now stated in the setup. The primary comparison itself is the same setting as module 04 notebook 02, so it was not changed, and fold-wise fitting had already been run as a sensitivity analysis.
- **The fold-wise PCA sensitivity in C was changing two factors (Codex, incorporated)**: The fold-wise PCA lacked the training-mean centering and the row L2 normalization. It was aligned with the primary comparison and re-run (Δ −0.017 → −0.015; the conclusion is the same, and all the other numbers were the same as in the earlier run).
- **Added the hash of `wilk2020_pbmc.h5ad` to the provenance of C (Codex)**: It is the file used to reconstruct the fine-grained labels.
- **`MIN_GROUPS` in B was only defined and not used (Codex, agy)**: An assert was added. The selection of the TraitGym snapshot (the first entry of `iterdir()`) now sorts only the directories and stops if there is not exactly one (agy).
- **Notation (agy)**: It was not shown whether the control and sensitivity Δ in C were the pooled macro-F1 or the balanced accuracy, and B's positive control had a pooled interval attached to the chromosome-mean Δ. Both are now distinguished. The numbers of bootstrap and permutation resamples of B are also given.
- **What the D negative control evaluates (Codex)**: It is the RMSE of a model trained on shuffled labels and compared with the *original* labels. The code was left as it is (only a comment was added, AST identical), and it is stated in the table header and the comment. It was not re-run.

**Not incorporated**
- **`decide()` exempts the interval condition when `ci is None` (agy)**: This is the design of the template notebook, and `DECISION_RULE` says "when an interval can be computed". The D script has `MIN_TEST_GROUPS = 8`, and when only the first part of the script was re-run, the minimum number of test clusters in every split was 162–840, so it did not trigger in this run.
- **`decide()` uses `len(deltas) < MIN_WINS` as the lower bound on valid units (agy)**: This is inherited from the template. If this condition is hit, it cannot be 'supported' anyway, and only the label changes to 'cannot judge'. The `MIN_POS_CHROM=10` sensitivity case that agy cited does not call `decide()`.
- **`LogisticRegression` has no `random_state`, and the warnings are hidden so the 13 warnings cannot be seen (agy)**: The lbfgs solver does not use `random_state`. The warnings are printed by joblib workers, and `ConvergenceWarning` was seen 13 times even in the re-run after the review.
- **The `weights_only` comment in the ESMFold notebook (agy)**: `from_pretrained` in transformers 5.22.1 defaults to `weights_only: bool = True`, and `.bin` files are read through the `torch.load(weights_only=True)` path (checked in `modeling_utils.py`). The comment is correct.

**The two Colab commits (1999904, f406cbd)**: Neither reviewer pointed out a problem. The outputs of the notebooks were empty, so the reviewers could not recompute the numbers of the Colab runs — this part was not independently verified.

## What this record did not do
- A learner pilot (measuring time and sticking points) and reading the remaining figures and tables of the track E paper (some were read and compared in step 3 of [track_E_traitgym.en.md](track_E_traitgym.en.md)) were not done. The other half of the questions was run next in [second_halves.en.md](second_halves.en.md) (what was still not done there is written in that document).
- After the review, only B (16 seconds) and C (80 seconds) were re-run to check the fixes. D was not re-run because only a comment changed.

<!-- i18n-source: 07_capstone/examples/dry_runs/README.md blob:97bb677e940590ed0fd1ad09ee5944ac20991545 -->
