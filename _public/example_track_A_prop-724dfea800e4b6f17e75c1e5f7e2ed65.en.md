# Example — Track A: Does a frozen ESM-2 8M probe beat the position baseline at *new positions*? (DN7A)

> **English translation.** AI-assisted translation of the [Korean original](example_track_A_proposal_and_report.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

> **What this example is**: It was filled in with the actual run results (about 20 seconds, 5 seeds) of the [template notebook](../templates/capstone_notebook_template.ipynb) in order to show the format. **It is not a real preregistration.** The design and the results were made at about the same time, and one rule was also changed after seeing the results (see the change log). So all of the "settings/rules" below were **fixed for demonstration purposes**. A learner's project must finalize (commit) the proposal before the results. The conclusions must not be generalized to other proteins or models. The rubric self-assessment at the end shows that *this example is not yet a finished capstone*.

---
# Part 1. Proposal (short form)
- **Track/tier**: A · 🟢 Mac (about 20 seconds) · no downloads or installs (module 02 and 06 caches)

## Question and claim scope
- Question: "In single substitutions of the DN7A stability assay, does a Ridge probe on the variant-position embedding of a frozen ESM-2 8M have a higher Spearman than the baseline that uses a position one-hot (B2) when predicting **position blocks that are not in train**?"
- Maximum scope that can be claimed: this one assay, one ESM-2 8M, a Ridge probe, 5 seeds.
- **What cannot be said**: other assays or model sizes, fine-tuning, layer choice, the effect of overlap with the pretraining corpus.

## Data and sources
| Item | Content |
|---|---|
| Data | ProteinGym v1.3 substitution DMS, `DN7A_SACS2_Tsuboyama_2023_1JIC` (length 55, 1,008 single substitutions, 55 positions) |
| Hashes | zip `3a83766254ac9ac9…`, reference CSV `a8f498011532a74a…` (provenance.json) |
| Model | `facebook/esm2_t6_8M_UR50D` @ `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`, loaded from a local snapshot path |
| License | ESM-2 weights MIT (HF metadata), ProteinGym repository code MIT (GitHub API), **the individual DMS data are unverified** |
| Duplicates | Checked `mutant` uniqueness with an assert (single substitutions, so there are no sequence duplicates) |

## Split, comparisons, metrics
- The leakage path to block is **another substitution at the same position being in train**. The split rules compare random (20% test) and position-block (20% test, in units of blocks of 10 consecutive positions). After splitting, measure the test fraction, the fraction of test positions that are in train, the fraction of the same wt→mt pair in train, the position distance to train, and the **number of test clusters (blocks)**.
- Comparisons: **B0** (the mean of the nearest train position; it learns no parameters and uses only train labels), **B1** (wt·mutant one-hot + BLOSUM62), **B2** (B1 + position one-hot), **probe** (frozen 8M, last layer at the variant position), **probe-randinit** (random initialization created fresh for each seed), and the negative diagnostic **probe-shuffled**. B1, B2, probe and randinit use the same Ridge pipeline (standardization then Ridge, with α chosen by 3-fold group CV inside the training data).
- The primary metric is Spearman. The **primary comparison** is probe − B2 in position-block, using 5 seeds. The split and the random initialization change with each seed, and the test sets of the seed repeats overlap each other.
- The **decision rule fixed for demonstration** is defined in one place, `decide()` in the code. It is 'supported' if the seed-mean Δ ≥ 0.02, the seeds with Δ > 0 are ≥ 4/5, and, when an interval can be computed (every seed has ≥ 8 test clusters), the lower bound of the 95% interval is > 0. Otherwise it is 'not supported'. The interval is a group bootstrap over position blocks, it is conditional, and it cannot reflect the overlap between seeds.
- The settings fixed for demonstration are block 10, Ridge and the last layer. Sensitivity looks at block sizes {3, 5, 10} and is for reporting.
- Accountability: since this is protein stability prediction, the dual-use risk is judged to be low †. Checking institutional policy is assumed to be not applicable. The license of the DMS data must be checked before release.

---
# Part 2. Report (short form)
## Summary
When predicting new position blocks of DN7A, the frozen ESM-2 8M probe was lower than the position baseline B2 (seed-mean Δ −0.065, 0/5 seeds). Under the demonstration decision rule the verdict is **'not supported'** (the interval was not computed because there were only 2 test clusters). In the random split, the difference between probe and B2 was small and its sign changed with the seed (+0.010, 4/5). It is limited to one assay · 8M · Ridge only · 5 seeds.

## Split and leakage measurement (seed mean)
| Split | test fraction | test positions that are in train | same wt→mt pair in train | median position distance to train | number of test clusters |
|---|---|---|---|---|---|
| random | 0.20 | 1.00 | 0.88 | 0 | 6 |
| position-block | 0.31 | 0.00 | 0.77 | 4.6 | 2 |

position-block blocks the position-sharing path (0.00), but adjacent positions (distance 4.6) and substitution types (0.77) are still in train. Because the block sizes are not uniform (the last block has 5 positions), the test fraction is 31%, not 20%, and train is smaller. Since there are only 2 test clusters, **the group bootstrap interval was not computed**.

## Results (test Spearman, seed mean)
| Method | random | position-block |
|---|---|---|
| B0 (mean of the nearest train position) | 0.69 | −0.04 (1 NaN excluded) |
| B1 | 0.69 | 0.59 |
| B2 | 0.80 | 0.57 |
| probe | 0.81 | 0.50 |
| probe-randinit | 0.64 | 0.49 |
| probe-shuffled (negative diagnostic) | −0.02 (SD 0.09) | −0.08 (SD 0.19, largest \|ρ\| 0.29) |

| Paired Δ (per seed, seeds won) | random | position-block |
|---|---|---|
| probe − B2 | +0.010 (+0.02, +0.01, +0.02, +0.01, −0.02; 4/5) | **−0.065 (−0.00, −0.01, −0.06, −0.19, −0.06; 0/5)** |
| probe − B1 | +0.112 (5/5) | −0.083 (0/5) |
| probe − probe-randinit | +0.167 (5/5) | +0.012 (+0.08, +0.05, +0.12, −0.12, −0.07; 3/5) |

For sensitivity, the block sizes 3 / 5 / 10 had at least 4 / 3 / 2 test clusters, the seed mean of Δ (probe − B2) was −0.064 / −0.013 / −0.065, and the seeds won were 2/5 · 2/5 · 0/5.

## Interpretation
- **Directly supported**: at new positions the probe (0.50) was lower than B1 (0.59) and B2 (0.57), and the verdict under the demonstration rule is 'not supported'. In the random split the difference between probe and B2 is small and the sign changes with the seed, so **the direction could not be determined**.
- The contribution of pretraining (probe − random initialization, with a fresh initialization for each seed) is clear in the random split (+0.167), but in position-block **the direction could not be determined** (+0.012, 3/5). If it had been compared with a single random initialization (seed 0) only, it would have looked like +0.06. If there is only one control, the luck of that initialization becomes the conclusion.
- **Hypothesis level**: the interpretation that the high score in the random split is explained by the position effect (100% of test positions are in train) is consistent with the leakage table above, but no experiment to separate the factors was done. The reasons why the probe is lower than B1 (the limits of a 320-dimensional linear head with about 700 training variants, the limits of the 8M representation, etc.) could not be confirmed.
- **Cannot be said**: other assays, model sizes, fine-tuning, layer choice. At block size 5 the difference is nearly 0, so the conclusion depends in part on the split setting.

## Limitations and what was not done
Only one assay and 8M were covered, and only Ridge and the last layer were used. There are 5 seeds, and because the split repeats have overlapping test sets, they are not independent repeats. **Because there are only 2–6 test clusters, no interval was computed.** There are 5 random-init controls and one pretrained model. The number of training variants differs by split (random ≈ 806, position-block ≈ 700). The license of the DMS data is unverified. The dual-use review is at the descriptive level. No independent review was received. `repo_dirty=True` when the notebook ran (a formal submission would be run after committing).

## Change log
| Date | What changed | Why | When the error was observed / whether the primary comparison result had been seen |
|---|---|---|---|
| 2026-11-06 | Sensitivity block sizes {5,10,20} → {3,5,10} | At length 55, blocks 15 and 20 gave only 3–4 groups, so the split and the inner CV did not work | Observed as a run error. The primary-comparison value had already been printed in that run (seen) |
| 2026-11-06 | Interpretation of the negative control changed to "diagnostic", and the verdict wording to "check whether the mean is near 0" | In position-block the \|ρ\| of one seed was 0.29 | After seeing the primary-comparison result. Only the wording criterion was changed; the decision rule of the primary comparison was unchanged |
| 2026-11-06 | Decision rule changed from "interval lower bound > 0" to "per-seed Δ (mean ≥ 0.02, seeds won ≥ 4/5) + an interval only when there are enough clusters" | An independent review pointed out, and it was confirmed, that a bootstrap interval is meaningless with 2 test clusters | After seeing the primary-comparison result (the verdict is the same, 'not supported'. It is stated that the rule change came after seeing the result) |
| 2026-11-06 | Random-init control changed from seed 0 only → fresh for each seed | The description ("the initialization changes with each seed") and the code differed | After seeing the primary-comparison result. The probe − randinit value changed from +0.063 to +0.012 |
| 2026-11-06 | Added a baseline that learns no parameters (B0) | Requirement of rubric dimension 4 | After seeing the primary-comparison result |

## Rubric self-assessment (example)
| # | Dimension | Score | Evidence / reason |
|---|---|---|---|
| 1 | Question and claim scope | 2 | The question, scope and what cannot be said are stated. There is no effect-size criterion |
| 2 | Preregistration and change management | 1 | The design and the results are from the same time (because it is an example). A change log exists |
| 3 | Data, provenance, leakage | 2 | Hash, revision, assert, leakage table. A partly unverified license is marked |
| 4 | Baselines and controls | 2 | B0 that learns no parameters, simple-feature B1 and B2, random initialization created fresh for each seed, negative diagnostic. There is no control that separates the selection effect |
| 5 | Evaluation and uncertainty | 2 | 5 seeds, paired Δ, it is stated with reasons that no interval was computed because there were too few clusters, and per-seed values and an effect-size criterion were used; 1-factor sensitivity |
| 6 | Reproducibility | 2 | Run from the start, executed copy = submitted code, the same numbers on a repeated run, number sentences generated by code. Recorded as `repo_dirty=True` (a formal submission runs after committing); the check of the text against the output was done by hand |
| 7 | Interpretation and limitations | 2 | Distinguishes directly supported / hypothesis / cannot be said, and has a list of what was not done |
| 8 | Accountability and communication | 1 | Dual-use review and independent review are incomplete |
| | **Total / 24** | **14** | Below the proposed pass line of 16. Filling in dimension 2 (committing the proposal before the results) and dimension 8 (independent review and dual-use review) would reach it. Structurally, this example cannot obtain either. |

<!-- i18n-source: 07_capstone/examples/example_track_A_proposal_and_report.md blob:fc889592e44b3b25131a5a2c330cb96f2adf8780 -->
