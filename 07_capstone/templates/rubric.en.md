# Capstone Evaluation Rubric

> **English translation.** AI-assisted translation of the [Korean original](rubric.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

Score each of the 8 dimensions from 0 to 3, for a maximum of 24 points. **Write down the evidence (file, cell, table)** and then assign the score. Get both a self-assessment and an independent review (by a human or a model), and discuss the dimensions where the two differ a lot. The pass and expert criteria below are **arbitrary criteria proposed by this course** and not a standard.

| Score | Meaning |
|---|---|
| 0 | Absent or wrong |
| 1 | Partial: present, but a key element is missing |
| 2 | Satisfactory: meets all the items under "2 points" below |
| 3 | Excellent: the "2 points" items + the "3 points" items |

## Criteria by dimension

| # | Dimension | 1 point | 2 points (satisfactory) | 3 points (excellent) |
|---|---|---|---|---|
| 1 | **Question and claim scope** | There is a question, but no scope or limitations | A one-sentence question, the maximum scope that can be claimed, and an explicit statement of **what cannot be said** | + defines in advance a falsifiable decision rule and an effect-size criterion that has practical meaning |
| 2 | **Preregistration and change management** | A proposal exists, but the primary comparison and decision rule are vague | Split, metric, primary comparison and decision rule **fixed before the results**, with a change log | + distinguishes whether a change came after seeing the results, and marks exploratory analyses separately from pre-specified ones |
| 3 | **Data, provenance, leakage** | Only a source is given, with no hash, revision or license evidence, or the leakage check amounts to the name of the split | Records hashes, revisions and versions; deduplicates; gives *evidence that the license was looked up* (or states "unverified"); **measures leakage per split with numbers** | + finds the paths that the split did not block, and checks their impact with stratification or a stricter split |
| 4 | **Baselines and controls** | 1 baseline, or an evaluation with a procedure different from the model's | **3 or more**, including a baseline that **learns no parameters** (may use train labels, estimated only on train), a simple feature baseline and random initialization, all with the **same procedure and the same split** | + controls that separate the selection effect/the contribution of pretraining (random initialization *fresh for each seed*, untrained dictionaries, etc.) and negative diagnostics (shuffling labels or inputs; a diagnostic, not a proof) |
| 5 | **Evaluation and uncertainty** | Only seed means, or intervals that ignore the correlation unit | Seed/split repeats, a **paired comparison**, and a **group bootstrap** over the correlation unit (only when there are enough test clusters, stating that it is a conditional interval). When clusters are too few, **write why the interval is not computed** and judge from per-seed values, consistency and a pre-set effect-size rule; state the metric definition (pooling, ties, gallery); state that seed repeats are re-splits of the same data and not independent repeats | + one-factor-at-a-time sensitivity, setting selection separated from the test set (fixed in advance or nested validation), and a distinction between statistics (difference of medians vs median of differences) |
| 6 | **Reproducibility** | A notebook exists, but the environment and seeds are not recorded | Runs to the end in a clean environment, provenance recorded, **code of the executed copy = submitted code** | + generates the number sentences and tables with code and checks that the text matches the output, compares repeated runs, reports time and cost |
| 7 | **Interpretation and limitations** | There is interpretation, but no generalization limits or list of what was not done | Distinguishes "directly supported / hypothesis / cannot be said", presents negative and unexpected results as they are, and has a **list of what was not done** | + experiments that directly check alternative explanations, and marking of uncertain statements (†) |
| 8 | **Accountability and communication** | Mentioned only in part | Checks the rights to use and release the data and models before use (if they could not be checked, limits the scope of release and redistribution), a dual-use review and a decision on the scope of release, and a report with a clear structure | + documentation at the level of a model/data card, and **verifies** the points raised in the independent review and records the grounds for accepting or rejecting them |

## Automatic caps (red flags)
If any of the following is present, the corresponding dimension (in parentheses) gets **at most 1 point**.
- A setting (layer, epochs, LR, threshold, keywords, etc.) was **chosen by test results and that test result was reported as the primary result** (regardless of whether it was described as "fixed in advance"). If there is no separate test set, marking it as an *exploratory result* lifts the cap. (2, 5)
- Wrote "no leakage" without measuring leakage after the split. (3)
- Standardization, PCA, feature selection or regularization selection was fitted on data that included the test set. (3, 5)
- The numbers in the text differ from the run output. (6, 7)
- Left out negative or unexpected results, or turned them into a generalizing sentence ("FMs are …"). (7)
- Accepted review points without verifying them, or ignored them without grounds. (8)
- Used a revision that was not pinned, remote code that was not read, or weights/pickles whose source was not checked. (3, 8)
- Agreed on the user's behalf to the terms of a gated model or dataset. (8)

## Score sheet

| # | Dimension | Score (0–3) | Evidence (file, cell, table) | red flag | Reviewer score | Discussion of differences |
|---|---|---|---|---|---|---|
| 1 | Question and claim scope | | | | | |
| 2 | Preregistration and change management | | | | | |
| 3 | Data, provenance, leakage | | | | | |
| 4 | Baselines and controls | | | | | |
| 5 | Evaluation and uncertainty | | | | | |
| 6 | Reproducibility | | | | | |
| 7 | Interpretation and limitations | | | | | |
| 8 | Accountability and communication | | | | | |
| | **Total / 24** | | | | | |

The **proposed criteria** are arbitrary and are as follows. A pass is a total of 16 or more, with no 0-point dimension and no red flag. The expert-level checkpoint is a total of 20 or more, with **every dimension at 2 or more and no red flag**. The rubric looks at the *quality of the procedure*, not at the size or novelty of the result. A negative result can also earn full marks.

<!-- i18n-source: 07_capstone/templates/rubric.md blob:edfeea8c5af58ab479f6b887037d768d6ce6e29c -->
