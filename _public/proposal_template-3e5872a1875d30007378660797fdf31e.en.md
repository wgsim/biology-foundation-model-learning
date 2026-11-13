# Capstone Proposal (Preregistration) Template

> **English translation.** AI-assisted translation of the [Korean original](proposal_template.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

Write it **before producing any results** and *commit* it (or store it somewhere that keeps a timestamp), then write down the writing date and that commit hash. A document that merely carries a date cannot show that it was not written after the fact. Record every later change, with its date and reason, in the "Change log" at the very bottom. Changing things is fine; hiding the changes is the problem.

```markdown
# [Project title]
- Date written / commit:
- Track: A Protein · B DNA/genome · C Single-cell · D Molecule/multimodal · E Paper reproduction
- Compute tier: 🟢 Mac / 🟡 Colab / 🔴 Cluster (expected time and cost)

## 1. Question and claim (one sentence each)
- Question: "In [setting], is [method A] better than [method B] on [metric]?"
- Maximum scope you can claim (generalization limits): one dataset / one model size / N seeds …
- What this project's results **cannot say**:

## 2. Data and sources
| Item | Content |
|---|---|
| Dataset name / version / file hash | |
| Is a download needed (file, source, size; user approval) | |
| License (evidence checked: file/page/API field) | If it could not be checked, write "unverified" |
| Source and uncertainty of the labels (experiment / annotation / transferred from sequence similarity) | |
| Deduplication rule (sequence / molecule / text / cell) | |
| Possible overlap with pretraining data | Did you check or not |

## 3. Split (before looking at the data)
- Leakage paths you want to block:
- Split rule (unit: variant / position block / family / donor / scaffold / institution):
- How to **measure leakage with numbers** after the split (e.g., test→train nearest-neighbor similarity, fraction of shared positions/families):
- Repeats/seeds: (do the split and the initialization change together? Seed repeats that re-split the same data have overlapping test sets, so they are not independent repeats)
- Lower bound on the number of test positions/groups (below this, do not use a group bootstrap and judge from per-seed values):

## 4. Comparisons
| Name | Needs training? | What this comparison shows |
|---|---|---|
| Baseline that learns no parameters (nearest-neighbor transfer / mean of the nearest train position …; uses train labels but estimates only on train) | No | |
| Simple feature baseline (one-hot / composition / fingerprint / descriptors …) | Yes | |
| Random-init control (generated **fresh for each seed**) | | Contribution of pretraining |
| Model (checkpoint ID + **revision**) | | |
- Evaluate all methods with the **same procedure and the same split** (head, standardization, regularization selection).

## 5. Metrics and decision rule
- Primary metric (one):  Secondary metrics:
- **Primary comparison** (one): [method] − [baseline], paired difference
- Uncertainty: use N seeds, and use a **group bootstrap over the correlation unit** only when there are enough test clusters (minimum number of clusters: ). State that the interval is conditional. If there are too few, do not compute an interval and judge from per-seed values and the effect-size rule.
- Decision rule (what decides "supported" or "not supported". Define the effect-size lower bound · number of seeds won · interval condition as **one function in code**):
- Setting selection: **fixed in advance** / chosen on a validation split (do not choose on the test set). Sensitivity is for reporting.

## 6. Sensitivity and negative controls
- Sensitivity that changes one factor at a time (what, over what range):
- Expected result of the negative diagnostics (shuffling labels / shuffling inputs / random initialization) (the mean should be near 0, and if it deviates, investigate leakage or bugs. A diagnostic, not a proof):

## 7. Accountability
- License constraints (redistribution, commercial use, gated access):
- Dual-use review (does the result/model have functions that could be misused, scope of release):
- Items that need your institution's policy or review:

## 8. Risks and limitations to write in advance
- 3 points where it could fail:
- Will you report it even if the result is negative (no effect): Yes

## 9. Schedule (2 weeks, 16–20 hours)
| Milestone | Deliverable | Expected hours |
|---|---|---|
| M1 Proposal and approval | This document | |
| M2 Data, split, baselines | Leakage measurement table, baseline results | |
| M3 Primary experiment | Paired comparison + interval | |
| M4 Sensitivity and controls | Tables | |
| M5 Report | report_template | |
| M6 Independent review | Record of how the review points were handled | |

## Change log
| Date | What changed | Why | When the error was observed / whether the primary comparison result had been seen |
|---|---|---|---|
```

<!-- i18n-source: 07_capstone/templates/proposal_template.md blob:693389c224de09da1a3ab1fba9cb58dbb766397e -->
