# Capstone Report Template

> **English translation.** AI-assisted translation of the [Korean original](report_template.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

Aim for 4–6 pages. Copy every number from **sentences that code generated from the notebook output**, and for each table state which cell's output it comes from. In modules 04 and 05 there are cases where a number was miscopied by hand.

```markdown
# [Title]
- Author / date / repository commit / proposal version:

## Summary (5 lines or fewer)
Include the following: the question, the setup (data, model, split, seeds), the primary result (numbers and interval), the verdict of the decision rule, and the most important limitation.

## 1. Question and preregistration
Paste the proposal's question, primary comparison and decision rule as they are. State **what you changed from the proposal** and why, and also whether the change was made after seeing the results.

## 2. Data and sources
- Record the files, versions and hashes, the model revisions, the environment (package versions, hardware), the license evidence and any unverified items.
- Record the results of the data inspection (number removed by deduplication, missing values, label conflicts).

## 3. Split and leakage measurement
Include a table that reports, for each split, **the overlap with train in numbers**. Also state which paths the split blocked and which it did not.

## 4. Results
### 4.1 Primary comparison (proposal commit hash: )
| Comparison | Δ seed mean (per-seed values) | Seeds won | 95% interval or "not computed: reason (number of test clusters)" | Verdict |
If there is an interval, state in one line what it is conditional on. Copy the output of the decision-rule function as it is.
### 4.2 Full table (all methods × splits, seed mean ± SD or per-seed values)
### 4.3 Sensitivity (one factor at a time) — state that it was not used to choose the settings
### 4.4 Negative diagnostics and random-initialization control (initialization generated fresh for each seed)

## 5. Interpretation
- Separate what the data directly support, what is at the hypothesis level, and what cannot be said.
- State 2 or more possible causes for results that differed from what you expected, and say for each cause whether you checked it or not.
- If a simple baseline won, write that as it is.

## 6. Limitations and what was not done
- State the limits of generalization (1 dataset? model size? number of seeds? settings that were not tuned?)
- State **the list of what you could not run and what you could not verify** (e.g., cluster, other hardware, licenses, other datasets).

## 7. How to reproduce
- State the environment installation, the run order, the time required and where the output goes. Also state whether you confirmed that it runs to the end from a clean checkout.

## 8. Accountability
- State the license, the result of the dual-use review, and the decision on the scope of release and its reason.

## Appendix
- Attach the change log, the points raised in the independent review and how they were handled (accepted or rejected, with the grounds), and a summary of provenance.json.
```

<!-- i18n-source: 07_capstone/templates/report_template.md blob:89d32718f3a2df8bf52e7c110d7696ef5ad5078d -->
