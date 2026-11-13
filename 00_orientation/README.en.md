# 00. Orientation — Study Method and Self-Assessment

> **English translation.** AI-assisted translation of the [Korean original](README.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

**Duration**: Week 0 (about 3–4 hours)
**Goal**: Decide how to run the 18-week course, and diagnose your current level to choose what to review.

---

## 1. How to use this course

### Weekly time budget (assuming 8–10 hours per week)
| Activity | Time | Content |
|---|---|---|
| Reading | 3h | Module lecture notes + 1–2 required (★) papers |
| Practice | 4h | Run the notebook → modify the code → exercises (`exercises.md`) |
| Write-up | 1–2h | Write paper notes and experiment logs, and work through the comprehension-check questions |
| Buffer | 1h | Places where you got stuck; advanced (◆/○) papers |

### Criteria for completing a module
- Work through the lecture note's "comprehension check" questions before looking at the answers.
- Run the notebook **as it is**, then **change one thing** in the exercises and record how the result changes.
- Produce 1 deliverable per module. The deliverable is one of (a) reproducing a simple baseline, (b) checking the effect of changing the split, or (c) comparing a frozen representation with fine-tuning.
- Write at least 1 paper note ([paper_note_template.en.md](paper_note_template.en.md)).

### Compute tiers
🟢 local Mac (CPU/MPS) · 🟡 Colab GPU · 🔴 institutional cluster. Environment setup is in [../SETUP.en.md](../SETUP.en.md).

---

## 2. Self-assessment
Work through the 10 ML questions and 10 biology questions in [self_assessment.en.md](self_assessment.en.md) with the **answers covered**.

| Result | Recommendation |
|---|---|
| 8 or more questions in each area | Skip the basics review for that area and go to module 01 |
| 5–7 questions | Review only the topics of the questions you missed, then continue |
| 4 or fewer | First spend 1–2 weeks on an introductory textbook or course for that area (this course assumes an intermediate level or above) |

---

## 3. How to read a biology FM paper (3-pass)

**Pass 1 (15 min) — What does it claim?**
- Read only the title, abstract, figures and conclusion.
- Summarize in one sentence: "A *[model]* trained on *[input]* with *[objective]* shows *[claim]* on *[task]*."

**Pass 2 (45 min) — What evidence does the claim rest on?**
- Read the methods, the evaluation design and the tables/figures, and fill in the checklist below.

**Pass 3 (reproduction level)** — Follow the key equations and the appendix, and run the code if possible.

### Critical-reading checklist (specific to biology FMs)
1. **Input/tokens**: What is a token? (single residue, k-mer, BPE, gene, etc.)
2. **Objective**: Is it MLM, autoregressive, contrastive learning or diffusion?
3. **Pretraining data**: its scope, how duplicates were removed, and when it was collected.
4. **Overlap between the evaluation data and the pretraining data**: Is there possible homology or duplication?
5. **Split method**: Is it random, or by homology/time/institution/batch?
6. **Baselines**: Are simple models (linear, k-mer, mean) and pre-FM methods (MSA-based, CNN, etc.) included?
7. **Evaluation setting**: Is it zero-shot, linear probe or fine-tuning? Was the comparison fair?
8. **Statistics**: Are repeated runs, confidence intervals and effect sizes reported?
9. **Experimental validation**: Were the predictions confirmed in the wet lab, or is everything in silico?
10. **Accessibility**: Whether the code, weights and data are released, and their licenses.
11. **Stated limitations**: What limitations do the authors state themselves?

> Tip: When an FM does not beat a baseline on a task (e.g., the single-cell critique papers), record it as "no benefit in this setting", not as "FMs are useless".

---

## 4. Reproduction principles
- **Baseline first**: Before scaling up the model, get a score with the simplest method (frequency, k-mer logistic regression).
- **Fix and record the split first**: Settle the split rule before you look at the data.
- **Record seeds and versions**: Write down the random seed, the package versions, and the checkpoint ID with the date you checked it.
- **Change one thing at a time**: In comparison experiments, change only one variable.
- **Record failures too**: Keep the experiments that did not work, and your hypotheses about their causes, in the log.

Templates: [paper_note_template.en.md](paper_note_template.en.md) · [experiment_log_template.en.md](experiment_log_template.en.md)

---

## 5. Note management (recommended)
- Keep your study notes in a separate folder or repository and accumulate them as per-module files (e.g., `notes/01_foundations.md`).
- This repository is for course material, so do not commit your personal experiment results or checkpoints (`data/`, `checkpoints/`, etc. are excluded in `.gitignore`).

## Next
→ [../01_foundations/](../01_foundations/)

<!-- i18n-source: 00_orientation/README.md blob:ee28cfcc1dfeaf0ca6d049b81a0016385f3e936c -->
