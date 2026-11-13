# Module 03 Exercises

> **English translation.** AI-assisted translation of the [Korean original](exercises.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

No answers are provided. Check for yourself whether you meet the **evaluation criteria**, and record the results with the [experiment log template](../00_orientation/experiment_log_template.en.md). Report every metric together with repeats, seeds or confidence intervals.

## Exercise 1 — probe vs 6-mer baseline (module deliverable a)  *(notebook 01)*
Add `promoter_all` (or another NT task, e.g., `enhancers`), and change at least two of the following and report the results.
- Embedding layer choice (last layer vs an intermediate layer)
- Reduce the number of training samples to 1/4
- Evaluate with sequences shifted by 1–5 nt (changing only the start position) to see the **frame-shift sensitivity**. To separate the effect of the single-base token at the end, include a **control with the length made a multiple of 6**, as in section 6 of notebook 01.
- The change when evaluating with the reverse complement of the sequence
- Evaluation criteria: ① a table including the 6-mer baseline (with seeds and repeats), ② at least 2 hypotheses on the conditions under which the embedding beats the baseline and those under which it does not, ③ the result of checking the possibility of leakage in the data split (chromosomes, repeat regions), or a statement of the limitation.

## Exercise 2 — Directly computed LLR vs published scores (module deliverable b)  *(notebook 02)*
Change the NT LLR computation of notebook 02. (a) Window length 120 / 600 / 1,200 bp (a **multiple of 6** is recommended. The code works even when the length is not a multiple of 6, as with 128/512/1,024, but the end of the window becomes a single-base token, so report this effect separately), (b) with or without averaging over frame offsets, (c) with or without reverse-complement averaging, (d) LLR vs absLLR.
- Evaluation criteria: ① a table of AUPRC by setting and a **bootstrap confidence interval at the level of matched groups**, not variants, ② comparison with phyloP-241m, CADD and the published NT score on the same variant set, ③ a p-value against the **permutation null distribution made by shuffling each score's own values within the match_group** (`perm_null(y, groups, score)` in notebook 02) (do not judge "the same as random" by whether the bootstrap interval includes 0.10), ④ an interpretation that reflects the limitation that the sample (number of chr1 variants, number of positives) is small, ⑤ propose a design that separates whether what changes when the frame is shifted is the token frame or the context (window boundary) (e.g., changing only the 6-mer boundary in the same context).

## Exercise 3 — Do the matching variables really carry no information?  *(notebook 02)*
Make a score from TSS distance and the variant type (consequence) alone and compute the AUPRC.
- Evaluation criteria: ① the result compared with the random expected value, ② an interpretation based on the matching design, ③ if it differs from expectation, hypotheses on the cause (sample size, incompleteness of the matching, etc.).

## Exercise 4 — Enformer ISM  *(notebook 03)*
Choose another gene (look up the TSS coordinates yourself, citing the source), draw the predicted tracks, and run ISM at selected positions within about 100 bp around the TSS.
- Evaluation criteria: ① the grounds for the gene, coordinates and tracks used (identifiers), ② a quantitative check of whether the predicted tracks agree with known biology (CAGE signal around the TSS, etc.), ③ a description of the sequence features of the positions with large ISM effects, ④ the difference between the input range (196,608 bp) and the output range (central 896 bins). Explain why a variant outside the output interval can affect the central output, and why the track at that position cannot be directly predicted.

## Exercise 5 (optional, 🔴) — Run Evo 2 yourself
Check the requirements in the README of the official repository (CUDA 12.1+, `evo2_7b` is possible in bf16, `evo2_1b_base` needs FP8 and Hopper) and, in an environment where it is possible, directly compute the LLR for some of the TraitGym variants. No notebook is provided.
- Evaluation criteria: ① the model, environment and window length used, ② the correlation with the published Evo 2 scores of TraitGym, ③ hypotheses on the cause in case of disagreement.

## Exercise 6 — Paper note
Organize BEND (Marin 2024) **or** TraitGym (Benegas 2025) with the [paper note template](../00_orientation/paper_note_template.en.md).
- Evaluation criteria: ① fill in the split, baseline, metric and pretraining–evaluation overlap items of the checklist with grounds from the main text, ② summarize whether the authors' conclusions agree with or differ from the numbers in our notebooks.

## Exercise 7 (optional) — RNA
Read the RiNALMo or RNA-FM paper, and design yourself a split that evaluates "generalization to RNA families not seen before" (state the data source and the similarity criterion).
- Evaluation criteria: ① the split design and the similarity criterion, ② a check of the possibility of leakage, ③ the correspondence with the checklist of section 5 of module 01.

<!-- i18n-source: 03_dna_genome/exercises.md blob:bb95f49311eeaf7ae0eff553f90b8548fb68950d -->
