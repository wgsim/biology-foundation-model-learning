# Module 01 Exercises

> **English translation.** AI-assisted translation of the [Korean original](exercises.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

No answers are provided for any exercise. Check for yourself whether you meet the **evaluation criteria**. Record the results with the [experiment log template](../00_orientation/experiment_log_template.en.md).

## Exercise 1 — Tokenizer choice (notebook 01)
Increase the motif length of the synthetic corpus from 8 (the default) to 12 and change the BPE vocabulary size to 128 / 512 / 2048, then measure the compression rate (number of tokens / number of bases).
- There are three evaluation criteria. ① Provide a table of compression rates by vocabulary size. ② State the smallest vocabulary size at which the motif becomes a single token. **If there is a motif that does not become a single token within the given vocabulary range, record it as "not found in range"**, and state the reason as a hypothesis, based on the number of tokens when it is encoded alone and the longest token that absorbed the surrounding sequence. ③ Write 2–3 sentences on the effect that a tokenizer with a high compression rate has on SNV resolution.

## Exercise 2 — What MLM learns (notebook 02)
Train while changing the number of coupled pairs (12) and the noise ratio (0.1), and report how the masking accuracy at the coupled positions changes.
- Evaluation criteria: ① a result table for at least 3 settings (mean ± standard deviation over 3 or more seeds), ② the improvement over the baseline (the same **per-position mode** prediction as in the notebook), ③ whether the learned pairs show up in the attention analysis, and a hypothesis for the case where they do not.
- Advanced: reduce the number of training samples to 1,000 / 10,000 / 100,000 and plot the relationship between data size and performance.

## Exercise 3 — Effect of the split (notebook 03) → module deliverable (b)
Report how the performance gap between the random split and the family split changes when the within-family divergence is lowered or raised.
- Evaluation criteria: ① a table comparing the AUROC of the two splits (5 seeds), ② a graph of accuracy by bin of maximum identity, ③ a discussion of "in what situations is a random split less risky".

## Exercise 4 — Paper note
Read DOME (Walsh et al. 2021), organize it with the [paper note template](../00_orientation/paper_note_template.en.md), and then map the checklist of 5.6 onto the DOME items.
- Evaluation criterion: map at least 1 checklist item of this course to each of the four DOME areas, and state which items are missing.

## Exercise 5 (optional) — Real sequences
Download several proteins from UniProt (network required, check the license) and replace the clustering code of notebook 03 with MMseqs2 results.
- Evaluation criterion: record the threshold used and the resulting distributions (cluster sizes, maximum identity).


<!-- i18n-source: 01_foundations/exercises.md blob:2ba6cbeb37bfb2011b9144f8c5d2f6912a28e36d -->
