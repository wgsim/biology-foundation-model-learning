# Module 04 Exercises

> **English translation.** AI-assisted translation of the [Korean original](exercises.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

No answers are provided. Check for yourself whether you meet the **evaluation criteria**, and record the results with the [experiment log template](../00_orientation/experiment_log_template.en.md). Report every metric together with repeats, seeds or confidence intervals, and **write the design (comparison targets, split, metrics, tuning procedure) before seeing the results**.

## Exercise 1 — What does tokenization change?  *(notebook 01)*
Change one thing of `RankTokenizer` and quantify how the token sequences and the embeddings change.
- (a) **Turn off median normalization** (depth normalization only, `use_median=False`): look at how much the sets of the top 100 genes of the same cell overlap (Jaccard) and the rank correlation. How does the embedding-based classification (LODO macro-F1 in notebook 02) change?
- (b) **Separating the effect of vocabulary differences**: V1 and V2 differ in vocabulary, weights, architecture and pooling together, so you cannot say that the classification difference is due to the vocabulary. Instead, measure the change when the input is restricted to the *common genes* of the V1 and V2 vocabularies in the **same model** (e.g., V1). Mark the direct V1↔V2 comparison separately as a confounded comparison.
- (c) **Does `n_counts` affect the rank?**: **prove by an equation** whether dividing all the genes in a cell by the same positive constant changes the rank (hint: notebook 01 §3.1), and check whether the token sequences are really the same when it is replaced with `nCount_RNA`. If they are the same, what are the paths through which `n_counts` could affect the results (e.g., 0 and non-finite values, floating-point ties, a sum defined over a gene subset)?
- Evaluation criteria: ① a table showing the amount of change (at the token-sequence level) and, where applicable, the LODO macro-F1 (at the embedding level), ② at least 2 hypotheses on why the change is large and why it is small, ③ a description that reflects the limitation that "we could not check against the official package".

## Exercise 2 — Extending the input ablation  *(notebook 03)*
03 §7 runs 4 inputs on V1 (full / detected set only / top 256 / order randomization) and 2 on V2 (full / detected set only). Add what has not been done yet. They are **top-N truncation on V2 (N = 128 / 512 / 2,048) and order randomization** (implemented so that `<cls>`/`<eos>` are kept), and a comparison of mean pooling instead of `<cls>` for V1 and V2.
- Evaluation criteria: ① define the comparison targets and metrics **in advance** and state that settings were not chosen by the test metric, ② the **paired** difference from the full ranking and a donor-level bootstrap interval (state that it is a conditional interval), ③ state that conditions where the input is artificially changed (order randomization, an input with all counts changed to 1, etc.) **may lie outside the pretraining distribution**, and describe the limits of interpreting the performance drop, ④ the gain relative to the computation time.

## Exercise 3 — Reproduction on another dataset  *(notebook 02, download needed)*
From CELLxGENE Discover, choose another study with **6 or more donors, integer counts in `raw.X`, and cell-type labels** (a different tissue is better) and run the pipeline of notebook 02. Download after checking the size, source and license.
- Evaluation criteria: ① a data inspection table (confounding of donor, disease and platform, the distribution of total counts), ② a comparison table of Geneformer vs PCA·Harmony·scVI (LODO macro-F1 + donor-level interval + batch-mixing metrics), ③ an interpretation that separates the items **in the same direction as the result of notebook 02 from those in a different direction**, ④ a discussion of the possibility that the data is included in the pretraining corpus and its effect.

## Exercise 4 — Linear baseline for perturbation prediction: a design document  *(mainly reading, running is optional)*
Read Ahlmann-Eltze et al. (Nat Methods 2025) and write the following as a **design document**. (a) two simple baselines to compare (e.g., "no change", an additive model) and what each represents, (b) the evaluation metric and the split (unseen perturbations / unseen combinations), (c) the pre-specified criterion for what result would make you accept that "the FM exceeded the baseline". Running it (Colab, perturbation data needed) is optional.
- Evaluation criteria: ① explain why the baselines are *deliberately simple*, ② check whether the split tests generalization without leakage, ③ include in the conclusion a sentence that does not generalize the result of one task to FMs as a whole.

## Exercise 5 — Comparing evaluation designs  *(reading + notebook 02)*
Compare the zero-shot evaluation design of Kedzierska et al. (Genome Biol 2025) with the design of notebook 02. The comparison items are data, labels, split (donor/batch), metrics, baselines and whether there was tuning.
- Evaluation criteria: ① a table of 3 or more differences, ② the effect each difference could have on the conclusion, ③ the *one thing you find most suspicious* in the design of notebook 02 and a proposal for an experiment to verify it.

<!-- i18n-source: 04_single_cell/exercises.md blob:80409504454476fb6b7d24d8716e879724d9bb40 -->
