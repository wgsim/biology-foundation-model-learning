# Benchmark Map

> **English translation.** AI-assisted translation of the [Korean original](benchmarks.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

Written: 2026-11-06. A table organizing what the benchmarks often met in biology FM evaluation **ask, and what they cannot say**. The bibliographic and abstract-level statements were checked in the same way as `reading_list.md` (see its legend).

**What this document does not do**
- **It does not write leaderboard scores or rankings.** Scores change with the version, the aggregation method and the model list, and copying them makes them go stale quickly. Read them on the official site or in the paper, *after checking the version*.
- The values measured directly in this repository are in "What we saw in this data" in each module README. The "This repository" column below points to that location. Those values are results of small models, few seeds and one dataset, so they **cannot be compared with the official scores of the benchmarks**.
- The "what cannot be said" column of the tables is the author's judgment (not a claim of the paper).

## 7 questions before using a benchmark (a summary of module 06 §6)
1. What is the origin of the labels? Is it an experiment, a human annotation, or an annotation transferred from sequence similarity (a leakage path)?
2. What leakage does the split block, and what does it fail to block? **Did you measure the overlap in numbers after splitting?**
3. Is the metric suited to the task? Is it a single average, or is it by type and with uncertainty?
4. Are simple baselines (one that learns no parameters, simple features, random initialization) included with the same procedure?
5. Can the pretraining data and the evaluation data overlap? Did you check?
6. If you keep looking at the same test and choosing models, you get fitted to the test. Is there a hidden test/version pinning?
7. Are the data version and the license stated?

## Proteins
| Benchmark | What it asks | Data and metric (source statement) | Split/leakage cautions | What cannot be said | This repository | Source |
|---|---|---|---|---|---|---|
| **TAPE** | Do self-supervised protein embeddings transfer to downstream tasks? | **5** semi-supervised tasks from different areas of protein biology (abstract). The train/val/test split was fixed per task to look at "biologically meaningful generalization" (abstract) | The split differs per task. You must check the split definition you will use for each task | The ranking of modern large PLMs (a 2019 benchmark, centered on the models of that time) | Reading list 02 | arXiv:1906.08230 ✅ Web |
| **FLIP** | Can prediction of the **fitness landscape** for protein engineering work even in low-data and extrapolation conditions? | Experimental data on AAV stability, GB1 stability and immunoglobulin binding, and the thermostability of several families (abstract). Low-data and extrapolation generalization settings are deliberately included | The split defines "extrapolation", so **which split a score was read under** is itself the conclusion | Other proteins, other assay types | Reading list 02 | 10.1101/2021.11.09.467890 ✅ bioRxiv |
| **PEER** | Sequence-based protein understanding overall (function, localization, structure, interaction, ligand) and multi-task learning | 17 tasks in 5 categories (web-search summary): function prediction, subcellular localization, structure, protein–protein interaction, protein–ligand interaction (abstract). Compares feature engineering, sequence encoders and large PLMs, and also surveys multi-task learning | With many tasks, collapsing to a single average erases the per-task differences | DMS fitness prediction at the scale of ProteinGym, design (whether any of the PEER tasks are variant-data-based is to be checked in the paper's tables†) | Reading list 02 | arXiv:2206.02096 ✅ Web |
| **ProteinGym** | Variant effect (fitness) prediction and design | More than 250 standardized DMS assays (millions of variant sequences) and clinically annotated data, zero-shot and supervised settings, a comparison of more than 70 models (abstract). The v1.3 substitution file of this repository has 217 assays. Per-assay Spearman is used | If other substitutions at the same position are in train, a model that has learned the position effect has an advantage. The DMS target proteins may be in the pretraining corpus. The official aggregation is per-type aggregation, so it **differs from the simple average of the notebook** | The DMS fitness score alone cannot say anything about the ability of clinical pathogenicity prediction (the clinical subset is a separate evaluation target, and this repository did not use it. The goal differs) | Module 02 (7 assays), 06, 07 template | 10.1101/2023.12.07.570727 ✅ bioRxiv · Bibliography: the conference-version title is *…Protein Fitness Prediction and Design* (bioRxiv v1 is *…Protein Design and Fitness Prediction*) |

> Evaluation of structure prediction (CASP, CAMEO, etc.) and functional annotation (CAFA) were neither covered nor checked in this repository. If needed, check the official materials directly. ⚠️ 미검증 (unverified)

## DNA · genome
| Benchmark | What it asks | Data and metric (source statement) | Split/leakage cautions | What cannot be said | This repository | Source |
|---|---|---|---|---|---|---|
| **GUE** (DNABERT-2 paper) | Multi-species genome sequence classification | 36 datasets × 9 tasks, input length 70–10000 (abstract) | The split and the handling of repeated sequences differ by dataset, so check | Variant effects, long-range regulation | Module 03 (model table) | arXiv:2306.15006 ✅ Web |
| **Nucleotide Transformer downstream tasks** | Prediction of molecular phenotypes of the human genome (promoters, etc.) | The downstream tasks of the NT paper. This repository uses `promoter_tata` and `promoter_no_tata` (300 bp promoter classification, AUROC) of the `revised` version published on HF | Repeated/duplicated sequences can straddle train/test. Duplicates must be checked | Variant effects, cross-species generalization | Module 03 notebook 01 (including the 6-mer frequency baseline) | 10.1038/s41592-024-02523-z ✅ PubMed |
| **Genomic Benchmarks** | Easy reproduction of regulatory-element sequence classification | **9** datasets (promoters, enhancers, open chromatin), human, mouse and worm, including a simple CNN baseline, distributed as a Python package (abstract) | Centered on small binary classification, and not a hard long-range/variant task | Variant effects, long-range dependence | Not used | 10.1186/s12863-023-01123-8 ✅ PubMed |
| **BEND** | Do DNA LMs solve biologically meaningful tasks? | A collection of tasks defined on the human genome. Reports that the embeddings of current DNA LMs come close to expert methods on some tasks but capture long-range features only to a limited extent (abstract) | Per-task split, length, sparsity (emphasized by the abstract) | Variant effects | Module 03 §7 (reading) | arXiv:2311.12570 ✅ Web |
| **TraitGym** | Prioritization of causal non-coding variants | (Abstract) Binary classification of the causal-candidate regulatory variants of 113 Mendelian and 83 complex traits **with matched controls**. (Confirmed in module 03) The main metric is AUPRC, and the random baseline of the `matched_9` version, with positive:control 1:9, is ≈ 0.1 | Even if the controls are matched on variant type, TSS distance and so on, **measure the residual information of simple features** to check (module 03: about 0.115) | Coding variants, inter-individual expression variation | Module 03 notebook 02 (evaluation of the published scores + direct NT LLR computation) | 10.1101/2025.02.11.637758 ✅ bioRxiv |
| **Huang 2023 (personal transcriptome)** | Do supervised sequence-to-function models explain inter-individual expression variation? | An evaluation study — reports that current models do not explain inter-individual expression variation well | Between-gene differences of the reference genome ≠ the personal cis-variant effect | Performance on other tasks | Module 03 §6–7 | 10.1038/s41588-023-01574-w ✅ PubMed |

> The way to read sequence-to-function models (Enformer, Borzoi, AlphaGenome) by the *held-out evaluation of their own papers*, and the benchmarks of RNA models, were not covered in this map (⚠️ 미검증 (unverified)). The Huang 2023 row deals only with the explanatory power of those models for inter-individual expression.

## Single-cell
| Benchmark/evaluation | What it asks | Design (source statement) | Split/leakage cautions | What cannot be said | This repository | Source |
|---|---|---|---|---|---|---|
| **scib** (Luecken 2022) | Quantitative evaluation of data integration (batch correction) | A collection of metrics that measure batch mixing and biological conservation together (ASW, etc.) | Do not be fooled by a method that is high on only one metric. Check the consistency between metrics | The downstream predictive ability of a model | Module 04 notebook 02 (ASW_batch, etc.) | 10.1038/s41592-021-01336-8 ✅ Crossref |
| **Kedzierska 2025** | **Zero-shot** embeddings of single-cell FMs | Evaluates scGPT and Geneformer without fine-tuning, and reports that they can fall behind simple methods (abstract) | The evaluation data may be in the pretraining corpus | Fine-tuning and few-label settings | Module 04 §6 | 10.1186/s13059-025-03574-x ✅ PubMed |
| **Ahlmann-Eltze 2025** | **Perturbation effect prediction** | Compares 5 FMs and 2 deep-learning models with deliberately simple baselines, and reports that none exceeded the baseline in predicting the transcriptome change after single and double gene perturbations (abstract) | The definition of the split of the perturbation data (new perturbation vs new cell) | Other tasks (cell-type annotation, etc.) | Module 04 §6 (no notebook) | 10.1038/s41592-025-02772-6 ✅ PubMed |
| **BioLLM** | An integration and evaluation *framework* for single-cell FMs | Not a model but an evaluation tool | — | — | Reading list 04 | 10.1016/j.patter.2025.101326 ✅ PubMed |

> A continuously updated single-cell benchmark such as OpenProblems was not checked in this repository (⚠️ 미검증 (unverified)).

## Molecules · chemistry
| Benchmark | What it asks | Design (source statement) | Split/leakage cautions | What cannot be said | This repository | Source |
|---|---|---|---|---|---|---|
| **MoleculeNet** | A collection of molecular property prediction | Several property datasets (this repository uses 3 small ones: ESOL, Lipophilicity, BBBP) | Even with the scaffold split, a considerable share of the test molecules are very similar to train (module 05: 26% of the Lipophilicity test at Tanimoto ≥ 0.7). The original source and license of each dataset need checking | Generalization to a new chemical space | Module 05 notebook 01 | 10.1039/c7sc02664a ✅ Crossref |
| **TDC** (Therapeutics Data Commons) | ML data and tasks across all stages of therapeutic development | 22 learning tasks, 66 datasets (web-search summary). NeurIPS 2021 Datasets & Benchmarks. Reports that even the best algorithms show limits under distribution shift and so on | Check the split definition per task | — | Not used | arXiv:2102.09548 ✅ Web |
| **Praski 2025** (evaluation study) | Do pretrained molecular embeddings exceed ECFP? | 28 models, 25 classification datasets. Reports that, except for one model (CheMeleon), none exceeded the ECFP count fingerprint (abstract, preprint) | A preprint, depends on the settings | Regression, other data | Module 05 | arXiv:2508.06199 ✅ Web |

## Pathology images · LLM agents
| Evaluation | What it asks | Design (source statement) | Caution | This repository | Source |
|---|---|---|---|---|---|
| **de Jong 2025** (robustness of pathology FMs) | Are pathology FMs robust to differences between medical centers? | Reports that all 10 public pathology FMs strongly represent the medical-center signal (abstract, preprint) | Weights such as UNI/CONCH need access approval. This repository did not use them | Module 05 (reading) | arXiv:2501.18055 ✅ Web |
| **LAB-Bench** | The **research-practice** biology ability of LLMs | More than 2,400 multiple-choice questions (literature, figure interpretation, database navigation, DNA/protein sequence manipulation, etc.; abstract). A public subset is provided on HF | Multiple-choice accuracy ≠ experimental design ability. Distinguish agent evaluation from model evaluation | Module 05 (reading) | arXiv:2407.10362 ✅ Web |

## The data used in this repository at a glance
| Module | Data | Source format |
|---|---|---|
| 02·06·07 | ProteinGym v1.3 substitution file (7 assays / 1 DN7A) | ProteinGym distribution file |
| 03 | NT downstream `revised` promoter task, TraitGym public scores and variants | HF dataset |
| 04 | Wilk 2020 PBMC (CELLxGENE distribution, 13 donors) | CELLxGENE H5AD |
| 05 | 3 MoleculeNet CSVs, UniProt human Swiss-Prot release 2026_04 | Public CSV/TSV |

For the license grounds of each dataset (what was confirmed and what could not be), see the table of module 06 §7.

## When reading a new benchmark (a 3-minute check)
- [ ] You did not take at face value the abstract's "best on our data". Check whether the setting selection was done on the test set
- [ ] You wrote the split rule in one line, and wrote the paths that cannot be blocked
- [ ] The score table has a **simple-baseline row**. If not, add it yourself (module 07 checklist)
- [ ] You looked at both the average score and the per-type/per-task scores
- [ ] You pinned the version (the leaderboard of an official site may be updated)

<!-- i18n-source: resources/benchmarks.md blob:7a0bde1b16f74d520992b3e341bd871b61f71349 -->
