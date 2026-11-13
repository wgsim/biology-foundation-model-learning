# Reading List

> **English translation.** AI-assisted translation of the [Korean original](reading_list.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original. The tags in square brackets, such as [복습] and [비판], are kept in Korean as in the original, and are explained in the legend.

Written/updated: 2026-11-06 (v7: 2 items added for the benchmark map, full DOI/arXiv ID check) · 2026-11-06 (v6: module 07 reading list added) · 2026-11-05 (v5: module 06 reading list added · v4: module 05 reading list added) · 2026-11-04 (v3: reflected the Codex·agy cross-review + re-verified the unverified items)

## Legend
**Verification status**
- ✅ PubMed: confirmed from PubMed metadata (title, journal, year, DOI)
- ✅ bioRxiv: confirmed with the bioRxiv API (a preprint — may be a version before peer review)
- ✅ Web: title, authors and ID confirmed from web search results (arXiv, conference proceedings, etc., not indexed in PubMed)
- ✅ Crossref: title, journal, volume, pages and year confirmed with the Crossref API (used when the PubMed lookup temporarily failed; statements at the abstract level confirmed with the Semantic Scholar abstract)
- ⚠️ 미검증 (unverified): not confirmed. It must be checked directly before use.

**Importance** (separate from recency and nature; the assignment of importance is the author's judgment + the opinions of the review models): ★ required · ◆ advanced · ○ optional (for a track/interest)
**Tags**: [복습] review — a light read if you are at intermediate level or above · [최신] latest · [비판] critique — verifying limitations and overestimation · [기준선] baseline — pre-FM methods (comparison targets) · [벤치마크] benchmark · [지도]/[자기지도] supervised/self-supervised learning method · [실습] hands-on — directly tied to a notebook path

---

## 01 Foundations
| Importance | Paper | ID | Verification |
|---|---|---|---|
| ○ [복습] | Vaswani et al., *Attention Is All You Need* | arXiv:1706.03762 | ✅ Web |
| ○ [복습] | Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* | arXiv:1810.04805 | ✅ Web |
| ◆ | Kaplan et al., *Scaling Laws for Neural Language Models* | arXiv:2001.08361 | ✅ Web |
| ◆ | Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla) — read as a pair with Kaplan | arXiv:2203.15556 | ✅ Web |
| ◆ | Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (the review models' opinions split: optional advanced vs required → placed as advanced) | arXiv:2312.00752 | ✅ Web |
| ◆ | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP) — based on contrastive learning, the basis of CONCH and sequence–text alignment | arXiv:2103.00020 | ✅ Web |
| ★ [실습] | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) | arXiv:2106.09685 | ✅ Web |
| ★ [벤치마크] | Walsh et al., *DOME: recommendations for supervised machine learning validation in biology* (Nat Methods 2021;18:1122–1127) — required reading on leakage and split validation, common to all | 10.1038/s41592-021-01205-4 | ✅ PubMed |
| ◆ [실습] | Steinegger & Söding, *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets* (Nat Biotechnol 2017;35:1026–1028) — a tool for homology-clustering-based splits | 10.1038/nbt.3988 | ✅ PubMed |

## 02 Protein
| Importance | Paper | Source / DOI | Verification |
|---|---|---|---|
| ★ | Rives et al., *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences* (PNAS 2021) | 10.1073/pnas.2016239118 | ✅ PubMed |
| ★ | Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM-2/ESMFold, Science 2023) | 10.1126/science.ade2574 | ✅ PubMed |
| ★ | Meier et al., *Language models enable zero-shot prediction of the effects of mutations on protein function* (ESM-1v) — bioRxiv 2021 / NeurIPS 2021 (Advances in Neural Information Processing Systems 34) conference version | 10.1101/2021.07.09.450648 | ✅ bioRxiv · ✅ Web |
| ◆ [비교] | Elnaggar et al., *ProtTrans* (IEEE TPAMI 2022) — for comparing training and data choices with ESM | 10.1109/TPAMI.2021.3095381 | ✅ PubMed |
| ○ [최신] | EvolutionaryScale/Biohub, *ESM Cambrian (ESM C)* — released 2024-12. HF `biohub/ESMC-300M` · `ESMC-600M` (updated 2026-09, license notation `mit`+`other`), and `transformers` has ESMC documentation. **There is no peer-reviewed paper** (a technical blog and model cards). The weight license may differ by model and date, so the model card must be checked before use | evolutionaryscale.ai/blog/esm-cambrian · huggingface.co/biohub/ESMC-600M | ✅ Web (blog · HF card) |
| ○ [최신] | Candido et al., *Language Modeling Materializes a World Model of Protein Biology* (bioRxiv 2026-06; Biohub, corresponding author A. Rives) — the supporting preprint of ESMFold2. According to the model card, it predicts all biomolecules (including small molecules and nucleic acids), uses the MSA optionally, and the weights are about 27 GB. The performance claims are the authors' reports | 10.64898/2026.06.03.729735 | ✅ bioRxiv |
| ★ | Jumper et al., *Highly accurate protein structure prediction with AlphaFold* (Nature 2021) | 10.1038/s41586-021-03819-2 | ✅ PubMed |
| ★ | Abramson et al., *Accurate structure prediction of biomolecular interactions with AlphaFold 3* (Nature 2024) — read the **Addendum** (10.1038/s41586-024-08416-7, ✅ PubMed) together with the main text | 10.1038/s41586-024-07487-w | ✅ PubMed |
| ◆ | Hayes et al., *Simulating 500 million years of evolution with a language model* (ESM3, Science 2025) | 10.1126/science.ads0018 | ✅ PubMed |
| ★ (design track) | Dauparas et al., *Robust deep learning-based protein sequence design using ProteinMPNN* (Science 2022) | 10.1126/science.add2187 | ✅ PubMed |
| ◆ (design track) | Watson et al., *De novo design of protein structure and function with RFdiffusion* (Nature 2023) | 10.1038/s41586-023-06415-8 | ✅ PubMed |
| ◆ (design track) | Hsu et al., *Learning inverse folding from millions of predicted structures* (ESM-IF1, ICML 2022 / PMLR 162; bioRxiv) | 10.1101/2022.04.10.487779 | ✅ Web |
| ◆ | Su et al., *SaProt: Protein Language Modeling with Structure-aware Vocabulary* (ICLR 2024; Foldseek structure tokens + sequence) | 10.1101/2023.10.01.560349 | ✅ Web |
| ★ (variant track) | Cheng et al., *Accurate proteome-wide missense variant effect prediction with AlphaMissense* (Science 2023) | 10.1126/science.adg7492 | ✅ PubMed |
| ★ [벤치마크] | Notin et al., *ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design* — bioRxiv 2023 / NeurIPS 2023 Datasets & Benchmarks Track conference version (the title of bioRxiv v1 was *…for Protein Design and Fitness Prediction*) | 10.1101/2023.12.07.570727 | ✅ bioRxiv · ✅ Web |
| ◆ [기준선] | Frazer et al., *Disease variant prediction with deep generative models of evolutionary data* (EVE, Nature 2021;599:91–95) — an MSA-based comparison reference | 10.1038/s41586-021-04043-8 | ✅ PubMed |
| ◆ | Madani et al., *Large language models generate functional protein sequences across diverse families* (ProGen, Nat Biotechnol 2023;41:1099–1106) — autoregressive generation | 10.1038/s41587-022-01618-2 | ✅ PubMed |
| ○ | Nijkamp et al., *ProGen2: Exploring the boundaries of protein language models* (Cell Syst 2023;14:968–978) | 10.1016/j.cels.2023.10.002 | ✅ PubMed |
| ◆ [벤치마크] | Rao et al., *Evaluating Protein Transfer Learning with TAPE* (2019) | arXiv:1906.08230 | ✅ Web |
| ◆ [벤치마크] | Dallago et al., *FLIP: Benchmark tasks in fitness landscape inference for proteins* (bioRxiv 2021) — low-data and extrapolation generalization conditions | 10.1101/2021.11.09.467890 | ✅ bioRxiv |
| ○ [벤치마크] | Xu et al., *PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding* (NeurIPS 2022 Datasets & Benchmarks) | arXiv:2206.02096 | ✅ Web |
| ◆ [해석] | Rao et al., *Transformer protein language models are unsupervised structure learners* (ICLR 2021; bioRxiv 2020-12) — attention → contact | 10.1101/2020.12.15.422761 | ✅ bioRxiv · ✅ Web |

## 03 DNA · Genome
| Importance | Paper | Source / DOI | Verification |
|---|---|---|---|
| ★ [기준선][지도] | Zhou & Troyanskaya, *Predicting effects of noncoding variants with deep learning-based sequence model* (DeepSEA, Nat Methods 2015;12:931–934) | 10.1038/nmeth.3547 | ✅ PubMed |
| ○ [기준선][지도] | Chen et al., *A sequence-based global map of regulatory activity for deciphering human genetics* (Sei, Nat Genet 2022;54:940–949) | 10.1038/s41588-022-01102-2 | ✅ PubMed |
| ★ [자기지도] | Ji et al., *DNABERT* (Bioinformatics 2021) | 10.1093/bioinformatics/btab083 | ✅ PubMed |
| ◆ [자기지도] | Zhou et al., *DNABERT-2* (ICLR 2024; the GUE benchmark) | arXiv:2306.15006 | ✅ Web |
| ★ [자기지도] | Dalla-Torre et al., *Nucleotide Transformer* (Nat Methods 2025;22:287, online 2024-11) | 10.1038/s41592-024-02523-z | ✅ PubMed |
| ◆ [자기지도] | Nguyen et al., *HyenaDNA* (NeurIPS 2023) | arXiv:2306.15794 | ✅ Web |
| ◆ [자기지도] | Schiff et al., *Caduceus* (ICML 2024) | arXiv:2403.03234 | ✅ Web |
| ◆ [자기지도] | Benegas, Batra, Song, *DNA language models are powerful predictors of genome-wide variant effects* (GPN, PNAS 2023) — plant (Arabidopsis) centered | 10.1073/pnas.2311219120 | ✅ PubMed |
| ○ [자기지도] | Karollus et al., *Species-aware DNA language models capture regulatory elements and their evolution* (Genome Biol 2024;25:83) | 10.1186/s13059-024-03221-x | ✅ Web |
| ★ [자기지도] | Nguyen et al., *Sequence modeling and design from molecular to genome scale with Evo* (Science 2024) | 10.1126/science.ado9336 | ✅ PubMed |
| ◆ [최신][자기지도] | Brixi et al., *Genome modelling and design across all domains of life with Evo 2* (Nature 2026;652:1349–1361; preprint bioRxiv 10.1101/2025.02.18.638918) | 10.1038/s41586-026-10176-5 | ✅ PubMed |
| ★ [지도] | Avsec et al., *Effective gene expression prediction from sequence by integrating long-range interactions* (Enformer, Nat Methods 2021) | 10.1038/s41592-021-01252-x | ✅ PubMed |
| ◆ [지도] | Linder et al., *Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation* (Borzoi, Nat Genet 2025;57:949–961) | 10.1038/s41588-024-02053-6 | ✅ Web |
| ◆ [최신][지도] | Avsec et al., *Advancing regulatory variant effect prediction with AlphaGenome* (Nature 2026;649:1206–1218) | 10.1038/s41586-025-10014-0 | ✅ Web |
| ○ [최신] | Hingerl et al., *Flashzoi: an enhanced Borzoi for accelerated genomic analysis* (Bioinformatics 2025) | 10.1093/bioinformatics/btaf467 | ✅ PubMed |
| ★ [벤치마크][비판] | Marin et al., *BEND: Benchmarking DNA Language Models on biologically meaningful tasks* (ICLR 2024) | arXiv:2311.12570 | ✅ Web |
| ★ [비판] | Huang et al., *Personal transcriptome variation is poorly explained by current genomic deep learning models* (Nat Genet 2023;55:2056–2059) | 10.1038/s41588-023-01574-w | ✅ PubMed |
| ★ [벤치마크][비판] | Benegas, Eraslan, Song, *Benchmarking DNA Sequence Models for Causal Regulatory Variant Prediction in Human Genetics* (TraitGym, bioRxiv 2025-02; HF `songlab/TraitGym`, MIT) — matched controls, AUPRC, precomputed per-model scores published | 10.1101/2025.02.11.637758 | ✅ bioRxiv |
| ○ [벤치마크] | Grešová et al., *Genomic benchmarks: a collection of datasets for genomic sequence classification* (BMC Genom Data 2023;24:25) — 9 datasets for classifying regulatory elements (promoters, enhancers, open chromatin), human, mouse and worm, including a simple CNN baseline (abstract) | 10.1186/s12863-023-01123-8 | ✅ PubMed |
| ◆ [자기지도] | Benegas et al., *GPN-MSA: an alignment-based DNA language model for genome-wide variant effect prediction* (bioRxiv 2023-10) → *A DNA language model based on multispecies alignment predicts the effects of genome-wide variants* (Nat Biotechnol 2025;43:1960–1965, Brief Communication) | 10.1038/s41587-024-02511-w | ✅ Web |

> Discussion point: **Supervised** models such as Enformer and Borzoi and **self-supervised pretrained** models such as DNABERT, NT and Evo have different training objectives. You must not rank them in the same category by scale alone.

## 04 Single-cell
| Importance | Paper | Source / DOI | Verification |
|---|---|---|---|
| ★ [기준선] | Lopez et al., *Deep generative modeling for single-cell transcriptomics* (scVI, Nat Methods 2018;15:1053–1058) | 10.1038/s41592-018-0229-2 | ✅ PubMed |
| ○ [기준선][실습] | Korsunsky et al., *Fast, sensitive and accurate integration of single-cell data with Harmony* (Nat Methods 2019;16:1289–1296) — the batch-correction baseline of notebook 02 | 10.1038/s41592-019-0619-0 | ✅ Crossref |
| ○ [벤치마크][실습] | Luecken et al., *Benchmarking atlas-level data integration in single-cell genomics* (scib, Nat Methods 2022;19:41–50, online 2021) — the definitions of integration metrics such as ASW | 10.1038/s41592-021-01336-8 | ✅ Crossref |
| ○ [데이터][실습] | Wilk et al., *A single-cell atlas of the peripheral immune response in patients with severe COVID-19* (Nat Med 2020;26:1070–1076) — the PBMC data of the module 04 notebooks (the CELLxGENE distribution, 13 donors). It is 2020 data, so whether it is included in the pretraining corpora is unverified | 10.1038/s41591-020-0944-y | ✅ Crossref |
| ○ [모델 카드] | Geneformer model card (Hugging Face `ctheodoris/Geneformer`): V1-10M (≈30 million cells, input 2,048) / V2-104M·316M (≈100 million cells, input 4,096). The notebooks are pinned to commit `1f7fbae4` | huggingface.co/ctheodoris/Geneformer | ✅ Web |
| ○ | Yang et al., *scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data* (Nat Mach Intell 2022;4:852–866; bioRxiv 10.1101/2021.12.05.471261) — an early reference point. It is reported that a Matters Arising (a comment disputing it) exists from 2024-12, but its content is unverified | 10.1038/s42256-022-00534-z | ✅ Web |
| ★ | Theodoris et al., *Transfer learning enables predictions in network biology* (Geneformer, Nature 2023) | 10.1038/s41586-023-06139-9 | ✅ PubMed |
| ★ | Cui et al., *scGPT* (Nat Methods 2024) | 10.1038/s41592-024-02201-0 | ✅ PubMed |
| ◆ | Hao et al., *Large-scale foundation model on single-cell transcriptomics* (scFoundation, Nat Methods 2024) | 10.1038/s41592-024-02305-7 | ✅ PubMed |
| ◆ | Rosen et al., *Universal Cell Embeddings: A Foundation Model for Cell Biology* (bioRxiv 2023-11) — no evidence of journal publication was found in the search results (treated as a preprint). Be careful not to confuse it with the separate paper SATURN (Nat Methods 2024) | 10.1101/2023.11.28.568918 | ✅ Web |
| ◆ [기준선] | Roohani, Huang, Leskovec, *Predicting transcriptional outcomes of novel multigene perturbations with GEARS* (Nat Biotechnol 2024;42:927–935, online 2023-08) — an evaluation target of the perturbation-prediction critique paper | 10.1038/s41587-023-01905-6 | ✅ PubMed |
| ★ [비판][최신] | Kedzierska et al., *Zero-shot evaluation reveals limitations of single-cell foundation models* (Genome Biol 2025) | 10.1186/s13059-025-03574-x | ✅ PubMed |
| ★ [비판][최신] | Ahlmann-Eltze, Huber, Anders, *Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines* (Nat Methods 2025;22:1657–1661) | 10.1038/s41592-025-02772-6 | ✅ PubMed |
| ○ [프레임워크][최신] | Qiu et al., *BioLLM: A standardized framework for integrating and benchmarking single-cell foundation models* (Patterns 2025) — not a model but an evaluation and integration tool | 10.1016/j.patter.2025.101326 | ✅ PubMed |

> Discussion point: The critique papers are results on **specific tasks (zero-shot embeddings, perturbation prediction)**. You must not generalize the failure on one task into a failure of FMs as a whole. You should sort out under what conditions a simple baseline wins. You must check the donor/study/batch-level splits.

## 05 Multimodal · Other
| Importance | Paper | Source / DOI | Verification |
|---|---|---|---|
| ★ [단일모달(이미지)] | Chen et al., *Towards a general-purpose foundation model for computational pathology* (UNI, Nat Med 2024) | 10.1038/s41591-024-02857-3 | ✅ PubMed |
| ★ [멀티모달(비전-언어)] | Lu et al., *A visual-language foundation model for computational pathology* (CONCH, Nat Med 2024) | 10.1038/s41591-024-02856-4 | ✅ PubMed |
| ◆ | Zhou et al., *Uni-Mol: A Universal 3D Molecular Representation Learning Framework* (ICLR 2023; preprint ChemRxiv) — no DOI, OpenReview id 6K2RM6wVqKu | openreview.net/forum?id=6K2RM6wVqKu | ✅ Web |
| ○ [화학] | Ross et al., *Large-scale chemical language representations capture molecular structure and properties* (MoLFormer, Nat Mach Intell 2022;4:1256–1264; arXiv:2106.09553) | 10.1038/s42256-022-00580-7 | ✅ Web |
| ○ [화학] | Chithrananda, Grand, Ramsundar, *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction* (2020) — a 3rd-party summary judges that it did not yet exceed the existing techniques of the time | arXiv:2010.09885 | ✅ Web |
| ◆ | Cao & Gao, *Multi-omics single-cell data integration and regulatory inference with graph-linked embedding* (GLUE, Nat Biotechnol 2022;40:1458–1466) | 10.1038/s41587-022-01284-4 | ✅ PubMed |
| ○ [최신] | Wohlwend et al., *Boltz-1: Democratizing Biomolecular Interaction Modeling* (bioRxiv 2024-11) — open-source complex structure prediction | 10.1101/2024.11.19.624167 | ✅ bioRxiv |
| ○ [최신] | *Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction* (bioRxiv 2025-06) — an extension to binding affinity | 10.1101/2025.06.14.659707 | ✅ Web |
| ○ [최신] | Chai Discovery (Boitreaud et al.), *Chai-1: Decoding the molecular interactions of life* (bioRxiv 2024-10) — the descriptions of the code and weight licenses differ by source, so they must be checked before use | 10.1101/2024.10.10.615955 | ✅ Web |
| ○ [RNA] | Chen et al., *Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions* (RNA-FM; arXiv:2204.00300) | bioRxiv 10.1101/2022.08.06.503062 | ✅ Web |
| ○ [RNA] | Penić et al., *RiNALMo: general-purpose RNA language models can generalize well on structure prediction tasks* (Nat Commun 2025;16:5671; arXiv:2403.00043) | 10.1038/s41467-025-60872-5 | ✅ Web |
| ○ [RNA] | *Ribonanza: deep learning of RNA structure through dual crowdsourcing* (including RibonanzaNet, bioRxiv 2024-02; led by the Das lab; the journal version is unverified) | 10.1101/2024.02.24.581671 | ✅ Web |
| ○ [에이전트] | Boiko et al., *Autonomous chemical research with large language models* (Coscientist, Nature 2023;624:570–578) | 10.1038/s41586-023-06792-0 | ✅ PubMed |
| ○ [에이전트] | M. Bran et al., *Augmenting large language models with chemistry tools* (ChemCrow, Nat Mach Intell 2024;6:525–535) | 10.1038/s42256-024-00832-8 | ✅ PubMed |
| ○ [에이전트] | Huang et al., *Biomni: A General-Purpose Biomedical AI Agent* (bioRxiv 2025-06) | 10.1101/2025.05.30.656746 | ✅ Web |
| ○ [LLM] | Luo et al., *BioGPT: generative pre-trained transformer for biomedical text generation and mining* (Brief Bioinform 2022;23(6); arXiv:2210.10341) | 10.1093/bib/bbac409 | ✅ Web |

| ★ [벤치마크][분자] | Wu et al., *MoleculeNet: a benchmark for molecular machine learning* (Chem Sci 2018;9:513–530) — the data collection of notebook 05-01 | 10.1039/c7sc02664a | ✅ Crossref |
| ◆ [기준선][분자] | Rogers & Hahn, *Extended-Connectivity Fingerprints* (J Chem Inf Model 2010;50:742–754) — ECFP | 10.1021/ci100050t | ✅ Crossref |
| ○ [split][분자] | Bemis & Murcko, *The Properties of Known Drugs. 1. Molecular Frameworks* (J Med Chem 1996;39:2887–2893) — the definition of the scaffold | 10.1021/jm9602928 | ✅ Crossref |
| ○ [split][분자] | Sheridan, *Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction* (J Chem Inf Model 2013;53:783–790) | 10.1021/ci400084k | ✅ Crossref |
| ○ [벤치마크][분자] | Huang et al., *Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development* (NeurIPS 2021 Datasets and Benchmarks; 22 tasks and 66 datasets per a web-search summary — the abstract itself was not directly checked) | arXiv:2102.09548 | ✅ Web |
| ◆ [기준선][분자] | Yang et al., *Analyzing Learned Molecular Representations for Property Prediction* (J Chem Inf Model 2019;59:3370–3388; Chemprop) | 10.1021/acs.jcim.9b00237 | ✅ Crossref |
| ◆ [화학] | Ahmad, Simon, Chithrananda, Grand et al., *ChemBERTa-2: Towards Chemical Foundation Models* (2022; whether it was published in a journal is unverified) — MLM/MTR pretraining, 77M molecules | arXiv:2209.01712 | ✅ Web |
| ★ [비판][분자][최신] | Praski, Adamczyk, Czech, *Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning* (2025; preprint) — abstract: reports that, except for one model (CheMeleon), none exceeded the ECFP count fingerprint across 28 models and 25 classification datasets | arXiv:2508.06199 | ✅ Web |
| ◆ [비판][split][분자] | Guo, Hernandez-Hernandez, Ballester, *Scaffold Splits Overestimate Virtual Screening Performance* (2024; preprint) — reports that molecules with different scaffolds are also similar to each other, so the train–test similarity is still high | arXiv:2406.00873 | ✅ Web |
| ○ [비판][분자] | Jiang et al., *Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models* (J Cheminform 2021;13) | 10.1186/s13321-020-00479-8 | ✅ Crossref |
| ○ [데이터][분자] | Delaney, *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure* (J Chem Inf Comput Sci 2004;44:1000–1005) · Martins et al., *A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling* (J Chem Inf Model 2012;52:1686–1697) | 10.1021/ci034243x · 10.1021/ci300124c | ✅ Crossref |
| ◆ [멀티모달][비판] | Liang, Zhang, Kwon, Yeung, Zou, *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning* (NeurIPS 2022, Advances in Neural Information Processing Systems 35:17612–17625) — the modality gap | 10.52202/068431-1280 | ✅ Crossref |
| ◆ [멀티모달: 단백질–텍스트] | Xu, Yuan, Miret, Tang, *ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts* (2023; the conference of publication was not directly checked) | arXiv:2301.12040 | ✅ Web |
| ○ [멀티모달: 단백질–텍스트] | *ProTrek: Navigating the Protein Universe through Tri-Modal Contrastive Learning* (bioRxiv 2024; the authors and the journal version are unverified) | 10.1101/2024.05.30.596740 | ✅ Crossref |
| ○ [멀티모달: 세포–텍스트] | *Multimodal learning of transcriptomes and text enables interactive single-cell RNA-seq data exploration with natural language* (CellWhisperer, bioRxiv 2024; the journal version is unverified) | 10.1101/2024.10.15.618501 | ✅ Crossref |
| ◆ [비판][병리][최신] | de Jong, Marcus, Teuwen, *Current Pathology Foundation Models are unrobust to Medical Center Differences* (2025; preprint) — abstract: reports that all 10 public pathology FMs evaluated strongly represent the medical-center signal | arXiv:2501.18055 | ✅ Web |
| ○ [병리] | Vorontsov et al., *A foundation model for clinical-grade computational pathology and rare cancers detection* (Virchow, Nat Med 2024;30:2924–2935) · Xu et al., *A whole-slide foundation model for digital pathology from real-world data* (Prov-GigaPath, Nature 2024;630:181–188) | 10.1038/s41591-024-03141-0 · 10.1038/s41586-024-07441-w | ✅ Crossref |
| ○ [에이전트][벤치마크] | Laurent et al., *LAB-Bench: Measuring Capabilities of Language Models for Biology Research* (2024; preprint) — at least 2,400 research-practice multiple-choice questions | arXiv:2407.10362 | ✅ Web |

> The agent items deal not with model performance but with **tool use and agent evaluation**. They must be read separately.

## 06 Advanced — Adaptation · Interpretation · Efficiency · Benchmarks · Accountability
| Importance | Paper | Source / DOI | Verification |
|---|---|---|---|
| ◆ [최신][해석] | Simon & Zou, *InterPLM: discovering interpretable features in protein language models via sparse autoencoders* (ESM-2 SAE, Nat Methods 2025;22:2107–2117; preprint bioRxiv 10.1101/2024.11.14.623630 · arXiv:2412.12101) | 10.1038/s41592-025-02836-7 | ✅ PubMed |
| ◆ [효율화] | Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022, Advances in Neural Information Processing Systems 35:16344–16359; arXiv:2205.14135) | 10.52202/068431-1189 | ✅ Crossref |
| ★ [적응] | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021; abstract: freezes the pretrained weights and injects trainable low-rank decomposition matrices into each layer; reports a reduction of the trainable parameters to 1/10,000 and of the GPU memory to 1/3 on GPT-3 175B) | arXiv:2106.09685 | ✅ Web |
| ◆ [적응][비판] | Biderman et al., *LoRA Learns Less and Forgets Less* (2024; abstract: reports that, for LLMs in the programming and mathematics domains, LoRA with a standard low-rank setting falls well short of full fine-tuning but better preserves performance outside the target domain — a result in an **LLM setting**) | arXiv:2405.09673 | ✅ Web |
| ◆ [적응][단백질] | Sledzieski et al., *Democratizing protein language models with parameter-efficient fine-tuning* (PNAS 2024;121; preprint bioRxiv 10.1101/2023.11.09.566187) — only the title level was checked | 10.1073/pnas.2405840121 | ✅ Crossref |
| ◆ [해석] | Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features in Language Models* (2023; abstract: assumes superposition as the cause of polysemanticity and reports finding more interpretable and monosemantic features with an SAE) | arXiv:2309.08600 | ✅ Web |
| ◆ [해석] | Gao et al., *Scaling and evaluating sparse autoencoders* (2024; abstract: directly controlling sparsity with a k-sparse (TopK) autoencoder, a modification that reduces dead latents, scaling laws for dictionary size and sparsity, and metrics for evaluating feature quality) | arXiv:2406.04093 | ✅ Web |
| ◆ [효율화] | Micikevicius et al., *Mixed Precision Training* (2017; abstract: trains in half precision, using a single-precision copy of the weights and loss scaling) | arXiv:1710.03740 | ✅ Web |
| ○ [효율화][분산] | Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2019) · Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* (2023) — only the title level was checked | arXiv:1910.02054 · arXiv:2304.11277 | ✅ Web |
| ○ [벤치마크][비판] | Raji et al., *AI and the Everything in the Whole Wide World Benchmark* (2021) — only the title level was checked | arXiv:2111.15366 | ✅ Web |
| ★ [책임성] | Baker & Church, *Protein design meets biosecurity* (Science 2024;383:349) — only the title level was checked (a short piece) | 10.1126/science.ado1671 | ✅ Crossref |
| ◆ [책임성] | Sandbrink, *Artificial intelligence and biological misuse: Differentiating risks of language models and biological design tools* (2023; abstract: distinguishes that LLMs lower the barrier for non-experts, whereas biological design tools (BDTs) can expand the capabilities of skilled actors) | arXiv:2306.13952 | ✅ Web |

(For the required reading on validation and splits, see DOME·MMseqs2 in 01, and for benchmarks, TAPE/FLIP/PEER/BEND in 02 and 03.)

## 07 Capstone — Reproduction · Reporting · Documentation
| Importance | Paper | Source / DOI | Verification |
|---|---|---|---|
| ★ [재현] | Pineau et al., *Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)* (abstract: reproducibility = obtaining results similar to those reported in the paper with the same code and data (where possible); the publication venue was not directly checked) | arXiv:2003.12206 | ✅ Web |
| ◆ [문서화] | Mitchell et al., *Model Cards for Model Reporting* (abstract: proposes that released models be accompanied by documents containing the intended use cases and performance characteristics; the venue of publication was not directly checked) | arXiv:1810.03993 | ✅ Web |
| ◆ [문서화] | Gebru et al., *Datasheets for datasets* (Commun ACM 2021;64:86–92; abstract: proposes documenting a dataset's motivation, composition, collection process, recommended uses, etc.) | 10.1145/3458723 | ✅ Crossref |

(For the required reading on validation and splits, see DOME·MMseqs2 in 01, and for the critical evaluation papers, see the section of each module.)

---

## Remaining uncertainty (⚠️)
- ⚠️ ESM C: no peer-reviewed paper was confirmed (only a blog and model cards). The license may change.
- Module 07 reading list: for Pineau and Mitchell, only up to the arXiv abstract was checked (whether they were published in a journal/conference is unverified). For Gebru, the Crossref bibliography and the arXiv abstract were checked.
- Module 06 reading list: for Sledzieski, ZeRO, FSDP, Raji and Baker & Church, only the title and bibliography level was checked (abstracts unverified). LoRA, SAE, mixed precision, Biderman and Sandbrink were checked up to the arXiv abstract, and whether they were published in a conference/journal is unverified (for FlashAttention, publication in NeurIPS 2022 was confirmed with Crossref).
- ⚠️ UCE, Ribonanza, Boltz-1/2, Chai-1, Biomni, the ESMFold2 preprint: only the preprint status was confirmed. A journal version was not found (this does not mean there is none).
- ⚠️ scBERT: the content and conclusions of the Matters Arising (2024-12) are unverified.
- ⚠️ Uni-Mol: there is no DOI (cited by OpenReview).
- ⚠️ Chai-1: the posting date of the preprint differs by one day depending on the source (10/10 vs 10/11). It is cited by DOI.
- ⚠️ Module 05: for ChemBERTa-2, ProtST, Praski, Guo, de Jong and LAB-Bench, only the arXiv level (abstract) was checked and whether they were published in a journal/conference is unverified. For ProTrek and CellWhisperer, only the title and DOI were checked, and the authors and journal version are unverified. The weights of UNI/CONCH/UNI2-h, Virchow and Prov-GigaPath need access approval (gated) on Hugging Face (by the HF metadata `gated=auto`, and the license is listed as cc-by-nc-nd-4.0 for UNI/CONCH). The notebooks of this repository do not use them.

## Operating notes (reflecting the cross-review)
- Reading alone can hardly bring you to an expert level in 18 weeks. **1 deliverable per module**: one of (a) reproducing a simple baseline, (b) checking the effect of changing the split, (c) comparing frozen representations with fine-tuning.
- Leakage-check items per module: protein = homology, genome = duplicates and homologous regions, single-cell = donor·study·batch, molecules = scaffold and time split, common = overlap between the pretraining data and the evaluation data.
- Disagreements between the review models are noted side by side in the text (e.g., Mamba). Of the bibliographic claims of the review models, only those confirmed with tools were reflected.

<!-- i18n-source: resources/reading_list.md blob:de891f5e281c72a5d9f119e24534597c9faff1ec -->
