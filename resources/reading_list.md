# Reading List

검증 상태 표기 (작성일 2026-11-04)
- ✅ PubMed: PubMed 메타데이터(제목·저널·연도·DOI)로 확인
- ✅ bioRxiv: bioRxiv API로 확인 (프리프린트 — 동료심사 전 버전일 수 있음)
- ✅ Web: 웹 검색 결과에서 제목·ID 확인 (arXiv 등, PubMed 미색인)
- ⚠️ 미검증: 기억에 기반한 항목. 사용 전 직접 확인할 것

수준: ★ 필수 · ◆ 심화 · ✦ 최신/비판적 평가

---

## 01 Foundations — ML 기초 (⚠️ 미검증, arXiv ID 기억 기반)
| 수준 | 논문 | ID |
|---|---|---|
| ★ | Vaswani et al., *Attention Is All You Need* | arXiv:1706.03762 ⚠️ |
| ★ | Devlin et al., *BERT* | arXiv:1810.04805 ⚠️ |
| ◆ | Kaplan et al., *Scaling Laws for Neural Language Models* | arXiv:2001.08361 ⚠️ |
| ◆ | Gu & Dao, *Mamba* | arXiv:2312.00752 ⚠️ |
| ★ | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) | arXiv:2106.09685 ✅ Web |

## 02 Protein
| 수준 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ★ | Rives et al., *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences* (PNAS 2021) | 10.1073/pnas.2016239118 | ✅ PubMed |
| ★ | Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM-2/ESMFold, Science 2023) | 10.1126/science.ade2574 | ✅ PubMed |
| ★ | Meier et al., *Language models enable zero-shot prediction of the effects of mutations on protein function* (ESM-1v, bioRxiv 2021) | 10.1101/2021.07.09.450648 | ✅ bioRxiv |
| ◆ | Elnaggar et al., *ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning* (IEEE TPAMI 2022) | 10.1109/TPAMI.2021.3095381 | ✅ PubMed |
| ★ | Jumper et al., *Highly accurate protein structure prediction with AlphaFold* (Nature 2021) | 10.1038/s41586-021-03819-2 | ✅ PubMed |
| ★ | Abramson et al., *Accurate structure prediction of biomolecular interactions with AlphaFold 3* (Nature 2024) | 10.1038/s41586-024-07487-w | ✅ PubMed |
| ◆ | Abramson et al., *Addendum: AlphaFold 3* (Nature 2024) — 원 논문 읽을 때 함께 확인 | 10.1038/s41586-024-08416-7 | ✅ PubMed |
| ◆ | Hayes et al., *Simulating 500 million years of evolution with a language model* (ESM3, Science 2025) | 10.1126/science.ads0018 | ✅ PubMed |
| ◆ | Dauparas et al., *Robust deep learning-based protein sequence design using ProteinMPNN* (Science 2022) | 10.1126/science.add2187 | ✅ PubMed |
| ◆ | Watson et al., *De novo design of protein structure and function with RFdiffusion* (Nature 2023) | 10.1038/s41586-023-06415-8 | ✅ PubMed |
| ★ | Cheng et al., *Accurate proteome-wide missense variant effect prediction with AlphaMissense* (Science 2023) | 10.1126/science.adg7492 | ✅ PubMed |
| ★ | Notin et al., *ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction* (bioRxiv 2023) | 10.1101/2023.12.07.570727 | ✅ bioRxiv |

## 03 DNA · Genome
| 수준 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ★ | Ji et al., *DNABERT* (Bioinformatics 2021) | 10.1093/bioinformatics/btab083 | ✅ PubMed |
| ◆ | Zhou et al., *DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome* (ICLR 2024; GUE 벤치마크) | arXiv:2306.15006 | ✅ Web |
| ★ | Dalla-Torre et al., *Nucleotide Transformer: building and evaluating robust foundation models for human genomics* (Nat Methods, 온라인 2024-11, vol 22 p287) | 10.1038/s41592-024-02523-z | ✅ PubMed |
| ◆ | Nguyen et al., *HyenaDNA* (NeurIPS 2023) | arXiv:2306.15794 | ✅ Web |
| ◆ | Schiff et al., *Caduceus* (ICML 2024) | arXiv:2403.03234 | ✅ Web |
| ★ | Nguyen et al., *Sequence modeling and design from molecular to genome scale with Evo* (Science 2024) | 10.1126/science.ado9336 | ✅ PubMed |
| ✦ | Brixi et al., *Genome modeling and design across all domains of life with Evo 2* (bioRxiv 2025-02) — 저널 게재 여부 ⚠️ 미확인 | 10.1101/2025.02.18.638918 | ✅ bioRxiv |
| ★ | Avsec et al., *Effective gene expression prediction from sequence by integrating long-range interactions* (Enformer, Nat Methods 2021) | 10.1038/s41592-021-01252-x | ✅ PubMed |
| ◆ | Linder et al., *Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation* (Borzoi, Nat Genet 2025;57:949–961) | 10.1038/s41588-024-02053-6 | ✅ Web |
| ✦ | Avsec et al., *Advancing regulatory variant effect prediction with AlphaGenome* (Nature 2026;649:1206–1218) | 10.1038/s41586-025-10014-0 | ✅ Web |
| ✦ | Hingerl et al., *Flashzoi: an enhanced Borzoi for accelerated genomic analysis* (Bioinformatics 2025) | 10.1093/bioinformatics/btaf467 | ✅ PubMed |

## 04 Single-cell
| 수준 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ★ | Theodoris et al., *Transfer learning enables predictions in network biology* (Geneformer, Nature 2023) | 10.1038/s41586-023-06139-9 | ✅ PubMed |
| ★ | Cui et al., *scGPT: toward building a foundation model for single-cell multi-omics using generative AI* (Nat Methods 2024) | 10.1038/s41592-024-02201-0 | ✅ PubMed |
| ◆ | Hao et al., *Large-scale foundation model on single-cell transcriptomics* (scFoundation, Nat Methods 2024) | 10.1038/s41592-024-02305-7 | ✅ PubMed |
| ◆ | Rosen et al., *Universal Cell Embeddings: A Foundation Model for Cell Biology* (bioRxiv 2023-11; 심사 게재 여부 ⚠️ 미확인) | 10.1101/2023.11.28.568918 | ✅ Web |
| ✦ | Kedzierska et al., *Zero-shot evaluation reveals limitations of single-cell foundation models* (Genome Biol 2025) | 10.1186/s13059-025-03574-x | ✅ PubMed |
| ✦ | Ahlmann-Eltze, Huber, Anders, *Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines* (Nat Methods 2025;22:1657–1661) | 10.1038/s41592-025-02772-6 | ✅ PubMed |
| ✦ | Qiu et al., *BioLLM: A standardized framework for integrating and benchmarking single-cell foundation models* (Patterns 2025) | 10.1016/j.patter.2025.101326 | ✅ PubMed |

## 05 Multimodal · 기타
| 수준 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ★ | Chen et al., *Towards a general-purpose foundation model for computational pathology* (UNI, Nat Med 2024) | 10.1038/s41591-024-02857-3 | ✅ PubMed |
| ★ | Lu et al., *A visual-language foundation model for computational pathology* (CONCH, Nat Med 2024) | 10.1038/s41591-024-02856-4 | ✅ PubMed |

## 06 Advanced — 해석·효율화
| 수준 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ✦ | Simon & Zou, *InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders* (ESM-2 SAE) | bioRxiv 10.1101/2024.11.14.623630 · arXiv:2412.12101 | ✅ Web |

---

## 아직 목록에 없는 항목 (후속 작업)
- 분자/화학 FM, 생물학 LLM·에이전트, RNA FM, Boltz 계열, ESM C: 논문을 아직 확인하지 않아 포함하지 않음. 모듈 05 작성 시 검증 후 추가.
- 서열 상동성 split, 벤치마크 설계 관련 논문(모듈 01·06)도 모듈 작성 시 추가.
- 노트북에서 사용할 HF 체크포인트 ID와 라이브러리 API는 각 노트북 작성 시점에 context7·모델 허브로 확인.
