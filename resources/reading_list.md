# Reading List

작성/갱신: 2026-11-06 (v7: 벤치마크 지도용 항목 2개 추가, 전체 DOI·arXiv ID 일괄 점검) · 2026-11-06 (v6: 모듈 07 읽기 목록 추가) · 2026-11-05 (v5: 모듈 06 읽기 목록 추가 · v4: 모듈 05 읽기 목록 추가) · 2026-11-04 (v3: Codex·agy 교차 검토 반영 + 미검증 항목 재검증)

## 범례
**검증 상태**
- ✅ PubMed: PubMed 메타데이터(제목·저널·연도·DOI)로 확인
- ✅ bioRxiv: bioRxiv API로 확인 (프리프린트 — 동료심사 전 버전일 수 있음)
- ✅ Web: 웹 검색 결과에서 제목·저자·ID 확인 (arXiv·학회 논문집 등, PubMed 미색인)
- ✅ Crossref: Crossref API로 제목·저널·권·쪽·연도 확인 (PubMed 조회가 일시 실패했을 때 사용; 초록 수준 서술 확인은 Semantic Scholar 초록)
- ⚠️ 미검증: 확인되지 않음. 사용 전 직접 확인해야 한다.

**필수도** (최신성·성격과 분리, 필수도 배정은 작성자 판단 + 검토 모델 의견): ★ 필수 · ◆ 심화 · ○ 선택(트랙/관심 시)
**태그**: [복습] 중급 이상이면 가볍게 · [최신] · [비판] 한계·과대평가 검증 · [기준선] FM 이전 방법(비교 대상) · [벤치마크] · [지도]/[자기지도] 학습 방식 · [실습] 노트북 경로에 직결

---

## 01 Foundations
| 필수도 | 논문 | ID | 검증 |
|---|---|---|---|
| ○ [복습] | Vaswani et al., *Attention Is All You Need* | arXiv:1706.03762 | ✅ Web |
| ○ [복습] | Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* | arXiv:1810.04805 | ✅ Web |
| ◆ | Kaplan et al., *Scaling Laws for Neural Language Models* | arXiv:2001.08361 | ✅ Web |
| ◆ | Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla) — Kaplan과 짝으로 읽기 | arXiv:2203.15556 | ✅ Web |
| ◆ | Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (검토 모델 간 의견 갈림: 선택 심화 vs 필수 → 심화로 배치) | arXiv:2312.00752 | ✅ Web |
| ◆ | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP) — 대조학습 기반, CONCH·서열-텍스트 정렬의 기반 | arXiv:2103.00020 | ✅ Web |
| ★ [실습] | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) | arXiv:2106.09685 | ✅ Web |
| ★ [벤치마크] | Walsh et al., *DOME: recommendations for supervised machine learning validation in biology* (Nat Methods 2021;18:1122–1127) — 누출·분할 검증 공통 필독 | 10.1038/s41592-021-01205-4 | ✅ PubMed |
| ◆ [실습] | Steinegger & Söding, *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets* (Nat Biotechnol 2017;35:1026–1028) — 상동성 클러스터링 기반 split 도구 | 10.1038/nbt.3988 | ✅ PubMed |

## 02 Protein
| 필수도 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ★ | Rives et al., *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences* (PNAS 2021) | 10.1073/pnas.2016239118 | ✅ PubMed |
| ★ | Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM-2/ESMFold, Science 2023) | 10.1126/science.ade2574 | ✅ PubMed |
| ★ | Meier et al., *Language models enable zero-shot prediction of the effects of mutations on protein function* (ESM-1v) — bioRxiv 2021 / NeurIPS 2021(Advances in Neural Information Processing Systems 34) 학회판 | 10.1101/2021.07.09.450648 | ✅ bioRxiv · ✅ Web |
| ◆ [비교] | Elnaggar et al., *ProtTrans* (IEEE TPAMI 2022) — ESM과 학습·데이터 선택 비교용 | 10.1109/TPAMI.2021.3095381 | ✅ PubMed |
| ○ [최신] | EvolutionaryScale/Biohub, *ESM Cambrian (ESM C)* — 2024-12 공개. HF `biohub/ESMC-300M`·`ESMC-600M`(2026-09 갱신, 라이선스 표기 `mit`+`other`)이며 `transformers`에 ESMC 문서가 있다. **동료심사 논문은 없다**(기술 블로그·모델 카드). 가중치 라이선스는 모델·시점별로 달라질 수 있으므로 사용 전 모델 카드를 확인해야 한다 | evolutionaryscale.ai/blog/esm-cambrian · huggingface.co/biohub/ESMC-600M | ✅ Web (블로그·HF 카드) |
| ○ [최신] | Candido et al., *Language Modeling Materializes a World Model of Protein Biology* (bioRxiv 2026-06; Biohub, 교신 A. Rives) — ESMFold2의 근거 프리프린트다. 모델 카드에 따르면 모든 생체분자(소분자·핵산 포함)를 예측하고, MSA는 선택적으로 쓰며, 가중치는 약 27 GB다. 성능 주장은 저자 보고다 | 10.64898/2026.06.03.729735 | ✅ bioRxiv |
| ★ | Jumper et al., *Highly accurate protein structure prediction with AlphaFold* (Nature 2021) | 10.1038/s41586-021-03819-2 | ✅ PubMed |
| ★ | Abramson et al., *Accurate structure prediction of biomolecular interactions with AlphaFold 3* (Nature 2024) — **Addendum**(10.1038/s41586-024-08416-7, ✅ PubMed)를 본문과 함께 읽기 | 10.1038/s41586-024-07487-w | ✅ PubMed |
| ◆ | Hayes et al., *Simulating 500 million years of evolution with a language model* (ESM3, Science 2025) | 10.1126/science.ads0018 | ✅ PubMed |
| ★ (설계 트랙) | Dauparas et al., *Robust deep learning-based protein sequence design using ProteinMPNN* (Science 2022) | 10.1126/science.add2187 | ✅ PubMed |
| ◆ (설계 트랙) | Watson et al., *De novo design of protein structure and function with RFdiffusion* (Nature 2023) | 10.1038/s41586-023-06415-8 | ✅ PubMed |
| ◆ (설계 트랙) | Hsu et al., *Learning inverse folding from millions of predicted structures* (ESM-IF1, ICML 2022 / PMLR 162; bioRxiv) | 10.1101/2022.04.10.487779 | ✅ Web |
| ◆ | Su et al., *SaProt: Protein Language Modeling with Structure-aware Vocabulary* (ICLR 2024; Foldseek 구조 토큰 + 서열) | 10.1101/2023.10.01.560349 | ✅ Web |
| ★ (변이 트랙) | Cheng et al., *Accurate proteome-wide missense variant effect prediction with AlphaMissense* (Science 2023) | 10.1126/science.adg7492 | ✅ PubMed |
| ★ [벤치마크] | Notin et al., *ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design* — bioRxiv 2023 / NeurIPS 2023 Datasets & Benchmarks Track 학회판 (bioRxiv v1의 제목은 *…for Protein Design and Fitness Prediction*) | 10.1101/2023.12.07.570727 | ✅ bioRxiv · ✅ Web |
| ◆ [기준선] | Frazer et al., *Disease variant prediction with deep generative models of evolutionary data* (EVE, Nature 2021;599:91–95) — MSA 기반 비교 기준 | 10.1038/s41586-021-04043-8 | ✅ PubMed |
| ◆ | Madani et al., *Large language models generate functional protein sequences across diverse families* (ProGen, Nat Biotechnol 2023;41:1099–1106) — 자기회귀 생성 | 10.1038/s41587-022-01618-2 | ✅ PubMed |
| ○ | Nijkamp et al., *ProGen2: Exploring the boundaries of protein language models* (Cell Syst 2023;14:968–978) | 10.1016/j.cels.2023.10.002 | ✅ PubMed |
| ◆ [벤치마크] | Rao et al., *Evaluating Protein Transfer Learning with TAPE* (2019) | arXiv:1906.08230 | ✅ Web |
| ◆ [벤치마크] | Dallago et al., *FLIP: Benchmark tasks in fitness landscape inference for proteins* (bioRxiv 2021) — 저자료·외삽 일반화 조건 | 10.1101/2021.11.09.467890 | ✅ bioRxiv |
| ○ [벤치마크] | Xu et al., *PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding* (NeurIPS 2022 Datasets & Benchmarks) | arXiv:2206.02096 | ✅ Web |
| ◆ [해석] | Rao et al., *Transformer protein language models are unsupervised structure learners* (ICLR 2021; bioRxiv 2020-12) — attention → contact | 10.1101/2020.12.15.422761 | ✅ bioRxiv · ✅ Web |

## 03 DNA · Genome
| 필수도 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ★ [기준선][지도] | Zhou & Troyanskaya, *Predicting effects of noncoding variants with deep learning-based sequence model* (DeepSEA, Nat Methods 2015;12:931–934) | 10.1038/nmeth.3547 | ✅ PubMed |
| ○ [기준선][지도] | Chen et al., *A sequence-based global map of regulatory activity for deciphering human genetics* (Sei, Nat Genet 2022;54:940–949) | 10.1038/s41588-022-01102-2 | ✅ PubMed |
| ★ [자기지도] | Ji et al., *DNABERT* (Bioinformatics 2021) | 10.1093/bioinformatics/btab083 | ✅ PubMed |
| ◆ [자기지도] | Zhou et al., *DNABERT-2* (ICLR 2024; GUE 벤치마크) | arXiv:2306.15006 | ✅ Web |
| ★ [자기지도] | Dalla-Torre et al., *Nucleotide Transformer* (Nat Methods 2025;22:287, online 2024-11) | 10.1038/s41592-024-02523-z | ✅ PubMed |
| ◆ [자기지도] | Nguyen et al., *HyenaDNA* (NeurIPS 2023) | arXiv:2306.15794 | ✅ Web |
| ◆ [자기지도] | Schiff et al., *Caduceus* (ICML 2024) | arXiv:2403.03234 | ✅ Web |
| ◆ [자기지도] | Benegas, Batra, Song, *DNA language models are powerful predictors of genome-wide variant effects* (GPN, PNAS 2023) — 식물(Arabidopsis) 중심 | 10.1073/pnas.2311219120 | ✅ PubMed |
| ○ [자기지도] | Karollus et al., *Species-aware DNA language models capture regulatory elements and their evolution* (Genome Biol 2024;25:83) | 10.1186/s13059-024-03221-x | ✅ Web |
| ★ [자기지도] | Nguyen et al., *Sequence modeling and design from molecular to genome scale with Evo* (Science 2024) | 10.1126/science.ado9336 | ✅ PubMed |
| ◆ [최신][자기지도] | Brixi et al., *Genome modelling and design across all domains of life with Evo 2* (Nature 2026;652:1349–1361; 사전판 bioRxiv 10.1101/2025.02.18.638918) | 10.1038/s41586-026-10176-5 | ✅ PubMed |
| ★ [지도] | Avsec et al., *Effective gene expression prediction from sequence by integrating long-range interactions* (Enformer, Nat Methods 2021) | 10.1038/s41592-021-01252-x | ✅ PubMed |
| ◆ [지도] | Linder et al., *Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation* (Borzoi, Nat Genet 2025;57:949–961) | 10.1038/s41588-024-02053-6 | ✅ Web |
| ◆ [최신][지도] | Avsec et al., *Advancing regulatory variant effect prediction with AlphaGenome* (Nature 2026;649:1206–1218) | 10.1038/s41586-025-10014-0 | ✅ Web |
| ○ [최신] | Hingerl et al., *Flashzoi: an enhanced Borzoi for accelerated genomic analysis* (Bioinformatics 2025) | 10.1093/bioinformatics/btaf467 | ✅ PubMed |
| ★ [벤치마크][비판] | Marin et al., *BEND: Benchmarking DNA Language Models on biologically meaningful tasks* (ICLR 2024) | arXiv:2311.12570 | ✅ Web |
| ★ [비판] | Huang et al., *Personal transcriptome variation is poorly explained by current genomic deep learning models* (Nat Genet 2023;55:2056–2059) | 10.1038/s41588-023-01574-w | ✅ PubMed |
| ★ [벤치마크][비판] | Benegas, Eraslan, Song, *Benchmarking DNA Sequence Models for Causal Regulatory Variant Prediction in Human Genetics* (TraitGym, bioRxiv 2025-02; HF `songlab/TraitGym`, MIT) — 매칭된 대조군, AUPRC, 모델별 사전 계산 점수 공개 | 10.1101/2025.02.11.637758 | ✅ bioRxiv |
| ○ [벤치마크] | Grešová et al., *Genomic benchmarks: a collection of datasets for genomic sequence classification* (BMC Genom Data 2023;24:25) — 조절 요소(프로모터·인핸서·열린 염색질) 분류 9개 데이터셋, 사람·생쥐·선충, 간단한 CNN baseline 포함(초록) | 10.1186/s12863-023-01123-8 | ✅ PubMed |
| ◆ [자기지도] | Benegas et al., *GPN-MSA: an alignment-based DNA language model for genome-wide variant effect prediction* (bioRxiv 2023-10) → *A DNA language model based on multispecies alignment predicts the effects of genome-wide variants* (Nat Biotechnol 2025;43:1960–1965, Brief Communication) | 10.1038/s41587-024-02511-w | ✅ Web |

> 토론 포인트: Enformer·Borzoi 같은 **지도학습** 모델과 DNABERT·NT·Evo 같은 **자기지도 사전학습** 모델은 학습 목적이 다르다. 규모만으로 같은 범주에서 순위를 매겨서는 안 된다.

## 04 Single-cell
| 필수도 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ★ [기준선] | Lopez et al., *Deep generative modeling for single-cell transcriptomics* (scVI, Nat Methods 2018;15:1053–1058) | 10.1038/s41592-018-0229-2 | ✅ PubMed |
| ○ [기준선][실습] | Korsunsky et al., *Fast, sensitive and accurate integration of single-cell data with Harmony* (Nat Methods 2019;16:1289–1296) — 노트북 02의 배치 보정 baseline | 10.1038/s41592-019-0619-0 | ✅ Crossref |
| ○ [벤치마크][실습] | Luecken et al., *Benchmarking atlas-level data integration in single-cell genomics* (scib, Nat Methods 2022;19:41–50, online 2021) — ASW 등 통합 지표의 정의 | 10.1038/s41592-021-01336-8 | ✅ Crossref |
| ○ [데이터][실습] | Wilk et al., *A single-cell atlas of the peripheral immune response in patients with severe COVID-19* (Nat Med 2020;26:1070–1076) — 모듈 04 노트북의 PBMC 데이터(CELLxGENE 배포판, 13명). 2020 데이터라 사전학습 코퍼스에 포함되었는지는 미확인이다 | 10.1038/s41591-020-0944-y | ✅ Crossref |
| ○ [모델 카드] | Geneformer 모델 카드 (Hugging Face `ctheodoris/Geneformer`): V1-10M(≈3천만 세포, 입력 2,048) / V2-104M·316M(≈1억 세포, 입력 4,096). 노트북은 커밋 `1f7fbae4`로 고정했다 | huggingface.co/ctheodoris/Geneformer | ✅ Web |
| ○ | Yang et al., *scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data* (Nat Mach Intell 2022;4:852–866; bioRxiv 10.1101/2021.12.05.471261) — 초기 기준점이다. 2024-12 Matters Arising(이의 논평)이 존재한다고 보고되었으나 내용은 미확인이다 | 10.1038/s42256-022-00534-z | ✅ Web |
| ★ | Theodoris et al., *Transfer learning enables predictions in network biology* (Geneformer, Nature 2023) | 10.1038/s41586-023-06139-9 | ✅ PubMed |
| ★ | Cui et al., *scGPT* (Nat Methods 2024) | 10.1038/s41592-024-02201-0 | ✅ PubMed |
| ◆ | Hao et al., *Large-scale foundation model on single-cell transcriptomics* (scFoundation, Nat Methods 2024) | 10.1038/s41592-024-02305-7 | ✅ PubMed |
| ◆ | Rosen et al., *Universal Cell Embeddings: A Foundation Model for Cell Biology* (bioRxiv 2023-11) — 검색 결과에서 저널 게재 근거를 찾지 못했다(프리프린트로 취급). 별개 논문 SATURN(Nat Methods 2024)과 혼동하지 않도록 주의한다 | 10.1101/2023.11.28.568918 | ✅ Web |
| ◆ [기준선] | Roohani, Huang, Leskovec, *Predicting transcriptional outcomes of novel multigene perturbations with GEARS* (Nat Biotechnol 2024;42:927–935, online 2023-08) — 섭동 예측 비판 논문의 평가 대상 | 10.1038/s41587-023-01905-6 | ✅ PubMed |
| ★ [비판][최신] | Kedzierska et al., *Zero-shot evaluation reveals limitations of single-cell foundation models* (Genome Biol 2025) | 10.1186/s13059-025-03574-x | ✅ PubMed |
| ★ [비판][최신] | Ahlmann-Eltze, Huber, Anders, *Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines* (Nat Methods 2025;22:1657–1661) | 10.1038/s41592-025-02772-6 | ✅ PubMed |
| ○ [프레임워크][최신] | Qiu et al., *BioLLM: A standardized framework for integrating and benchmarking single-cell foundation models* (Patterns 2025) — 모델이 아니라 평가·통합 도구 | 10.1016/j.patter.2025.101326 | ✅ PubMed |

> 토론 포인트: 비판 논문은 **특정 과제(zero-shot 임베딩, 섭동 예측)**에서의 결과다. 한 과제의 실패를 FM 전체의 실패로 일반화해서는 안 된다. 어떤 조건에서 단순 baseline이 이기는지 정리해야 한다. donor·study·batch 단위 분할을 확인해야 한다.

## 05 Multimodal · 기타
| 필수도 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ★ [단일모달(이미지)] | Chen et al., *Towards a general-purpose foundation model for computational pathology* (UNI, Nat Med 2024) | 10.1038/s41591-024-02857-3 | ✅ PubMed |
| ★ [멀티모달(비전-언어)] | Lu et al., *A visual-language foundation model for computational pathology* (CONCH, Nat Med 2024) | 10.1038/s41591-024-02856-4 | ✅ PubMed |
| ◆ | Zhou et al., *Uni-Mol: A Universal 3D Molecular Representation Learning Framework* (ICLR 2023; 사전판 ChemRxiv) — DOI 없음, OpenReview id 6K2RM6wVqKu | openreview.net/forum?id=6K2RM6wVqKu | ✅ Web |
| ○ [화학] | Ross et al., *Large-scale chemical language representations capture molecular structure and properties* (MoLFormer, Nat Mach Intell 2022;4:1256–1264; arXiv:2106.09553) | 10.1038/s42256-022-00580-7 | ✅ Web |
| ○ [화학] | Chithrananda, Grand, Ramsundar, *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction* (2020) — 제3자 요약은 당시 기존 기법을 아직 넘지 못한다는 평가 | arXiv:2010.09885 | ✅ Web |
| ◆ | Cao & Gao, *Multi-omics single-cell data integration and regulatory inference with graph-linked embedding* (GLUE, Nat Biotechnol 2022;40:1458–1466) | 10.1038/s41587-022-01284-4 | ✅ PubMed |
| ○ [최신] | Wohlwend et al., *Boltz-1: Democratizing Biomolecular Interaction Modeling* (bioRxiv 2024-11) — 오픈소스 복합체 구조 예측 | 10.1101/2024.11.19.624167 | ✅ bioRxiv |
| ○ [최신] | *Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction* (bioRxiv 2025-06) — 결합 친화도 확장 | 10.1101/2025.06.14.659707 | ✅ Web |
| ○ [최신] | Chai Discovery (Boitreaud et al.), *Chai-1: Decoding the molecular interactions of life* (bioRxiv 2024-10) — 코드·가중치 라이선스 설명이 출처마다 달라 사용 전에 확인해야 한다 | 10.1101/2024.10.10.615955 | ✅ Web |
| ○ [RNA] | Chen et al., *Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions* (RNA-FM; arXiv:2204.00300) | bioRxiv 10.1101/2022.08.06.503062 | ✅ Web |
| ○ [RNA] | Penić et al., *RiNALMo: general-purpose RNA language models can generalize well on structure prediction tasks* (Nat Commun 2025;16:5671; arXiv:2403.00043) | 10.1038/s41467-025-60872-5 | ✅ Web |
| ○ [RNA] | *Ribonanza: deep learning of RNA structure through dual crowdsourcing* (RibonanzaNet 포함, bioRxiv 2024-02; Das lab 주도; 저널 게재판 미확인) | 10.1101/2024.02.24.581671 | ✅ Web |
| ○ [에이전트] | Boiko et al., *Autonomous chemical research with large language models* (Coscientist, Nature 2023;624:570–578) | 10.1038/s41586-023-06792-0 | ✅ PubMed |
| ○ [에이전트] | M. Bran et al., *Augmenting large language models with chemistry tools* (ChemCrow, Nat Mach Intell 2024;6:525–535) | 10.1038/s42256-024-00832-8 | ✅ PubMed |
| ○ [에이전트] | Huang et al., *Biomni: A General-Purpose Biomedical AI Agent* (bioRxiv 2025-06) | 10.1101/2025.05.30.656746 | ✅ Web |
| ○ [LLM] | Luo et al., *BioGPT: generative pre-trained transformer for biomedical text generation and mining* (Brief Bioinform 2022;23(6); arXiv:2210.10341) | 10.1093/bib/bbac409 | ✅ Web |

| ★ [벤치마크][분자] | Wu et al., *MoleculeNet: a benchmark for molecular machine learning* (Chem Sci 2018;9:513–530) — 노트북 05-01의 데이터 모음 | 10.1039/c7sc02664a | ✅ Crossref |
| ◆ [기준선][분자] | Rogers & Hahn, *Extended-Connectivity Fingerprints* (J Chem Inf Model 2010;50:742–754) — ECFP | 10.1021/ci100050t | ✅ Crossref |
| ○ [split][분자] | Bemis & Murcko, *The Properties of Known Drugs. 1. Molecular Frameworks* (J Med Chem 1996;39:2887–2893) — scaffold 정의 | 10.1021/jm9602928 | ✅ Crossref |
| ○ [split][분자] | Sheridan, *Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction* (J Chem Inf Model 2013;53:783–790) | 10.1021/ci400084k | ✅ Crossref |
| ○ [벤치마크][분자] | Huang et al., *Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development* (NeurIPS 2021 Datasets and Benchmarks; 웹 검색 요약 기준 22개 과제·66개 데이터셋 — 초록 원문은 직접 확인 안 됨) | arXiv:2102.09548 | ✅ Web |
| ◆ [기준선][분자] | Yang et al., *Analyzing Learned Molecular Representations for Property Prediction* (J Chem Inf Model 2019;59:3370–3388; Chemprop) | 10.1021/acs.jcim.9b00237 | ✅ Crossref |
| ◆ [화학] | Ahmad, Simon, Chithrananda, Grand et al., *ChemBERTa-2: Towards Chemical Foundation Models* (2022; 학술지 게재 여부 미확인) — MLM/MTR 사전학습, 77M 분자 | arXiv:2209.01712 | ✅ Web |
| ★ [비판][분자][최신] | Praski, Adamczyk, Czech, *Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning* (2025; 프리프린트) — 초록: 28개 모델·25개 분류 데이터셋에서 한 모델(CheMeleon)을 제외하고는 ECFP count 지문을 넘지 못했다고 보고 | arXiv:2508.06199 | ✅ Web |
| ◆ [비판][split][분자] | Guo, Hernandez-Hernandez, Ballester, *Scaffold Splits Overestimate Virtual Screening Performance* (2024; 프리프린트) — scaffold가 다른 분자도 서로 비슷해 train·test 유사도가 여전히 높다고 보고 | arXiv:2406.00873 | ✅ Web |
| ○ [비판][분자] | Jiang et al., *Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models* (J Cheminform 2021;13) | 10.1186/s13321-020-00479-8 | ✅ Crossref |
| ○ [데이터][분자] | Delaney, *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure* (J Chem Inf Comput Sci 2004;44:1000–1005) · Martins et al., *A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling* (J Chem Inf Model 2012;52:1686–1697) | 10.1021/ci034243x · 10.1021/ci300124c | ✅ Crossref |
| ◆ [멀티모달][비판] | Liang, Zhang, Kwon, Yeung, Zou, *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning* (NeurIPS 2022, Advances in Neural Information Processing Systems 35:17612–17625) — modality gap | 10.52202/068431-1280 | ✅ Crossref |
| ◆ [멀티모달: 단백질–텍스트] | Xu, Yuan, Miret, Tang, *ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts* (2023; 학회 게재처는 직접 확인 안 됨) | arXiv:2301.12040 | ✅ Web |
| ○ [멀티모달: 단백질–텍스트] | *ProTrek: Navigating the Protein Universe through Tri-Modal Contrastive Learning* (bioRxiv 2024; 저자·저널판 미확인) | 10.1101/2024.05.30.596740 | ✅ Crossref |
| ○ [멀티모달: 세포–텍스트] | *Multimodal learning of transcriptomes and text enables interactive single-cell RNA-seq data exploration with natural language* (CellWhisperer, bioRxiv 2024; 저널판 미확인) | 10.1101/2024.10.15.618501 | ✅ Crossref |
| ◆ [비판][병리][최신] | de Jong, Marcus, Teuwen, *Current Pathology Foundation Models are unrobust to Medical Center Differences* (2025; 프리프린트) — 초록: 평가한 공개 병리 FM 10개가 모두 의료기관 신호를 강하게 표현했다고 보고 | arXiv:2501.18055 | ✅ Web |
| ○ [병리] | Vorontsov et al., *A foundation model for clinical-grade computational pathology and rare cancers detection* (Virchow, Nat Med 2024;30:2924–2935) · Xu et al., *A whole-slide foundation model for digital pathology from real-world data* (Prov-GigaPath, Nature 2024;630:181–188) | 10.1038/s41591-024-03141-0 · 10.1038/s41586-024-07441-w | ✅ Crossref |
| ○ [에이전트][벤치마크] | Laurent et al., *LAB-Bench: Measuring Capabilities of Language Models for Biology Research* (2024; 프리프린트) — 연구 실무형 객관식 2,400문항 이상 | arXiv:2407.10362 | ✅ Web |

> 에이전트 항목은 모델 성능이 아니라 **도구 사용·에이전트 평가**를 다룬다. 구분해서 읽어야 한다.

## 06 Advanced — 적응 · 해석 · 효율화 · 벤치마크 · 책임성
| 필수도 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ◆ [최신][해석] | Simon & Zou, *InterPLM: discovering interpretable features in protein language models via sparse autoencoders* (ESM-2 SAE, Nat Methods 2025;22:2107–2117; 사전판 bioRxiv 10.1101/2024.11.14.623630 · arXiv:2412.12101) | 10.1038/s41592-025-02836-7 | ✅ PubMed |
| ◆ [효율화] | Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022, Advances in Neural Information Processing Systems 35:16344–16359; arXiv:2205.14135) | 10.52202/068431-1189 | ✅ Crossref |
| ★ [적응] | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021; 초록: 사전학습 가중치를 고정하고 각 층에 학습 가능한 저랭크 분해 행렬을 주입, GPT-3 175B에서 학습 매개변수 1만 분의 1·GPU 메모리 3분의 1 보고) | arXiv:2106.09685 | ✅ Web |
| ◆ [적응][비판] | Biderman et al., *LoRA Learns Less and Forgets Less* (2024; 초록: 프로그래밍·수학 도메인의 LLM에서 표준 저랭크 설정의 LoRA가 전체 파인튜닝보다 크게 못 미치지만 대상 도메인 밖 성능은 더 잘 보존한다고 보고 — **LLM 설정**의 결과) | arXiv:2405.09673 | ✅ Web |
| ◆ [적응][단백질] | Sledzieski et al., *Democratizing protein language models with parameter-efficient fine-tuning* (PNAS 2024;121; 사전판 bioRxiv 10.1101/2023.11.09.566187) — 제목 수준만 확인 | 10.1073/pnas.2405840121 | ✅ Crossref |
| ◆ [해석] | Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features in Language Models* (2023; 초록: 다의성(polysemanticity)의 원인으로 superposition을 가정하고 SAE로 더 해석 가능하고 단의적인 특징을 찾는다고 보고) | arXiv:2309.08600 | ✅ Web |
| ◆ [해석] | Gao et al., *Scaling and evaluating sparse autoencoders* (2024; 초록: k-sparse(TopK) 오토인코더로 희소도를 직접 제어, 죽은 잠재 변수를 줄이는 수정, 사전 크기·희소도에 대한 스케일링 법칙, 특징 품질 평가 지표) | arXiv:2406.04093 | ✅ Web |
| ◆ [효율화] | Micikevicius et al., *Mixed Precision Training* (2017; 초록: 반정밀도로 학습하되 단정밀도 가중치 사본과 손실 스케일링 사용) | arXiv:1710.03740 | ✅ Web |
| ○ [효율화][분산] | Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2019) · Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* (2023) — 제목 수준만 확인 | arXiv:1910.02054 · arXiv:2304.11277 | ✅ Web |
| ○ [벤치마크][비판] | Raji et al., *AI and the Everything in the Whole Wide World Benchmark* (2021) — 제목 수준만 확인 | arXiv:2111.15366 | ✅ Web |
| ★ [책임성] | Baker & Church, *Protein design meets biosecurity* (Science 2024;383:349) — 제목 수준만 확인(짧은 글) | 10.1126/science.ado1671 | ✅ Crossref |
| ◆ [책임성] | Sandbrink, *Artificial intelligence and biological misuse: Differentiating risks of language models and biological design tools* (2023; 초록: LLM은 비전문가의 장벽을 낮추고 생물 설계 도구(BDT)는 숙련된 행위자의 능력을 넓힐 수 있다고 구분) | arXiv:2306.13952 | ✅ Web |

(검증·분할 관련 필독은 01의 DOME·MMseqs2, 벤치마크는 02·03의 TAPE/FLIP/PEER/BEND를 참고한다.)

## 07 Capstone — 재현 · 보고 · 문서화
| 필수도 | 논문 | 출처 / DOI | 검증 |
|---|---|---|---|
| ★ [재현] | Pineau et al., *Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)* (초록: 재현성 = 같은 코드·데이터(가능하면)로 논문에 보고된 것과 비슷한 결과를 얻는 것; 학술지 게재 정보는 직접 확인 안 됨) | arXiv:2003.12206 | ✅ Web |
| ◆ [문서화] | Mitchell et al., *Model Cards for Model Reporting* (초록: 공개 모델에 의도한 사용 사례와 성능 특성을 담은 문서를 동반하자고 제안; 게재처는 직접 확인 안 됨) | arXiv:1810.03993 | ✅ Web |
| ◆ [문서화] | Gebru et al., *Datasheets for datasets* (Commun ACM 2021;64:86–92; 초록: 데이터셋의 동기·구성·수집 과정·권장 용도 등을 문서화하자고 제안) | 10.1145/3458723 | ✅ Crossref |

(검증·분할 관련 필독은 01의 DOME·MMseqs2, 비판적 평가 논문은 각 모듈 구역을 참고한다.)

---

## 남은 불확실성 (⚠️)
- ⚠️ ESM C: 동료심사 논문이 확인되지 않았다(블로그·모델 카드만 있다). 라이선스가 바뀔 수 있다.
- 모듈 07 읽기 목록: Pineau·Mitchell은 arXiv 초록까지만 확인했다(학술지/학회 게재 여부는 미확인). Gebru는 Crossref 서지와 arXiv 초록을 확인했다.
- 모듈 06 읽기 목록: Sledzieski·ZeRO·FSDP·Raji·Baker & Church는 제목·서지 수준만 확인했다(초록 미확인). LoRA·SAE·혼합 정밀도·Biderman·Sandbrink는 arXiv 초록까지 확인했고 학회/학술지 게재 여부는 미확인이다(FlashAttention은 NeurIPS 2022 게재를 Crossref로 확인했다).
- ⚠️ UCE, Ribonanza, Boltz-1/2, Chai-1, Biomni, ESMFold2 프리프린트: 프리프린트 상태만 확인했다. 저널 게재판은 찾지 못했다(없다는 뜻은 아니다).
- ⚠️ scBERT: Matters Arising(2024-12)의 내용·결론은 미확인이다.
- ⚠️ Uni-Mol: DOI가 없다(OpenReview 기준으로 인용했다).
- ⚠️ Chai-1: 사전판 게시일이 출처마다 하루 차이가 난다(10/10 vs 10/11). DOI로 인용한다.
- ⚠️ 모듈 05: ChemBERTa-2·ProtST·Praski·Guo·de Jong·LAB-Bench는 arXiv 수준(초록)만 확인했고 학술지/학회 게재 여부는 미확인이다. ProTrek·CellWhisperer는 제목·DOI만 확인했고 저자·저널판은 미확인이다. UNI/CONCH/UNI2-h·Virchow·Prov-GigaPath 가중치는 Hugging Face에서 접근 승인(gated)이 필요하다(HF 메타데이터 `gated=auto` 기준이며, 라이선스는 UNI/CONCH가 cc-by-nc-nd-4.0로 표기되어 있다). 이 저장소의 노트북은 이들을 쓰지 않는다.

## 운영 메모 (교차 검토 반영)
- 읽기만으로는 18주에 전문가 수준에 닿기 어렵다. **모듈마다 산출물 1개**: (a) 단순 baseline 재현, (b) split 변경 효과 확인, (c) 동결 표현 vs 미세조정 비교 중 하나.
- 모듈마다 데이터 누출 점검 항목: 단백질=상동성, 유전체=중복·상동 구간, 단일세포=donor·study·batch, 분자=scaffold·시간 분할, 공통=사전학습 데이터와 평가 데이터의 중복.
- 검토 모델 간 이견은 본문에 병기했다(예: Mamba). 검토 모델의 서지 주장은 도구로 확인된 것만 반영했다.
