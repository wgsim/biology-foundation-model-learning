# Glossary

> **English translation.** AI-assisted translation of the [Korean original](glossary.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original. The first column keeps the Korean term, so that you can look up the term used in the Korean documents.

Written: 2026-11-06. The terms used in this course (modules 00–07), organized with explanations (written in Korean in the original). Details specific to a model or a paper (size, data scale, etc.) are not written here; they are in `reading_list.md` and in each module README. The last column of the tables is the module where the term was first or most thoroughly covered. The definitions follow the meaning used in this course, and where the usage differs by field, it is noted in the remarks.

## 1. Models and training

| Term | English | Explanation | Module |
|---|---|---|---|
| 파운데이션 모델 | foundation model (FM) | A model that is pretrained on large-scale data and then used for many downstream tasks. Even if it is called an "FM", it does not always beat a simple baseline | 00, 04 |
| 사전학습 | pretraining | The stage of first learning a general representation from large-scale unlabeled (or weakly labeled) data | 01 |
| 자기지도 학습 | self-supervised learning | The learning signal is made from the input itself (guessing masked tokens, predicting the next token, etc.) | 01 |
| 마스크 언어모델 | masked language model (MLM) | An objective that masks some tokens and guesses them. Used by ESM-2, DNABERT and Nucleotide Transformer | 01 |
| 자기회귀 | autoregressive | Predicting the next token from the preceding tokens. Suited to generation. ProGen, the Evo family | 01 |
| 대조학습 | contrastive learning | Learning to bring matching pairs close and non-matching pairs far. CLIP, sequence–text alignment | 05 |
| InfoNCE | InfoNCE | A contrastive loss that treats the other items in the batch as negatives and solves the classification in which "the same index is the answer". If all the logits look the same, the initial loss is ln(batch size) | 05 |
| 확산 모델 | diffusion model | Generation by removing noise step by step. RFdiffusion, the structure-generation part of AlphaFold 3 | 02 |
| 토크나이저 | tokenizer | The rule for cutting a sequence into the units of model input | 01 |
| k-mer | k-mer | A contiguous subsequence of length k. It can be used overlapping or non-overlapping | 01 |
| BPE | byte-pair encoding | Tokenization that builds the vocabulary by repeatedly merging pieces that often occur together. Adopted by DNABERT-2 (the early DNABERT used overlapping k-mers) | 01, 03 |
| 임베딩 | embedding | A vector that represents a sequence, a cell or a molecule. It depends on which layer of the model and how it was pooled | 01 |
| 평균 풀링 | mean pooling | Averaging the per-token vectors into a single vector for a sequence. Check whether padding and special tokens (`<cls>`, `<eos>`) were excluded | 02 |
| 선형 probe | linear probe | An evaluation that trains only a linear model (Ridge, etc.) on frozen embeddings to see whether the representation holds the information | 02 |
| 동결 | frozen | The pretrained weights are not updated | 02, 06 |
| 미세조정 | fine-tuning | Updating even the pretrained weights with task data. Risk of overfitting and forgetting | 06 |
| few-shot | few-shot | Giving only a few task examples (a prompt or a small amount of training). With no examples it is zero-shot, and with all of them it is ordinary supervised learning. A setting not covered in this course | — |
| 분포 이동 | distribution shift | A state in which the distributions of the training and evaluation (deployment) data differ. Splitting by new positions, new families and new donors imitates it | 04, 07 |
| 매개변수 효율 미세조정 | parameter-efficient fine-tuning (PEFT) | Training only a small subset of parameters (LoRA, etc.) | 06 |
| LoRA | low-rank adaptation | Adds a low-rank decomposition matrix to the frozen weights and trains only that matrix. It reduces the trainable parameters and the optimizer memory, but the forward and backward computation remains | 06 |
| LP-FT | linear probing then fine-tuning | The order of first training only the head (linear probe) and then fine-tuning the whole. A method meant to reduce the damage that the large gradients of a randomly initialized head do to the pretrained representation†. Not run in this repository | 06 |
| 스케일링 법칙 | scaling law | The empirical relationship between model size, data, compute and loss. It may differ by domain (sequences) | 01 |
| 어텐션 | attention | An operation that mixes the context by a weighted sum between tokens. The standard implementation makes an L×L score matrix, so time and memory grow as L² (FlashAttention-type methods do not store the score matrix and so reduce memory, but the amount of computation is still L²†) | 01 |
| FlashAttention | FlashAttention | A CUDA implementation that computes exact attention tile by tile and reduces memory I/O†. Not available on Mac MPS | 06 |
| 상태공간 모델 / 장거리 합성곱 | state-space model (SSM) / long convolution | A family that processes long sequences at a lower cost than attention. Mamba is an SSM and Hyena is an implicit long convolution, and DNA models such as HyenaDNA and Caduceus use them for long-range context | 01, 03 |
| 혼합 정밀도 | mixed precision | Half-precision (bf16/fp16) computation + an fp32 copy of the weights. The gain is mainly in speed and activation memory | 06 |
| 그래디언트 체크포인팅 | gradient checkpointing | Trades time for memory by not storing some activations and recomputing them in backpropagation | 06 |
| 데이터 병렬 | data parallel (DDP) | Replicates the model, splits the data and averages the gradients | 06 |
| ZeRO / FSDP | ZeRO / FSDP | A method that reduces the memory of one device by storing the parameters, gradients and optimizer state split across devices† | 06 |
| zero-shot | zero-shot | Predicting from only the output of the pretrained model, without training for that task | 02 |
| 헤드 | head | A small prediction module (a linear layer, etc.) placed on top of the embeddings | 02 |
| 리비전 고정 | revision pinning | Fixing a model/data to a specific commit hash of the repository to secure reproducibility | 05, 07 |

## 2. Evaluation, statistics and research methods

| Term | English | Explanation | Module |
|---|---|---|---|
| 누출 | leakage | A state in which information used for evaluation has leaked into the training process (split, preprocessing, setting selection) | 01, 07 |
| 상동성 분할 | homology split | A split that divides train/test by a sequence-identity threshold or by family | 01 |
| 위치 블록 분할 | position-block split | A split that groups consecutive position intervals together (blocking the leakage of other substitutions at the same position) | 07 |
| scaffold 분할 | scaffold split | A split that puts molecules with the same Bemis–Murcko scaffold on the same side. Molecules can still be similar even if the scaffolds differ | 05 |
| 시간 분할 | time split | Splitting train and test by publication/measurement time | 05 |
| 기증자 단위 분할 | donor-level split | Splitting so that cells of the same donor do not straddle train and test | 04 |
| LODO | leave-one-donor-out | A cross-validation that leaves one donor as test | 04 |
| transductive 적합 | transductive fitting | Fitting the preprocessing (HVG selection, scaling, PCA, batch correction) without labels, including the held-out samples. It is not label leakage, but it can favor that baseline, so the result fitted with only the training folds is looked at together as a sensitivity check | 04, 07 |
| 오염 | contamination | The evaluation data is in the pretraining corpus | 01 |
| baseline | baseline | A comparison reference method. In this course: ① one that learns no parameters, ② simple features, ③ a randomly initialized model | 07 |
| 무작위 초기화 대조군 | random-init control | A model initialized from the same architecture without pretraining. It separates the contribution of pretraining. It is created fresh for each seed | 04, 07 |
| 음성 대조 | negative control | Look at the score when the signal is absent by shuffling labels/inputs. A diagnostic, not proof that leakage is absent | 05, 07 |
| 선택 효과 | selection effect | The performance difference created by the procedure itself of picking, among candidates, the one with the highest train score | 06 |
| 짝지은 비교 | paired comparison | Comparing by taking the score difference of two methods on the same split and the same items | 04, 07 |
| 그룹 부트스트랩 | group bootstrap | Obtains an interval by resampling the correlation unit (groups) with replacement. It is meaningless when there are few test clusters | 07 |
| 조건부 구간 | conditional interval | An interval obtained by fixing the training and initialization and redrawing only the test. It does not reflect the training variation | 07 |
| Spearman 상관 | Spearman correlation | Rank correlation. Undefined (NaN) for a constant prediction | 02 |
| AUROC | ROC-AUC | The probability that a randomly chosen positive is scored higher than a negative (ties count 1/2). A ranking performance, separate from probability calibration. For fold models, compute the scores per fold and average | 05 |
| AUPRC | average precision (AP) / PR-AUC | Suited to classification with rare positives. `average_precision_score` is AP, a step-wise integral, and differs slightly from the trapezoidal area of the PR curve. The random baseline is roughly the positive fraction (depends on prevalence) | 03 |
| macro-F1 | macro-averaged F1 | The simple average of per-class F1. Treats classes equally in imbalanced data | 04 |
| 균형정확도 | balanced accuracy | The simple average of per-class recall. The uniform-guess baseline is the inverse of *the number of classes in that evaluation set* | 04, 07 |
| Recall@k | recall@k | The fraction of cases where the correct answer is within the top k (when there is one correct answer per query; with several, distinguish the number of correct answers found / the total number of correct answers from the hit rate of "found at least one"). Depends on the gallery size and the tie rule | 05 |
| 갤러리 | gallery | The candidate set from which the correct answer must be chosen in retrieval | 05 |
| ties-against-truth | ties-against-truth | A conservative tie rule that counts candidates with a score equal to the correct answer as ahead of it | 05 |
| 사전 등록 | preregistration | Fixing (committing) and recording the split, metrics, primary comparison and decision rule before producing the results | 07 |
| 결정 규칙 | decision rule | The condition that decides "supported/not supported". Fixed as a function in code | 07 |
| 민감도 분석 | sensitivity analysis | Change only one factor at a time and see how much the conclusion changes. Not used to choose settings | 05, 07 |
| 효과 크기 | effect size | The size of the difference (separate from statistical significance). Decide the minimum meaningful difference in advance | 07 |
| red flag | red flag | A procedural defect in the rubric that automatically lowers the score of the corresponding dimension | 07 |
| 재현성 | reproducibility | Getting the same result with the same code and data. Record the environment, seeds and hashes | 07 |
| DOME | DOME | Recommendations for reporting the validation of supervised ML in biology (data, optimization, model, evaluation) | 01 |

## 3. Proteins

| Term | English | Explanation | Module |
|---|---|---|---|
| 아미노산 서열 | amino-acid sequence | A string of the 20 standard amino acids. Single-residue tokenization is the default | 01 |
| MSA | multiple sequence alignment | Homologous sequences aligned. Input/training information of models such as AlphaFold 2 and EVE | 02 |
| DMS | deep mutational scanning | An experiment that measures the effects of many variants at once. It has noise | 02 |
| 적합도 | fitness | The functional/growth score of a variant measured in DMS. Its goal differs from clinical pathogenicity | 02 |
| 상위성 | epistasis | Variant effects are not a simple sum but depend on each other | 02 |
| masked marginal | masked marginal | Scores by the difference between the log probabilities of the variant amino acid and the wild type when the variant position is masked. Distinguish it from the wild-type marginal (scoring all positions with 1 pass of the wild-type sequence without masking; cheap but the wild-type residue is visible in the input) and from the ratio of the pseudo-log-likelihoods (PLL) of the variant and wild-type sequences (expensive but extendable to multiple variants) | 02 |
| contact map | contact map | A matrix of whether residue pairs are close in 3D. There are studies that read it from attention. To evaluate predictions, the precision of the top L (or L/5) predictions (precision@L/5) is often used (module 02 notebook 03) | 02 |
| pLDDT | predicted LDDT | Per-residue confidence of a structure prediction (0–100) | 02 |
| PAE / pAE | predicted aligned error | The expected error of the position of another residue when aligned on one residue (per residue pair) | 02 |
| pTM / ipTM | predicted TM-score / interface pTM | Predicted scores of the structural accuracy of the whole/interface | 02 |
| inverse folding | inverse folding | Designing a sequence from a backbone structure. ProteinMPNN, ESM-IF1 | 02 |
| de novo 설계 | de novo design | Designing from scratch a protein that is not in nature. A subject of dual-use review | 02, 06 |
| 임상 병원성 예측 | clinical variant pathogenicity prediction | Predicting whether a human variant is associated with disease. AlphaMissense, EVE | 02 |

## 4. DNA · RNA · genome

| Term | English | Explanation | Module |
|---|---|---|---|
| 참조 게놈 | reference genome | The genome assembly that serves as the coordinate reference (e.g., hg38). If the coordinate systems are not matched, the positions are off | 03 |
| 프로모터 | promoter | A regulatory sequence near the transcription start | 03 |
| 인핸서 | enhancer | A regulatory sequence that raises transcription from a long distance | 03 |
| TSS | transcription start site | The transcription start position | 03 |
| 염색질 접근성 | chromatin accessibility | Open chromatin measured by the DNase/ATAC signal | 03 |
| ChIP / CAGE | ChIP / CAGE | Functional-genomics experiments that measure protein–DNA binding (including histone marks) / transcription starts | 03 |
| sequence-to-function | sequence-to-function | A supervised model that directly predicts functional tracks (expression, accessibility, etc.) from a DNA sequence. Enformer, Borzoi, AlphaGenome | 03 |
| 트랙 / bin | track / bin | One line of signal prediction that divides a sequence into intervals (bins) of a fixed width | 03 |
| LLR | log-likelihood ratio | log p(alt\|context) − log p(ref\|context). The zero-shot variant score of a DNA LM | 03 |
| ISM | in silico mutagenesis | A way to see the change in the prediction by changing one base | 03 |
| phyloP / phastCons | phyloP / phastCons | Conservation scores across species. A strong baseline even without training | 03 |
| CADD | CADD | A variant deleteriousness score that integrates many annotations | 03 |
| 멘델 / 복합 형질 | Mendelian / complex traits | Single-gene disease vs traits to which many variants contribute | 03 |
| 매칭된 대조군 | matched controls | Non-causal variants matched on variant type, TSS distance, etc. A design to reduce the information of simple features | 03 |
| 개인 간 cis 변이 효과 | personal cis-variant effect | The effect by which a variant near a gene produces an individual's expression difference. A different task from the between-gene differences of the reference genome | 03 |
| RNA 기반 FM | RNA foundation model | A pretrained model that predicts structure and function from RNA sequences (RNA-FM, RiNALMo, etc.) | 05 |

## 5. Single-cell

| Term | English | Explanation | Module |
|---|---|---|---|
| 카운트 행렬 | count matrix | Cell × gene UMI counts | 04 |
| HVG | highly variable genes | Selecting the genes with large variation between cells | 04 |
| 순위값 인코딩 | rank value encoding | A sequence of gene IDs sorted in descending order of value after dividing by the per-gene corpus median after depth normalization. Geneformer | 04 |
| 희소성 | sparsity (zeros) | The property that most of the single-cell counts are 0. Biological non-expression and technical detection failure are mixed†  | 04, 05 |
| 배치 효과 | batch effect | Non-biological variation that comes from differences in experimental conditions | 04 |
| 배치 보정 | batch correction / integration | Methods that reduce the batch signal (Harmony, scVI, etc.) | 04 |
| ASW | average silhouette width | A cluster separation metric. Used in scib for batch mixing/type preservation | 04 |
| 보존 vs 혼합 | bio-conservation vs batch mixing | Two goals of mixing the batches while keeping the cell-type structure. Read them together | 04 |
| 섭동 예측 | perturbation prediction | A task of predicting the transcriptome change after a gene perturbation | 04 |
| scVI | scVI | A single-cell model based on a variational autoencoder. Because training is stochastic, the numbers differ slightly from run to run | 04 |

## 6. Molecules and multimodal

| Term | English | Explanation | Module |
|---|---|---|---|
| SMILES | SMILES | A notation that writes a molecule as a string. The same molecule can have several notations | 05 |
| ECFP / Morgan 지문 | ECFP / Morgan fingerprint | A bit (count) fingerprint that hashes the structure around atoms. A strong baseline | 05 |
| Tanimoto 유사도 | Tanimoto similarity | The ratio of the intersection to the union of two bit fingerprints (0–1). For count fingerprints, an extension of the form sum of minima / sum of maxima is used† | 05 |
| Murcko 골격 | Bemis–Murcko scaffold | A molecular scaffold that keeps only the ring systems and the connecting chains | 05 |
| Butina 군집 | Butina clustering | Molecular clustering based on Tanimoto distance. Used as the correlation unit of the bootstrap | 05 |
| 모달리티 | modality | A kind of data, such as sequence, structure, text and images | 05 |
| 모달리티 간극 | modality gap | The phenomenon in which the embeddings of different modalities sit in separate regions of the shared space of contrastive learning | 05 |
| 거짓 음성 | false negative (in contrastive) | An item treated as a negative in the batch that is actually a matching pair | 05 |
| 전체 슬라이드 이미지 | whole-slide image (WSI) | An image of a whole pathology tissue slide | 05 |
| 에이전트 | agent | A setup in which an LLM calls tools and carries out a multi-step task | 05 |

## 7. Interpretation, accountability and safety

| Term | English | Explanation | Module |
|---|---|---|---|
| 희소 오토인코더 | sparse autoencoder (SAE) | A method that reconstructs an embedding with an overcomplete sparse code to find interpretable features | 06 |
| TopK SAE | TopK SAE | A sparsity constraint that keeps only the k largest activations. Controls sparsity directly | 06 |
| 다의성 / superposition | polysemanticity / superposition | A hypothesis that the phenomenon of a neuron responding to several concepts arises because the network represents more features than the number of neurons by superposing them | 06 |
| 죽은 특징 | dead latent | An SAE feature that is almost never on across the whole data | 06 |
| 선택 효과 대조 | selection-effect control | A control in which the same selection procedure is applied to an untrained random dictionary too | 06 |
| 이중 용도 | dual use | The property that beneficial research can also be misused. The scope of release is reviewed | 06 |
| 접근 승인(gated) 모델 | gated model | A model/data that needs agreement to terms. An agent does not agree on your behalf | 05 |
| pickle / `trust_remote_code` | pickle / trust_remote_code | Serialization/remote code that can be a code-execution path. Only after pinning the revision and reviewing | 05 |
| 모델 카드 | model card | A document describing a model's use, performance and limitations | 07 |
| 데이터시트 | datasheet for datasets | A document on the motivation, composition, collection process and recommended use of a dataset | 07 |
| 라이선스 근거 | license evidence | Where the license was confirmed (file, API field). If it could not be confirmed, write "unverified" | 06 |

## 8. Notation in this repository

| Notation | Meaning |
|---|---|
| 🟢 🟡 🔴 | Compute tier: local Mac / Colab GPU / cluster |
| † | A statement that could not be confirmed in the paper text or the official documents |
| ⚠️ 미검증 (unverified) | Must be checked directly before use |
| ✅ PubMed / bioRxiv / Web / Crossref | The sources used to confirm the bibliography (the reading_list legend) |
| ★ ◆ ○ | Required / advanced / optional reading |

<!-- i18n-source: resources/glossary.md blob:5572940014cbf9da763dc4057d01432ac0978937 -->
