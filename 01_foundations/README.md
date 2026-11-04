# 01. Foundations — 토큰, 목적함수, 평가

**기간**: 1~2주차 (약 16~20시간) · **실행 등급**: 🟢 Mac 로컬 (CPU/MPS)
**전제**: Transformer와 기본 딥러닝(중급), 분자생물학 기초(중급). 부족하면 [00_orientation](../00_orientation/)의 자가진단을 먼저.

> 표기: † = 논문 본문에서 직접 확인이 필요한 서술(초록·제목·검색으로 확인되지 않음). 인용 논문은 [reading_list](../resources/reading_list.md)의 01~06 구역 참고.

---

## 학습 목표
이 모듈을 마치면 다음을 할 수 있다.
1. 생물 서열의 **토크나이징 방식**을 비교하고 과제에 맞게 고른다 (단일 잔기·k-mer·BPE).
2. **자기지도 목적함수**(MLM, 자기회귀, 대조학습, 확산)의 수식과 쓰임새를 설명한다.
3. 사전학습 모델로 **zero-shot 변이 점수**를 만드는 두 방식(masked marginal, log-likelihood 비율)을 설명한다.
4. **데이터 누출**(상동성 등)이 성능을 어떻게 부풀리는지 실험으로 보이고, 올바른 split을 설계한다.
5. FM 평가에서 **baseline·지표·통계**의 최소 요건을 체크리스트로 점검한다.

---

## 1. 생물 서열을 토큰으로

언어모델에서 "단어"에 해당하는 단위를 정하는 일이다. 선택이 **시퀀스 길이(=attention 비용), 어휘 크기, 학습 신호**를 모두 바꾼다.

| 방식 | 설명 | 장점 | 단점 | 예 |
|---|---|---|---|---|
| 단일 잔기 / 단일 뉴클레오타이드 | 문자 하나 = 토큰 | 어휘 작음(단백질 약 20, DNA 4 + 특수 토큰), 해상도 최대, 단순 | 긴 DNA에서 시퀀스 길이가 그대로 비용이 됨 | ESM-2(아미노산 단위)†, HyenaDNA·Evo 2(단일 뉴클레오타이드 해상도) |
| k-mer (겹침) | 길이 k 윈도를 1칸씩 이동 | 짧은 모티프를 토큰 하나로 표현 | 토큰 수 ≈ 서열 길이(압축 없음), 마스킹 시 이웃 토큰에 정답이 일부 노출될 수 있음† | DNABERT |
| k-mer (비겹침) | 길이 k 윈도를 k칸씩 이동 | 토큰 수가 L/k로 줄어듦 | **한 염기만 삽입/삭제돼도 프레임이 밀려 토큰이 전부 바뀜** | Nucleotide Transformer(k-mer 계열)† |
| BPE | 빈출 쌍을 병합해 가변 길이 토큰 학습 | 데이터 기반 압축, 프레임 이동에 덜 민감 | 어휘·압축 trade-off, 토큰 경계가 생물학적 단위와 안 맞을 수 있음 | DNABERT-2 |
| 구조 인지 토큰 | 잔기 + 구조 토큰(예: Foldseek 알파벳)을 결합 | 서열에 3D 정보를 직접 주입 | 구조가 필요함(예측 구조의 오차 포함) | SaProt |
| 유전자 토큰 (단일세포) | 세포 = 유전자 토큰의 시퀀스(순서 규칙 필요) | 서열이 아닌 발현 프로파일을 시퀀스로 | 순서 없는 집합을 시퀀스로 만드는 규칙(예: 발현 순위)이 설계 선택† | Geneformer, scGPT |

**선택 기준 질문**
- 입력 길이는 얼마인가? (단백질 수백 vs 유전체 영역 수십 kb~Mb)
- 정밀한 변이(SNV) 해상도가 필요한가?
- 마스킹/예측 목표가 토큰 경계와 어긋나지는 않는가?
- 압축률이 이득인가, 아니면 생물학적 해석을 해치는가?

> **실습 [01_tokenizers](notebooks/01_tokenizers_compare.ipynb)**: 같은 합성 서열에 세 방식을 적용해 토큰 수·어휘 크기·프레임 이동 민감도·마스킹 누출을 직접 측정한다.

---

## 2. 아키텍처 복습 — 생물학 관점에서

### 2.1 Self-attention과 비용
$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right)V$$

| 연산 | 시간 | 메모리 | 비고 |
|---|---|---|---|
| 표준 self-attention | O(L²·d) | O(L²) (attention map) | 긴 DNA에 병목 |
| FlashAttention | O(L²·d) | 선형에 가깝게 | exact attention, I/O 최적화 |
| Hyena / Mamba 계열 (SSM·implicit conv) | 서브쿼드라틱~선형 | 선형 | HyenaDNA, Caduceus(BiMamba) 등 |

### 2.2 위치 표현
절대 위치 임베딩 / 상대 위치 / 회전(RoPE) / ALiBi. DNABERT-2는 ALiBi를, Flashzoi는 회전 위치 인코딩을 사용한다(초록 기준). 긴 문맥으로 외삽할 수 있는지가 핵심 차이.

### 2.3 생물학 특화 고려
- **방향성**: DNA 서열 정보는 읽는 방향이 고정돼 있지 않다 → 양방향 모델, reverse-complement 등변성(Caduceus).
- **길이 분포**: 단백질 vs 유전체 영역은 수천 배 차이 → 아키텍처 선택이 갈린다.
- **지도 vs 자기지도**: Enformer·Borzoi는 실험 트랙을 직접 예측하도록 **지도학습**하고, DNABERT·Evo 등은 **서열만으로 자기지도 사전학습**한다. 같은 표에서 순위를 매기면 안 된다.

---

## 3. 자기지도 목적함수

| 목적함수 | 손실 | 예 | 자연스러운 용도 |
|---|---|---|---|
| MLM | $-\sum_{i\in M}\log p(x_i\mid x_{\setminus M})$ | ESM-2, DNABERT | 임베딩, masked-marginal 변이 점수 |
| 자기회귀 | $-\sum_t\log p(x_t\mid x_{<t})$ | ProGen, Evo | 서열 생성, 서열 log-likelihood |
| 대조학습(InfoNCE) | $-\log\frac{\exp(s(q,k^+)/\tau)}{\sum_k \exp(s(q,k)/\tau)}$ | CLIP, CONCH | 모달 간 정렬(이미지–텍스트) |
| 확산/노이즈 제거 | 노이즈 예측(또는 데이터 복원) | RFdiffusion, AlphaFold 3 | 3D 구조·복합체 생성 |

### 3.1 Zero-shot 변이 효과 점수
사전학습 분포가 진화적 제약을 반영한다는 가정에서, 변이를 추가 학습 없이 점수화한다.

- **Masked marginal (MLM)**: 위치 $i$를 가리고 $\;s=\log p(x_i=\text{mut}\mid x_{\setminus i}) - \log p(x_i=\text{wt}\mid x_{\setminus i})$
- **전체 서열 log-likelihood 비율 (자기회귀 등)**: $\;s=\log p(x^{\text{mut}})-\log p(x^{\text{wt}})$

해석 주의: 점수는 "자연 서열 분포에서 얼마나 그럴듯한가"의 대리 지표이며, 실제 fitness와의 일치는 assay·단백질 가족마다 달라 **벤치마크(ProteinGym 등)로 검증**해야 한다.

---

## 4. Scaling

- 언어모델의 손실은 모델 크기·데이터·연산량과 멱법칙 관계를 보이고(Kaplan et al.), 주어진 연산 예산에서 모델과 토큰 수를 함께 늘려야 한다는 결과(Hoffmann et al.)가 있다 — 읽고 두 논문의 결론이 어떻게 다른지 비교해 보자.
- 생물학 사례(초록 기준): ESM-2 논문은 모델을 최대 15B까지 키우면 표현에서 원자 수준 구조 정보가 드러난다고 보고했고, InterPLM은 ESM-2 650M이 8M보다 해석 가능한 개념을 더 많이 담는다고 보고했다.
- 그러나 **"클수록 항상 좋다"는 생물학에서 과제·데이터에 따라 다르다.** 변이 효과나 단일세포 과제에서 작은 모델·단순 baseline과 비교한 결과는 해당 벤치마크/비판 논문에서 직접 확인할 것(모듈 02·04).
- 데이터 측면: 서열 데이터베이스는 계통·종 표본 편향이 크고 중복이 많다. 중복 제거(클러스터링) 기준이 결과를 바꾼다.

---

## 5. 평가의 함정 — 이 모듈의 핵심

### 5.1 상동성 누출 (homology leakage)
서열은 진화적으로 관련돼 있어 i.i.d.가 아니다. random split을 하면 테스트 서열의 **가까운 친척이 훈련 세트에** 들어가 모델이 일반화가 아니라 **기억**으로 점수를 얻는다. 보고된 성능은 "새로운 가족"이 아니라 "아는 가족의 새 구성원"에 대한 성능이다.

> **실습 [03_homology_split_leakage](notebooks/03_homology_split_leakage.ipynb)**: 합성 단백질 가족으로 random split vs 클러스터/가족 split을 비교하고, 훈련 세트와의 최대 서열 동일도 구간별 정확도를 그린다.

### 5.2 split 설계

| 데이터 | 권장 split 단위 | 주의 |
|---|---|---|
| 단백질 서열 | 서열 동일도 임계값 클러스터(예: MMseqs2) 또는 가족/슈퍼패밀리 | 임계값이 느슨할수록 보수적. 임계값을 **보고**할 것 |
| DNA / 유전체 | 염색체/영역 단위 holdout, 반복·중복 구간 제거 | 유전자 중복·반복서열이 경계를 넘는지 확인 |
| 단일세포 | donor / study / batch 단위 | 같은 donor의 세포가 train·test에 섞이면 누출 |
| 분자(소분자) | scaffold 또는 시간 split | 유사 구조 누출 |
| 공통 | 사전학습 코퍼스 ∩ 평가 데이터 | **FM은 평가 데이터를 이미 봤을 수 있다** |

MMseqs2 클러스터링 예시(설치 후 `mmseqs easy-cluster -h`로 옵션 확인):
```bash
mmseqs easy-cluster seqs.fasta out tmp --min-seq-id 0.3 -c 0.8
```

### 5.3 Baseline
FM을 평가할 때 항상 함께 보고할 것:
- **단순 baseline**: 아미노산/k-mer 빈도, 선형·로지스틱 회귀, 평균 예측.
- **FM 이전 방법**: MSA 기반 모델(EVE), CNN 기반 서열→기능 모델(DeepSEA), scVI 같은 확률 모델 등 — reading_list의 [기준선] 태그.
- **같은 조건**: 같은 split, 같은 튜닝 예산.

### 5.4 지표와 통계
- 순위 상관(Spearman), 분류에서는 불균형이면 AUPRC도. 
- 신뢰구간은 **그룹 단위(가족/donor) 부트스트랩**으로 — 샘플 단위 부트스트랩은 상관된 샘플 때문에 과소 추정된다.
- 반복 시드 및 분산 보고.

### 5.5 DOME 권고 (Walsh et al. 2021)
지도 ML 검증 권고를 **D**ata, **O**ptimization, **M**odel, **E**valuation 네 영역으로 정리한다. 논문을 읽고 위 체크리스트와 대응시켜 보자 (reading_list 01 ★).

### 5.6 점검 체크리스트
- [ ] split 규칙을 데이터를 보기 전에 정했는가?
- [ ] 훈련·테스트 사이의 최대 서열 동일도(또는 donor 중복)를 측정했는가?
- [ ] 사전학습 데이터와 평가 데이터의 겹침을 확인했는가?
- [ ] 단순 baseline과 FM 이전 방법을 같은 split에서 돌렸는가?
- [ ] 그룹 단위 신뢰구간을 보고했는가?

---

## 6. 이해도 점검

**Q1.** 비겹침 6-mer 토크나이저에서 서열 맨 앞에 염기 1개가 삽입되면 토큰은 어떻게 변하는가? 단일 뉴클레오타이드 토크나이저는?
<details><summary>답</summary>
비겹침 k-mer는 윈도 경계가 1칸 밀려 거의 모든 토큰이 다른 6-mer로 바뀐다(어휘 안의 다른 토큰). 단일 뉴클레오타이드 토크나이저는 맨 앞에 토큰 1개가 추가될 뿐 나머지는 위치만 1칸 이동한다. (노트북 01에서 수치로 확인)
</details>

**Q2.** 겹침 6-mer에서 MLM으로 한 토큰만 마스킹하면 안 되는 이유는?
<details><summary>답</summary>
이웃 토큰들이 마스킹된 토큰과 염기를 공유하므로(바로 옆 토큰은 5개 공유), 마스킹된 구간의 염기가 이웃 토큰에 그대로 남아 있다. 예측이 쉬워져 학습 신호가 약해지므로, 특정 염기를 숨기려면 그 염기를 포함하는 겹치는 토큰들을 함께(연속 구간으로) 마스킹해야 한다. (노트북 01 4절)
</details>

**Q3.** MLM 학습 모델의 masked-marginal 점수가 높다는 것은 무엇을 뜻하고 무엇을 뜻하지 않는가?
<details><summary>답</summary>
뜻함: 해당 변이가 자연 서열 분포(진화적 제약) 안에서 상대적으로 그럴듯하다. 뜻하지 않음: 그 변이가 특정 assay의 기능을 보존한다는 보장 — 과제/단백질에 따라 상관이 다르므로 벤치마크로 검증해야 한다.
</details>

**Q4.** Random split과 클러스터 split의 성능 차이가 클수록 무엇을 의미하는가?
<details><summary>답</summary>
모델(또는 특징)이 일반화보다 훈련 가족의 기억에 의존한다는 신호다. 보고된 random-split 성능은 새 가족에 대한 성능을 과대평가한다.
</details>

**Q5.** 자기회귀 모델과 MLM 모델의 zero-shot 점수를 같은 표에서 비교할 때 주의할 점은?
<details><summary>답</summary>
점수 정의가 다르고(위치 단위 masked marginal vs 전체 서열 log-likelihood 비율), 토크나이저·사전학습 데이터·문맥 길이가 다르다. 같은 평가 데이터와 지표(Spearman 등)로 비교하되, 모델 크기·데이터 차이를 함께 보고해야 한다.
</details>

**Q6.** 단일세포 FM의 평가 split이 같은 donor를 train과 test에 섞으면 어떤 문제가 생기는가?
<details><summary>답</summary>
donor 고유의 배치/유전적 효과를 모델이 기억해 일반화 성능이 부풀려진다. 새 donor·새 study에 대한 성능이 실제 관심사이므로 donor/study 단위로 분할해야 한다.
</details>

**Q7.** Kaplan 등과 Hoffmann 등이 계산 예산을 모델 크기와 데이터에 배분하는 방식에서 어떻게 다른지 두 논문에서 찾아 한 문단으로 정리하라. *(정답 없음 — 논문 읽기 과제)*

**Q8.** 사전학습 코퍼스와 평가 데이터의 겹침을 확인하려면 어떤 절차가 필요한가?
<details><summary>답</summary>
평가 서열을 사전학습 코퍼스(공개된 경우)에 대해 서열 동일도 검색/클러스터링(예: MMseqs2)으로 조회해 최대 동일도 분포를 보고한다. 코퍼스가 비공개이면 한계로 명시하고, 사전학습 이후에 공개된 데이터(시간 split)로 평가하는 방법을 고려한다.
</details>

---

## 7. 읽기 (reading_list 01 구역)
| 필수도 | 항목 | 비고 |
|---|---|---|
| ★ | Walsh et al., DOME (Nat Methods 2021) | 5절과 함께 |
| ★ [실습] | Hu et al., LoRA | 모듈 06 대비 |
| ◆ | Kaplan et al. / Hoffmann et al. | 4절, Q7 |
| ◆ | Radford et al., CLIP | 3절 대조학습, 모듈 05 대비 |
| ◆ | Gu & Dao, Mamba | 2.1 |
| ◆ [실습] | Steinegger & Söding, MMseqs2 | 5.2 |
| ○ [복습] | Vaswani et al., BERT | 중급 이상이면 가볍게 |

## 8. 실습 노트북
| 노트북 | 등급 | 내용 |
|---|---|---|
| [01_tokenizers_compare](notebooks/01_tokenizers_compare.ipynb) | 🟢 | 토크나이저 비교, 프레임 이동 민감도, 마스킹 누출 |
| [02_mini_protein_mlm](notebooks/02_mini_protein_mlm.ipynb) | 🟢 | 합성 서열에서 PyTorch로 작은 MLM을 직접 학습 — 의존성(공진화)을 학습하는가 |
| [03_homology_split_leakage](notebooks/03_homology_split_leakage.ipynb) | 🟢 | random vs 클러스터/가족 split, 동일도 구간별 성능 |

모든 노트북은 합성 데이터로 **네트워크 없이** 실행된다. 합성 데이터의 결과는 개념 시연이며 실제 단백질/게놈의 수치가 아니다.

**참고 결과** (Mac M1 Pro, Python 3.12, torch 2.15.0, seed=0 — 버전·장치에 따라 달라질 수 있음)
| 노트북 | 확인 포인트 | 기대 값(대략) | 소요 |
|---|---|---|---|
| 01 | 비겹침 6-mer의 프레임 이동 후 토큰 보존 비율 | 0.50 (단일 nt/겹침/BPE는 약 1.0) | 약 10초 |
| 01 | 겹침 6-mer에서 누출 0이 되는 마스킹 폭 | 연속 11토큰(±5) | |
| 02 | 커플링 위치 마스킹 정확도 (오라클 / baseline) | 약 0.91 (0.91 / 0.13) | 약 2~3분 (MPS) |
| 03 | AUROC: random / family / cluster(0.20) | 약 0.98 / 0.91 / 0.87(±0.07) | 약 10초 |

과제: [exercises.md](exercises.md)

## 다음
→ [02_protein](../02_protein/) (모듈 02에서 실제 단백질 언어모델을 다룬다)
