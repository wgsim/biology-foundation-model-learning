# 03. DNA · Genome · RNA — DNA 언어모델, 변이 효과, sequence-to-function

**기간**: 7~9주차 (3주, 약 24~30시간) · **실행 등급**: 🟢 Mac 로컬 2종 + 🟢 격리 환경 1종(Enformer) + 🔴 Evo 2(권장 안 함, 아래 참고)
**전제**: [모듈 01](../01_foundations/)(토크나이징·프레임 이동·평가의 함정), [모듈 02](../02_protein/)(zero-shot 점수, Spearman, 부트스트랩).

> 표기: † = 논문 본문·모델 카드에서 직접 확인이 필요한 서술. 인용은 [reading_list](../resources/reading_list.md) 03·05 구역. 2026-11 시점 정보이며 새 모델이 빠르게 나오므로 사용 전 모델 카드를 확인하라.

---

## 학습 목표
1. DNA 언어모델의 **토크나이징·문맥 길이·목적함수**(MLM vs 자기회귀)를 비교한다 (DNABERT, NT, HyenaDNA, Caduceus, GPN, Evo 계열).
2. **지도 sequence-to-function 모델**(DeepSEA, Enformer, Borzoi, AlphaGenome)과 **자기지도 DNA LM**은 학습 목적이 다르다는 점을 설명하고 같은 순위표에 놓지 않는다.
3. NT 임베딩으로 **linear probe**를 만들고 **k-mer baseline**과 공정하게 비교한다.
4. DNA LM의 **log-likelihood ratio(LLR)** 변이 점수를 계산하고 **TraitGym**으로 평가하며, 보존(conservation) 기반 점수(phyloP, CADD)와 비교해 해석한다.
5. **비판적 평가**(BEND, Huang 2023, TraitGym)가 말하는 한계를 요약한다.
6. RNA 모델(RNA-FM, RiNALMo, Ribonanza)의 위치와 한계를 개관한다.

## 3주 일정
| 주 | 주제 | 노트북 | 읽기(★ 중심) |
|---|---|---|---|
| 7 | DNA LM: 토크나이징, 임베딩, probe | [01](notebooks/01_nt_embeddings_promoter_probe.ipynb) | DNABERT, Nucleotide Transformer, DeepSEA |
| 8 | 변이 효과 zero-shot과 벤치마크 | [02](notebooks/02_zero_shot_variant_scores_traitgym.ipynb) | TraitGym, GPN-MSA, Evo, Huang 2023 |
| 9 | sequence-to-function + 산출물 | [03](notebooks/03_enformer_tracks.ipynb) | Enformer, Borzoi, AlphaGenome, BEND |

**모듈 산출물**: (a) NT 임베딩 probe와 **6-mer 빈도 baseline**을 같은 split에서 비교한 표, 또는 (b) TraitGym 변이에서 **본인이 계산한 LLR**을 보존 기반 점수·공개 DNA LM 점수와 같은 변이 집합에서 비교한 표 ([exercises.md](exercises.md) 과제 1·2).

---

## 1. DNA 데이터의 특징 — 단백질과 무엇이 다른가
- **대부분 비코딩**이다. 기능을 결정하는 신호가 짧은 조절 모티프, 간격, 거리에 흩어져 있다.
- **긴 거리 상호작용**: 인핸서는 수십~수백 kb 떨어진 유전자를 조절할 수 있다 → 문맥 길이가 중요 (모듈 01 2.1의 attention 비용 L²).
- **이중가닥과 방향성**: 어느 가닥에서 읽어도 같은 분자의 대체 표현이다 → reverse-complement 처리(증강, 평균, 등변 구조). 단, strand-specific 과제에서는 방향이 중요하다.
- **라벨이 드물다**: 변이 효과의 "정답"은 실험이 어렵고, 대안으로 **진화적 보존**이나 **실험 트랙**을 대리 신호로 쓴다.
- **반복·중복 서열**: 같은 서열이 게놈 여러 곳에 있어 train/test 누출이 쉽게 생긴다 (모듈 01 5.2).

## 2. 모델 계보와 역할

### 2.1 자기지도 DNA 언어모델
| 모델 | 토큰 | 목적함수/구조 | 문맥 | 비고 |
|---|---|---|---|---|
| DNABERT | 겹침 k-mer | BERT식 MLM | 짧음 | 프로모터·스플라이스·TF 결합 부위 미세조정 |
| DNABERT-2 | BPE | MLM, ALiBi, FlashAttention | 길이 제한 완화 | 다종(multi-species) 사전학습, GUE 벤치마크(36개 데이터셋·9개 과제, 4종) |
| Nucleotide Transformer (v1) | 6-mer(가능하면) + 단일 뉴클레오타이드 | MLM (ESM 구조) | 1,000 토큰 ≈ 6 kb | 사람 참조 게놈 / 3,200개 사람 게놈 / 다종 등 학습 데이터 변형 (모델 카드) |
| HyenaDNA | 단일 뉴클레오타이드 | Hyena(암묵적 합성곱) | 최대 1M 토큰 | 서브쿼드라틱 비용 |
| Caduceus | 단일 뉴클레오타이드 | BiMamba + RC 등변 | 최대 131k† | CUDA 환경 필요† |
| GPN / GPN-MSA | 단일 뉴클레오타이드 / MSA 입력 | MLM | — | GPN은 *Arabidopsis*에서 변이 효과 예측. GPN-MSA는 다종 정렬을 입력으로 사용 |
| Evo / Evo 2 | 단일 뉴클레오타이드 | 자기회귀 | 최대 1M (Evo 2) | Evo 2는 1B/7B/40B†, 저자 보고로 BRCA1 변이를 학습 없이 예측 |

**Evo 2를 직접 돌릴 때의 요건** (공식 README 기준, 2026-11 확인): CUDA 12.1+ NVIDIA GPU, Python 3.11/3.12.
`evo2_1b_base`·40B는 **FP8(Transformer Engine)이 필요하고 Hopper 계열 GPU**가 요구된다. `evo2_7b`는 bf16으로 Transformer Engine 없이 실행 가능하다고 한다.
→ 무료 Colab GPU(T4)로는 어렵다. 그래서 이 모듈은 Evo 2를 직접 실행하는 노트북 대신 **TraitGym이 공개한 Evo 2 사전 계산 점수**를 노트북 02에서 비교한다. 직접 실행은 과제 5(선택, 🔴).

### 2.2 지도 sequence-to-function 모델
| 모델 | 입력 | 출력 | 비고 |
|---|---|---|---|
| DeepSEA | 짧은 창 | 염색질 프로파일 | 비코딩 변이 효과 예측의 초기 기준선 |
| Sei | 긴 창 | 21,907 염색질 프로파일에서 학습한 "sequence class" | 기준선 |
| Enformer | 196,608 bp | 128 bp 단위 트랙 (중앙 896 bins ≈ 114,688 bp) × 사람 5,313 / 생쥐 1,643 | DNase/ATAC, 히스톤·TF ChIP, CAGE 등 |
| Borzoi | 약 524 kb | RNA-seq 커버리지 등 32 bp 단위 | Enformer 확장 |
| AlphaGenome | 1 Mb | 다양한 게놈 트랙 + 변이 효과 | 2026 Nature |

**같은 순위표에 놓지 말 것**: Enformer류는 **실험 트랙을 직접 예측하도록 지도학습**되고, DNA LM은 **서열 분포만 자기지도 학습**한다. 규모만으로 순위를 매기거나 한쪽의 약점을 다른 쪽에 적용하면 안 된다.

## 3. 토크나이징과 문맥 — 모듈 01의 복습
- NT v1은 서열을 **겹치지 않는 6-mer**로 자르고, 6으로 나누어 떨어지지 않는 끝부분은 단일 뉴클레오타이드로 처리한다. → 모듈 01 노트북 01에서 본 **프레임 이동 민감도**가 그대로 적용된다: 시작 위치를 1칸 바꾸면 토큰이 전부 달라진다.
- 한 SNV는 **정확히 한 개의 6-mer 토큰**을 바꾼다 (비겹침 토큰이므로). 따라서 NT의 LLR은 "토큰 하나를 마스크했을 때 alt 6-mer와 ref 6-mer의 로그확률 차이"로 정의할 수 있다.
- 위치 임베딩이 절대 위치(NT v1 설정 기준)이고 학습 길이가 1,000 토큰이므로, 학습 길이를 넘는 서열로의 외삽은 보장되지 않는다.

## 4. 임베딩 + linear probe: 프로모터 예측

> **실습 [01_nt_embeddings_promoter_probe](notebooks/01_nt_embeddings_promoter_probe.ipynb)**

- 과제: 300 bp 서열이 프로모터인지 분류 (NT 다운스트림 과제 `promoter_tata`, `promoter_no_tata`; 데이터는 HF에 공개된 `revised` 버전).
- 비교: **6-mer 빈도 + 로지스틱 회귀**(NT가 쓰는 토큰 단위를 빈도로만 쓴 가장 단순한 baseline) vs **NT 임베딩 + 로지스틱 회귀**.
- 주의: 이 데이터의 train/test split이 **염색체/반복 구간 중복을 얼마나 막는지**는 NT 논문·데이터 설명에서 확인하라†. 높은 점수가 쉬운 과제의 신호일 수 있다.

## 5. 변이 효과: zero-shot LLR과 TraitGym

### 5.1 LLR 정의
참조 대립유전자(ref)와 대체 대립유전자(alt)에 대해
$$\mathrm{LLR}=\log p(\text{alt}\mid \text{context})-\log p(\text{ref}\mid \text{context})$$
DNA LM이 "자연 서열 분포"에서 변이가 얼마나 덜 그럴듯한지를 점수화한다. 부호와 절대값을 모두 쓴다 (`LLR`, `absLLR`) — 해로운 변이는 ref가 alt보다 확률이 높을 수 있고(음수 LLR), 방향에 관계없이 "특이한" 변이일 수도 있으므로 TraitGym은 두 변형을 모두 평가한다.

### 5.2 TraitGym
- 사람 **비코딩 조절 변이**의 인과 후보(멘델·복합형질)와 **매칭된 대조군**으로 구성한 벤치마크. 문제는 이진 분류이고 주 지표는 **AUPRC**다. 양성 : 대조 = 1 : 9인 `matched_9` 버전에서는 무작위 예측의 AUPRC가 양성 비율(약 0.1)이다.
- 저자 보고(초록): CADD·GPN-MSA 같은 **정렬 기반** 모델이 멘델 형질에서 유리하고, 기능유전체 지도학습 모델(Enformer, Borzoi)은 복합 비질병 형질에서 더 낫다. Evo 2는 규모와 함께 향상되지만 정렬 기반 모델에는 다소 못 미치고 인핸서 변이에서 특히 약하다고 한다. **우리는 이 주장을 노트북 02에서 일부 변이 집합으로 직접 확인한다.**
- **매칭된 대조군**: 대조군은 변이 유형, TSS 거리 등으로 양성과 맞춰져 있다. 그래서 TSS 거리 같은 단순 특징은 거의 정보가 없어야 한다 (노트북 02에서 확인).
- 보존 기반 점수(phyloP, phastCons)와 CADD는 학습 없이도 강한 기준선이다. **DNA LM이 이 기준선을 넘는지**가 핵심 질문이다.

> **실습 [02_zero_shot_variant_scores_traitgym](notebooks/02_zero_shot_variant_scores_traitgym.ipynb)**: (1) TraitGym이 공개한 모델별 사전 계산 점수를 평가하고, (2) NT v1(500M, 사람 참조)으로 chr1 변이의 LLR을 **직접 계산**해 같은 변이에서 비교한다.

## 6. Sequence-to-function: Enformer

- 196,608 bp 입력 → 128 bp 단위로 평균 낸 트랙 5,313개(사람)를 **중앙 896 bins**에 대해 예측. 합성곱으로 줄이고 transformer로 장거리 상호작용을 통합한다.
- **in silico mutagenesis(ISM)**: 서열의 한 염기를 바꿔 예측이 얼마나 변하는지로 변이 효과를 본다. TraitGym의 `Enformer_L2` 같은 점수는 이런 ref/alt 예측 차이를 요약한 값이다†.
- 한계: Huang et al. (2023)은 현재 모델들이 **개인 간 발현 변이**를 잘 설명하지 못하고 변이의 효과 방향도 자주 틀린다고 보고했다. 참조 게놈의 **유전자 간 차이**를 잘 맞히는 것과 **개인 간 cis 변이 효과**를 맞히는 것은 다른 과제다.

> **실습 [03_enformer_tracks](notebooks/03_enformer_tracks.ipynb)** (격리 환경 `.venv-enformer`): 유전자 TSS 주변 서열 → 예측 트랙 → 알려진 생물학과의 정합성 점검.

## 7. 비판적 평가의 교훈
| 연구 | 핵심 메시지 (저자 보고) |
|---|---|
| BEND | 현재 DNA LM의 임베딩은 일부 과제에서 전문가 방법에 근접하지만 **장거리 특징은 제한적으로** 담는다 |
| Huang et al. 2023 | 지도 sequence-to-function 모델도 **개인 간 발현 변이**를 설명하는 성능이 제한적이고, 효과 방향이 틀리기도 한다 |
| TraitGym | 정렬/보존 기반 모델이 강한 기준선이고, 자기지도 LM은 규모에 따라 개선되지만 아직 따라잡지 못하는 영역이 있다 |
| GPN-MSA (저자 보고) | 이전 사람 DNA LM(NT 등)은 단순 보존 점수보다 변이 효과 예측이 나빴다고 서술 |

읽는 법: 이 결과들은 **특정 모델·과제·데이터셋**에 대한 것이다. "DNA LM은 쓸모없다"가 아니라 "이 설정에서는 이 기준선을 넘지 못했다"로 기록하라 (모듈 00의 읽기 원칙).

## 8. RNA 모델 개관 (노트북 없음)
- **RNA-FM**: 비번역 RNA(RNAcentral의 약 2,300만 서열)로 자기지도 학습, 구조·기능 예측에 사용.
- **RiNALMo**: 6.5억 파라미터 RNA 언어모델, 약 3,600만 ncRNA로 학습, **본 적 없는 RNA 가족**에 대한 구조 예측 일반화가 핵심 주장 (저자 보고).
- **Ribonanza / RibonanzaNet**: 화학 매핑(chemical mapping) 측정 200만 서열 규모의 크라우드소싱 데이터와 그 위에서 학습한 모델.
- 공통 주의: RNA 구조 예측은 가족 간 일반화가 어려워 **split(가족 단위)**이 결과를 좌우한다 (모듈 01 5.2).

---

## 9. 이해도 점검

**Q1.** 비겹침 6-mer 토크나이저를 쓰는 NT에서 SNV 하나는 몇 개의 토큰을 바꾸는가? 겹침 6-mer(DNABERT)와 비교하라.
<details><summary>답</summary>
비겹침이므로 SNV가 속한 6-mer **하나**만 바뀐다. 겹침 6-mer에서는 그 염기를 포함하는 윈도 6개가 모두 바뀐다. 그래서 NT의 LLR은 토큰 하나의 로그확률 차이로 정의할 수 있다.
</details>

**Q2.** 같은 SNV라도 창(window)의 시작 위치를 바꾸면 NT 점수가 달라질 수 있는 이유와 완화책은?
<details><summary>답</summary>
비겹침 6-mer의 프레임이 바뀌면 변이가 6-mer 안의 다른 위치에 놓이고 문맥 토큰도 모두 달라진다(모듈 01의 프레임 이동 민감도). 시작 오프셋 0~5를 모두 계산해 평균하면 프레임 의존성을 줄일 수 있다.
</details>

**Q3.** LLR과 absLLR의 차이와, 둘 다 평가하는 이유는?
<details><summary>답</summary>
LLR은 부호가 있는 점수(alt가 ref보다 덜 그럴듯하면 음수)이고 absLLR은 방향과 무관한 크기다. 어떤 변이는 방향과 상관없이 "자연 분포에서 벗어난 정도"가 중요해서 absLLR이 더 잘 맞을 수 있다. 어느 쪽이 맞는지는 모델·과제에 따라 달라 둘 다 평가한다.
</details>

**Q4.** TraitGym에서 TSS 거리만으로 양성과 대조군을 거의 구분하지 못해야 하는 이유는?
<details><summary>답</summary>
대조군을 변이 유형, TSS 거리 등으로 양성과 맞춰 구성했기 때문이다. 이런 매칭은 단순한 교란 요인이 점수를 부풀리지 못하게 하려는 설계다. 매칭 변수로 높은 점수가 나오면 오히려 데이터 구성이나 코드 오류를 의심한다.
</details>

**Q5.** AUPRC에서 무작위 예측의 기대값은 얼마이고, AUROC와 무엇이 다른가?
<details><summary>답</summary>
무작위 예측의 AUPRC는 양성 비율(1:9 매칭이면 약 0.1)이다. AUROC의 무작위 기대값은 0.5로 클래스 비율과 무관하다. 양성이 드물 때는 AUPRC가 양성을 상위에 놓는 능력을 더 민감하게 반영한다.
</details>

**Q6.** NT 프로모터 분류에서 6-mer 빈도 baseline이 임베딩 probe와 비슷하다면 무엇을 의미할 수 있는가?
<details><summary>답</summary>
과제가 6-mer 수준의 서열 조성만으로 충분히 풀릴 만큼 쉽거나(예: GC/모티프 조성 차이), 임베딩이 그 이상의 정보를 선형으로 추가하지 못한다는 뜻이다. 데이터 split의 누출 가능성, probe 용량, 층 선택도 함께 점검한다.
</details>

**Q7.** 지도 sequence-to-function 모델과 자기지도 DNA LM을 같은 표에서 비교할 때 지켜야 할 점은?
<details><summary>답</summary>
학습 데이터(실험 트랙 vs 서열만)와 목적이 다르므로 규모나 순위로 우열을 단정하지 않는다. 같은 변이 집합, 같은 지표, 같은 split에서 평가하고 보존 기반 기준선을 함께 보고하며, 어떤 변이 유형에서 강/약한지 층화해서 본다.
</details>

**Q8.** Enformer 출력이 입력보다 짧은(중앙 896 bins ≈ 114,688 bp) 이유는?
<details><summary>답</summary>
가장자리 서열은 양쪽 문맥이 부족해 신뢰하기 어렵기 때문에 중앙 영역만 예측 대상으로 삼는다. 따라서 관심 변이를 평가하려면 변이가 출력 구간 안에 있어야 한다. 입력 길이와 출력 범위는 모델 문서에서 확인한다†.
</details>

**Q9.** Huang et al.(2023)의 결과가 "Enformer가 쓸모없다"는 뜻이 아닌 이유는?
<details><summary>답</summary>
이 연구는 **개인 간 cis 변이에 의한 발현 차이**라는 특정 과제를 평가했다. 유전자 간·세포 유형 간 발현 패턴, 조절 요소 주석, 변이 우선순위 등 다른 과제에서의 성능은 별개다. 다만 모델 점수를 개인 수준 효과로 해석하는 데 주의가 필요하다는 경고다.
</details>

**Q10.** Evo 2를 직접 돌리는 대신 공개된 사전 계산 점수를 쓰면 어떤 것이 보장되지 않는가?
<details><summary>답</summary>
점수 계산의 세부(창 길이, 앵커링, 리버스 컴플리먼트 처리, 모델 변형)를 우리가 통제하지 못한다. 공개 점수는 벤치마크와 동일한 변이 목록에 대해 계산된 값이지만, 방법의 모든 선택을 재현·변경하며 실험할 수는 없다.
</details>

---

## 10. 읽기 (reading_list 03·05 구역)
| 필수도 | 항목 | 주차 |
|---|---|---|
| ★ | DNABERT, Dalla-Torre 2025(NT) | 7 |
| ★ [기준선] | Zhou & Troyanskaya 2015(DeepSEA) | 7 |
| ◆ | DNABERT-2, HyenaDNA, Caduceus, Benegas 2023(GPN) | 7 |
| ★ [벤치마크][비판] | Benegas 2025(TraitGym), Marin 2024(BEND), Huang 2023 | 8~9 |
| ◆ | Benegas(GPN-MSA, Nat Biotechnol 2025) | 8 |
| ★ | Nguyen 2024(Evo), ◆ Brixi 2026(Evo 2) | 8 |
| ★ [지도] | Avsec 2021(Enformer) / ◆ Linder 2025(Borzoi) / ◆ Avsec 2026(AlphaGenome) | 9 |
| ○ [RNA] | RNA-FM, RiNALMo, Ribonanza | 9 |

## 11. 노트북
| 노트북 | 등급 | 내용 | 다운로드 |
|---|---|---|---|
| [01_nt_embeddings_promoter_probe](notebooks/01_nt_embeddings_promoter_probe.ipynb) | 🟢 | NT 임베딩 + 프로모터 probe vs 6-mer baseline | NT 500M(1.9 GB), NT 과제 데이터(약 6 MB) |
| [02_zero_shot_variant_scores_traitgym](notebooks/02_zero_shot_variant_scores_traitgym.ipynb) | 🟢 | TraitGym 공개 점수 평가 + NT LLR 직접 계산 | TraitGym(약 3 MB), hg38 chr1(75 MB), NT 500M |
| [03_enformer_tracks](notebooks/03_enformer_tracks.ipynb) | 🟢* | Enformer 예측 트랙, 생물학 정합성 점검 | Enformer(1.0 GB), 트랙 메타데이터(0.8 MB), UCSC API |

**참고 결과** (Mac M1 Pro 32GB, MPS, Python 3.12, torch 2.15.0; 노트북 03은 transformers 4.56.2 격리 환경 — 버전·장치에 따라 달라질 수 있음)
| 노트북 | 확인 포인트 | 기대 값(대략) | 소요 |
|---|---|---|---|
| 01 | `promoter_no_tata` test AUROC: 6-mer 빈도 / NT 18층 / NT 24층 | 0.936 / 0.946 / 0.944 | 약 7분 |
| 01 | `promoter_tata` (test 212개): 6-mer 빈도 / NT 18층 / NT 24층 | 0.925 / 0.942 / 0.946 (구간이 겹쳐 우열 불가) | |
| 01 | 앞 1~5 nt 제거 / reverse complement 시 NT probe AUROC | 0.938~0.941 / 0.930 (원래 0.946) | |
| 02 | TraitGym 멘델 3,380개 AUPRC (무작위 0.10): CADD / GPN-MSA / phyloP-241m | 0.69 / 0.69 / 0.67 | 약 2분 |
| 02 | 같은 표: Evo 2 1B / 7B / 40B, TraitGym NT LLR | 0.15 / 0.38 / 0.53, 0.10 (무작위 수준) | |
| 02 | chr1(양성 21개) 직접 계산 NT LLR (6프레임×정/역방향) | 0.14 [0.10, 0.23] — 무작위와 구별 안 됨 | NT LLR 계산 약 80초 |
| 03 | GAPDH TSS 주변 K562 CAGE 예측 비율(원래 / 섞은 서열) | 61.9 / 2.65 | 약 8.5분 (예측 1회 약 6초) |
| 03 | ISM \|Δ\| 평균: TSS ±30 bp vs +40 kb | 2.59 vs 0.056 (약 46배) | |

\* 별도 환경 `.venv-enformer` 필요 (`requirements/enformer.txt`, 이유는 SETUP.md). 데이터·모델은 노트북이 내려받으며 저장 위치는 노트북 상단에 표시된다. 과제: [exercises.md](exercises.md)

## 다음
→ [04_single_cell](../04_single_cell/)
