# 02. Protein — 단백질 언어모델, 변이 효과, 구조, 설계

**기간**: 3~6주차 (4주, 약 32~40시간) · **실행 등급**: 🟢 Mac 로컬 3종 + 🟡 Colab 1종
**전제**: [모듈 01](../01_foundations/)의 토크나이징·목적함수·split 설계(특히 5절 평가의 함정).

> 표기: †는 논문 본문이나 모델 카드에서 직접 확인해야 하는 서술이다. 인용한 논문은 [reading_list](../resources/reading_list.md) 02 구역에 있다. 이 모듈의 수치와 모델 정보는 2026-11 시점의 것이다. 새 모델이 빠르게 나오므로 사용하기 전에 모델 카드를 확인한다.

---

## 학습 목표
1. 단백질 언어모델(PLM)의 학습 방식과 계보(ESM-1b → ESM-2 → ESM C, ProGen 계열, 구조 인지 모델)를 비교한다.
2. PLM 임베딩으로 **linear probe**를 만들고, **baseline·split**을 갖춰 공정하게 평가한다.
3. **zero-shot 변이 효과 점수**(masked marginal)를 직접 계산하고 ProteinGym 방식(Spearman)으로 평가한다.
4. 구조 예측 모델(AlphaFold 2/3, ESMFold)의 입출력과 **pLDDT 등 신뢰도 지표**를 해석한다.
5. attention과 SAE로 PLM 내부를 들여다보는 연구의 **가능성과 한계**를 설명한다.
6. 설계 모델(inverse folding, 확산, 생성 LM)의 역할을 구분하고, in silico 평가와 실험 검증의 차이를 설명한다.

## 4주 일정
| 주 | 주제 | 노트북 | 읽기(★ 중심) |
|---|---|---|---|
| 3 | PLM 기초: 표현, probe | [01](notebooks/01_esm2_embeddings_probe.ipynb) | ESM-1b(Rives 2021), ESM-2(Lin 2023) |
| 4 | 변이 효과 zero-shot, 벤치마크 | [02](notebooks/02_zero_shot_variant_effect.ipynb) | ESM-1v, ProteinGym, EVE, AlphaMissense |
| 5 | 구조 예측과 해석 | [03](notebooks/03_attention_contact_map.ipynb), [04](notebooks/04_esmfold_plddt_colab.ipynb) | AlphaFold 2·3, Rao 2021, InterPLM |
| 6 | 설계 + 모듈 산출물 | (읽기·정리) | ProteinMPNN, RFdiffusion, ESM-IF1, ESM3 |

**모듈 산출물**: ProteinGym의 한 assay에서 단순 baseline(one-hot, 아미노산 조성)과 ESM-2 임베딩 probe를 **random split과 위치 단위 split** 두 가지로 비교한 표와 그 해석이다([exercises.md](exercises.md) 과제 1).

---

## 1. 데이터와 사전학습

- 단백질 서열 DB는 **계통·종 표본 편향이 크고 중복이 많다**. 사전학습 코퍼스는 보통 서열 동일도로 클러스터링한 대표 서열 집합(예: UniRef 계열)을 쓴다. ESM-2 체크포인트 이름의 `UR50D`가 가리키는 데이터 구성은 모델 카드와 논문에서 확인한다†.
- ESM-1b 논문은 약 2.5억 서열(860억 아미노산)로 학습한 모델에서 2차·3차 구조 정보가 표현에 담기고, 선형 사영으로 읽어낼 수 있다고 보고했다(초록 기준).
- **평가 서열과 사전학습 서열이 겹칠** 수 있다. 모듈 01 5.2 표의 "공통" 행에 해당하는 문제다. ProteinGym 결과는 DMS 대상 단백질이 사전학습 코퍼스에 들어 있을 수 있다는 점을 염두에 두고 해석한다.

## 2. 모델 계보와 역할

| 계열 | 대표 모델 | 입력/목적 | 용도 | 비고 |
|---|---|---|---|---|
| MLM PLM | ESM-1b, ESM-2 (8M~15B), ProtTrans | 단일 서열, MLM | 임베딩, 변이 점수 | ESM-2: 모델을 최대 15B까지 키우면 원자 수준 구조 정보가 표현에 드러난다고 보고 (Lin 2023 초록) |
| MLM PLM (신) | ESM C | 단일 서열, MLM | 임베딩 | 2024-12 공개, 논문이 아닌 기술 블로그·모델 카드. 가중치 라이선스는 모델 카드에서 확인 |
| 구조 인지 PLM | SaProt | 잔기 + Foldseek 구조 토큰 | 임베딩, 변이 점수 | 구조가 필요 |
| 자기회귀 PLM | ProGen, ProGen2 | 단일 서열, 다음 토큰 | 생성, likelihood | 제어 태그 기반 생성 (ProGen) |
| 다중 트랙 생성 | ESM3 | 서열·구조·기능 등† | 조건부 생성 | 상세는 논문 확인† |
| 구조 예측 | AlphaFold 2, ESMFold, AlphaFold 3, ESMFold2, Boltz-1/2, Chai-1 | 서열(+MSA)(+리간드·핵산) | 3D 좌표 + 신뢰도 | AF2는 MSA 기반, ESMFold는 단일 서열 + PLM, AF3는 확산 기반 복합체 예측 |
| 설계(서열) | ProteinMPNN, ESM-IF1 | 백본 구조 → 서열 (inverse folding) | 서열 설계 | ESM-IF1은 AF2 예측 구조로 학습 데이터를 늘림 |
| 설계(백본) | RFdiffusion | 확산 모델 | 백본/기능 설계 | 설계, ProteinMPNN, 구조 재예측으로 이어지는 파이프라인이 흔하다 |
| 변이 효과 | ESM-1v, EVE, AlphaMissense | 서열 분포/진화 정보 | 병원성·fitness 예측 | EVE는 MSA 기반 생성 모델, 라벨 없이 학습 |
| 벤치마크 | TAPE, FLIP, PEER, ProteinGym | — | 평가 | ProteinGym: 논문 초록 기준 250개 이상의 DMS assay + 임상 데이터. 이 모듈에서 내려받는 v1.3 치환 파일에는 217개 assay가 들어 있다 |

**ESMFold2(2026)** 는 Biohub(ESM 팀)에서 공개한 구조 예측 모델이다. 모델 카드에 따르면 소분자·DNA·RNA를 포함한 모든 생체분자를 다루고, MSA 입력은 선택이며, pLDDT·pAE·pTM·iPTM을 출력한다. 가중치가 수십 GB라 로컬 실습 범위를 넘는다. 근거 논문은 bioRxiv 프리프린트(reading_list 02)이고, 성능 주장은 저자가 보고한 것이다.

---

## 3. 표현을 평가하기: 임베딩과 linear probe

### 3.1 임베딩 추출
- 서열을 잔기 단위 토크나이저에 넣으면 `<cls>`/`<eos>` 같은 특수 토큰이 붙고, 모델은 층마다 hidden state를 낸다.
- **서열 단위 임베딩**: 잔기 위치 평균(mean pooling). 평균에는 **패딩과 `<cls>`/`<eos>` 특수 토큰을 제외**해야 한다. attention mask는 **패딩만** 걸러 주고 `<cls>`/`<eos>`는 mask=1이라는 점에 주의한다. 그래서 특수 토큰 위치는 따로 0으로 만들어야 한다(노트북 01의 `embed()` 참고). 길이가 다른 서열이 섞이면 특히 중요하다.
- **어느 층을 쓸까**: 마지막 층이 항상 최선은 아니다. 중간 층이 더 좋은 과제도 있어 층별로 비교한다†. 노트북 01의 `embed(..., layer=)`는 `hidden_states` 인덱스를 받는다. 인덱스 0은 임베딩 층 출력이고, 마지막 인덱스는 `last_hidden_state`와 같음을 노트북에서 확인한다. 중간 층의 값은 최종 LayerNorm 이전 값이라 척도가 다를 수 있다†.

### 3.2 Linear probe 설계 원칙
1. 임베딩은 **고정**하고 선형 모델(Ridge/로지스틱)만 학습한다. 이렇게 하면 표현 자체의 품질을 볼 수 있다.
2. **baseline을 먼저** 만든다. one-hot(위치 × 잔기)과 아미노산 조성을 쓰고, 가능하면 BLOSUM 기반 점수도 더한다.
3. **split을 목적에 맞게** 고른다. DMS 변이 데이터에서 random split을 하면 같은 위치의 다른 치환이 train에 들어간다. 그래서 **새 위치**에 대한 일반화를 말하기에는 낙관적인 평가가 된다. 반대로 목적이 "이미 측정된 위치의 **미측정 치환**을 채우는 것"이라면 random split이 그 목적에 맞는 평가일 수 있다. 목적을 먼저 정하고 그에 맞는 split(새 위치 일반화라면 **위치 단위/연속 블록 split**)을 쓴다.
4. 규제 강도는 **train 안에서** 선택하고 test는 한 번만 본다.

> **실습 [01_esm2_embeddings_probe](notebooks/01_esm2_embeddings_probe.ipynb)**

---

## 4. Zero-shot 변이 효과 예측

모듈 01 3.1의 점수 정의를 PLM에 적용한다.

| 방식 | 계산 | forward 횟수 | 메모 |
|---|---|---|---|
| masked marginal | 위치 $i$를 마스크 → $\log p(\text{mut})-\log p(\text{wt})$ | 서열 길이 L (위치당 1회) | 단일 치환의 표준적 방식 |
| wild-type marginal | 마스크 없이 wt 서열 1회 → 같은 식 | 1 | 가장 싸지만 wt 잔기가 입력에 보임 |
| pseudo-log-likelihood 비율 | 변이/wt 서열 각각의 PLL | 변이 수 × L | 비쌈, 다중 변이에 확장 가능 |

Meier et al. 2021이 여러 점수 방식을 비교한다. 세부는 논문에서 확인한다†.

### 4.1 평가: ProteinGym 방식
- assay마다 **Spearman 상관**(점수 vs DMS 점수)을 구하고 평균한다. 노트북 02의 평균은 선택한 7개 assay의 **단순 평균**일 뿐이다. 유형별 집계 등을 거치는 ProteinGym 공식 집계 절차와 같지 않으므로, 공식 점수와 비교할 때는 공식 문서를 확인한다†. 같은 모델도 assay 유형(안정성, 활성, 결합, 발현, 적합도 등)에 따라 성능이 크게 다르다. 그러므로 **평균 하나로 판단하지 말고 유형별로 본다**.
- DMS 측정에는 **노이즈가 있으므로** 상관의 상한이 1이 아니다.
- 단일 치환과 다중 치환은 구분한다. 위 표의 masked marginal은 단일 치환 기준이다. 점수를 합산해 다중 치환으로 확장하면 상위성을 무시하게 된다.
- 임상 변이 병원성 예측(AlphaMissense, EVE)은 DMS 적합도 예측과 **목표가 다르다**. 같은 지표로 비교하지 않는다.

> **실습 [02_zero_shot_variant_effect](notebooks/02_zero_shot_variant_effect.ipynb)**: 모델 크기(8M~650M) × 여러 assay.

---

## 5. 구조 예측과 신뢰도

### 5.1 모델 비교 (개념)
| 모델 | 입력 | 핵심 | 신뢰도 출력 |
|---|---|---|---|
| AlphaFold 2 | 서열 + MSA | MSA와 구조 모듈을 결합한 end-to-end 예측 | pLDDT, PAE, pTM |
| ESMFold | 단일 서열 | ESM-2 표현 위에 접힘 모듈. AF2 대비 매우 빠르며 PLM이 잘 이해하는(낮은 perplexity) 서열에서 정확도가 비슷하다고 보고 (HF 문서의 초록 기준) | pLDDT, PAE, pTM (HF 구현 출력 키로 확인) |
| AlphaFold 3 | 서열·리간드·핵산 등 | 확산 기반 구조 생성으로 복합체 예측 | pLDDT, PAE, ipTM† |
| ESMFold2 | 서열(+MSA 선택) | 모델 카드 참고 | pLDDT, pAE, pTM, iPTM |

### 5.2 pLDDT 읽는 법
- pLDDT는 잔기별 예측 신뢰도다. 모델에 따라 0~1 또는 0~100 척도를 쓰므로 **척도를 확인한다**. 값이 낮은 영역은 내재적 무질서이거나 예측이 불확실한 곳이다.
- 높은 pLDDT는 **구조가 맞다는 보장이 아니고**, 안정성·활성과도 직접 같지 않다. 변이 효과 예측에 pLDDT 변화를 쓰는 접근은 한계가 있다†. 과제 5에서 직접 검증한다.
- PAE와 pTM은 도메인 간 상대 배치와 전체 폴드의 신뢰도를 나타낸다. 도메인 배치는 pLDDT만으로 알 수 없다.

> **실습 [04_esmfold_plddt_colab](notebooks/04_esmfold_plddt_colab.ipynb)** 🟡: 가중치가 커서 Colab GPU를 권장한다. Colab T4에서 처음부터 끝까지 실행해 확인했다(2026-11-06, 단일 환경). 메모리에 주의한다. 가중치를 CPU RAM으로 먼저 불러오면 12 GB 런타임에서 커널이 죽을 수 있어서, 노트북은 가중치를 GPU로 바로 올린다. 이때 GPU 메모리를 순간 최대 약 14 GB 쓴다. 자세한 내용은 노트북 상단에 있다.

---

## 6. 내부 들여다보기: attention과 SAE

- **Attention → contact** (Rao et al. 2021): 언어모델의 attention map이 비지도 MLM 목표만으로 잔기 접촉을 학습한다. 초록에 따르면 용량이 큰 모델은 비지도 contact 예측 파이프라인을 앞섰다. 단일 head의 raw attention을 그대로 쓰기보다 **대칭화와 APC 같은 보정**을 흔히 거치며, 최종 contact 예측에는 지도 학습된 probe를 쓰는 경우가 많다†.
- **Sparse autoencoder (InterPLM)**: ESM-2 임베딩에 SAE를 학습해 층마다 수천 개의 해석 가능한 feature를 얻고 일부는 결합 부위·구조 모티프 등과 대응한다. 초록에 따르면 개별 뉴런보다 개념 정렬이 낫고, 더 큰 PLM(650M)이 8M보다 더 많은 개념을 담는다. **feature의 인과적 역할은 별도로 검증해야** 한다.
- 해석 연구에는 공통된 주의점이 있다. 상관은 인과가 아니고, 사후적 라벨링에는 편향이 있으며, 합성 설정에서 잘 되는 결과는 일반화에 한계가 있다(모듈 01 노트북 02 참고).

> **실습 [03_attention_contact_map](notebooks/03_attention_contact_map.ipynb)**: 소형 단백질 구조(PDB)와 비교.

---

## 7. 설계 모델의 역할 구분

| 질문 | 모델군 | 입력 → 출력 |
|---|---|---|
| 이 서열은 어떤 구조인가? | 구조 예측 | 서열 → 좌표 |
| 이 백본에 맞는 서열은? | inverse folding (ProteinMPNN, ESM-IF1) | 백본 → 서열 |
| 새 백본/구조를 만들려면? | 확산 모델 (RFdiffusion) | 조건 → 백본 |
| 서열 자체를 생성하려면? | 생성 LM (ProGen, ESM3) | 조건/프롬프트 → 서열 |

**in silico 평가**로는 설계한 서열을 구조 예측기로 다시 접어 목표 구조와 비교하는 self-consistency와 예측 신뢰도(pLDDT)가 흔히 쓰인다. 그러나 이것은 발현·접힘·기능 같은 **실험적 성공의 대리 지표**일 뿐이고, 예측기와 설계기가 같은 편향을 공유할 수도 있다. 생물안전·데이터 라이선스 문제는 모듈 06에서 다룬다.

---

## 8. 이해도 점검

**Q1.** ESM-2 서열 임베딩을 평균할 때 `<cls>`, `<eos>`, 패딩 토큰을 왜 제외해야 하는가?
<details><summary>답</summary>
특수 토큰과 패딩은 서열 정보가 아니다. 길이가 다른 서열을 배치로 묶으면 패딩 수가 서열마다 달라서, 평균이 길이와 배치 구성에 따라 왜곡된다. attention mask는 패딩만 0으로 만들고 `<cls>`/`<eos>`는 1이므로, 평균에서 이 두 위치를 별도로 제외해야 실제 잔기 위치만 평균된다. (토크나이저의 특수 토큰 구성은 모델 카드에서 확인†)
</details>

**Q2.** DMS 데이터에서 변이 단위 random split이 낙관적인 이유와, 새 위치 일반화를 평가하는 split은?
<details><summary>답</summary>
같은 위치의 다른 치환 변이가 train에 있으면, 모델이 위치별 평균 효과를 외우기만 해도 점수가 높게 나온다(이웃 변이 누출). 위치나 연속 구간 단위로 분할해 test 위치가 train에 전혀 없게 한다. 목적이 "새 변이 예측"인지 "새 위치/새 단백질 예측"인지에 따라 split을 고른다.
</details>

**Q3.** masked marginal과 wild-type marginal의 계산량과 차이는?
<details><summary>답</summary>
masked marginal은 위치마다 마스크한 서열로 forward를 한 번씩(L회) 수행해 해당 위치 분포를 얻는다. wild-type marginal은 마스크 없이 forward 1회로 모든 위치의 분포를 얻으므로 훨씬 싸다. 다만 예측하려는 wt 잔기가 입력에 노출되므로 분포가 달라진다. 성능 차이는 벤치마크로 확인한다.
</details>

**Q4.** ProteinGym에서 모델의 평균 Spearman이 같아도 두 모델이 다르게 쓰여야 할 수 있는 이유는?
<details><summary>답</summary>
assay 유형(안정성, 활성, 결합, 발현 등)별 성능이 다를 수 있고, 같은 평균이어도 강점이 다르다. 목적과 같은 유형에서의 성능, 분산, 다중 치환 처리 능력 등을 따로 본다.
</details>

**Q5.** pLDDT가 높은 영역에서 변이 한 개가 안정성을 크게 떨어뜨리는지 pLDDT로 알 수 있는가?
<details><summary>답</summary>
알 수 없다고 봐야 한다. pLDDT는 예측 구조의 국소 신뢰도이며, 변이가 접힘 자유에너지에 미치는 효과를 직접 측정하지 않는다. 변이 효과를 알려면 zero-shot 점수나 안정성 예측기 같은 별도 방법과 실험 검증이 필요하다. 과제 5에서 pLDDT 변화와 DMS의 상관을 직접 확인한다.
</details>

**Q6.** ESMFold가 AlphaFold 2보다 빠른 이유와 정확도에서 주의할 점은?
<details><summary>답</summary>
MSA 검색·처리 없이 단일 서열의 PLM 표현을 입력으로 쓰기 때문에 훨씬 빠르다. 정확도는 PLM이 잘 이해하는(낮은 perplexity) 서열에서 AF2와 비슷하다고 보고됐다. 다만 깊은 MSA가 도움이 되는 서열처럼 PLM이 잘 모르는 경우에는 격차가 날 수 있다. 사용 전 대상 서열의 perplexity/신뢰도를 확인한다.
</details>

**Q7.** inverse folding, 구조 예측, 백본 생성(확산)을 입력과 출력으로 구분해 보자.
<details><summary>답</summary>
구조 예측: 서열 → 좌표. inverse folding: 백본 → 서열. 백본 생성: 조건(모티프, 결합 대상 등) → 새 백본 좌표. 설계 파이프라인은 백본 생성 → inverse folding → 구조 재예측 순서로 이어 붙이는 경우가 많다.
</details>

**Q8.** 설계 서열의 self-consistency(재접힘) 성공이 실험 성공을 보장하지 않는 이유는?
<details><summary>답</summary>
구조 예측기에도 학습 데이터의 편향과 오류가 있고, 설계기와 평가기가 비슷한 가정을 공유할 수 있다. 발현량, 용해도, 실제 접힘 경로, 기능 활성은 예측 범위 밖이다. 그러므로 wet-lab 검증이 필요하다.
</details>

**Q9.** attention map에서 contact를 읽을 때 대칭화와 APC 보정을 하는 이유는?
<details><summary>답</summary>
contact는 잔기쌍 사이의 대칭 관계인 반면, attention은 쿼리에서 키로 가는 방향이 있다. 또 특정 위치가 전반적으로 많이 attend되는 배경 편향도 있다. 대칭화와 APC 같은 보정은 이런 위치 편향을 줄여 쌍 신호를 드러낸다†(Rao 2021). 노트북 03의 "보정의 효과" 표에서 raw → 대칭화 → +APC 순으로 3개 단백질 평균 정밀도가 올랐다(예: 150M 0.03 → 0.23 → 0.36). 반면 대각선(자기 attention)을 0으로 만드는 변형은 일관된 차이가 없었다.
</details>

**Q10.** InterPLM류 SAE 연구에서 "feature가 결합 부위와 대응한다"는 결과를 해석할 때 주의할 점은?
<details><summary>답</summary>
대응은 상관적 증거다. feature를 조작했을 때 모델 출력이 바뀌는지, 곧 인과 관계가 있는지는 별도 실험으로 확인해야 한다. 자동 해석이라는 라벨링 방법, 주석 DB의 불완전성, 임계값 선택이 결과를 좌우한다. 논문이 제시한 steering·annotation 보강 실험이 어디까지 지지하는지 본문에서 확인한다.
</details>

---

## 9. 읽기 (reading_list 02 구역)
| 필수도 | 항목 | 주차 |
|---|---|---|
| ★ | Rives 2021 (ESM-1b), Lin 2023 (ESM-2/ESMFold) | 3 |
| ★ | Meier 2021 (ESM-1v), Notin 2023 (ProteinGym) | 4 |
| ★ (변이 트랙) | Cheng 2023 (AlphaMissense) / ◆ Frazer 2021 (EVE) | 4 |
| ★ | Jumper 2021 (AF2), Abramson 2024 (AF3) + Addendum | 5 |
| ◆ | Rao 2021 (attention→contact), Simon & Zou 2025 (InterPLM) | 5 |
| ★ (설계 트랙) | Dauparas 2022 (ProteinMPNN) / ◆ Watson 2023 (RFdiffusion), Hsu 2022 (ESM-IF1) | 6 |
| ◆ | Hayes 2025 (ESM3), Madani 2023 (ProGen), SaProt | 6 |
| ◆ [벤치마크] | TAPE, FLIP, PEER | 3~4 |
| ○ [최신] | ESM C 모델 카드, ESMFold2 프리프린트 | 5 |

## 10. 노트북
| 노트북 | 등급 | 내용 | 다운로드 |
|---|---|---|---|
| [01_esm2_embeddings_probe](notebooks/01_esm2_embeddings_probe.ipynb) | 🟢 | ESM-2 임베딩 + Ridge probe, baseline, random vs 위치 split | ESM-2 8M·35M, ProteinGym |
| [02_zero_shot_variant_effect](notebooks/02_zero_shot_variant_effect.ipynb) | 🟢 | masked marginal 점수, 모델 크기별 Spearman, BLOSUM baseline | ESM-2 8M~650M, ProteinGym |
| [03_attention_contact_map](notebooks/03_attention_contact_map.ipynb) | 🟢 | attention → contact map, APC, 모델 크기별 precision | ESM-2, PDB 파일 |
| [04_esmfold_plddt_colab](notebooks/04_esmfold_plddt_colab.ipynb) | 🟡 | ESMFold로 접고 pLDDT 해석 | ESMFold v1 (약 8.4 GB) |

데이터와 모델은 노트북이 내려받는다. 저장 위치는 노트북 상단에 표시된다. 과제: [exercises.md](exercises.md)

**참고 결과.** Mac M1 Pro 32GB, MPS, Python 3.12, torch 2.15.0, transformers 5.22.1에서 얻은 값이며 버전·장치에 따라 달라질 수 있다.
| 노트북 | 확인 포인트 | 기대 값(대략) | 소요 |
|---|---|---|---|
| 01 | UBC9 test Spearman — random: B1 / B2(+위치) / ESM-2 35M 변이 위치 임베딩 | 0.42 / 0.70 / 0.63 | 약 1분 |
| 01 | UBC9 test Spearman — position-block: B1 / B2 / ESM-2 35M 변이 위치 임베딩 | 0.28 / 0.30 / 0.15 (ESM 특징이 baseline에 못 미침) | |
| 02 | 7 assay 평균 Spearman: BLOSUM62 / 8M / 35M / 150M / 650M | 0.24 / 0.21 / 0.28 / 0.42 / 0.45 | 약 2분 |
| 02 | 150M에서 wt marginal vs masked marginal: 평균 Spearman / assay당 forward 시간 | 0.414 vs 0.420 / 약 0.05초 vs 약 3.6초 (약 70배) | |
| 03 | contact head precision@L/5 (3 단백질 평균): 8M / 35M / 150M / 650M | 0.30 / 0.55 / 0.83 / 0.89 | 약 25초 |
| 03 | 비지도 attention 보정 (3 단백질 평균, 150M): raw / 대칭화 / +APC | 0.03 / 0.23 / 0.36 | |
| 03 | contact head, 서열 거리 ≥ 24 기준 (8M→650M) | 0.25 / 0.55 / 0.66 / 0.82 | |
| 04 | (Colab T4, Python 3.13, torch 2.14.1, transformers 5.22.1, 로드 방식을 바꾼 뒤 2회 포함 3회 실행에서 수치 동일) 평균 pLDDT — 유비퀴틴 / GB1 / 섞은 서열 / `(GS)n` | 90.5 / 89.0 / 42.8 / 27.3 | 약 1.5분 (가중치 캐시됨) |
| 04 | 실험 구조와 CA RMSD — 유비퀴틴(1UBQ) / GB1(1PGA) | 0.84 Å / 0.42 Å | |

## 다음
[03_dna_genome](../03_dna_genome/)으로 넘어간다.
