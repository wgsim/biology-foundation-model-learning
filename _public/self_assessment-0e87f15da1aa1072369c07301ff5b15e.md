# 자가진단 (ML 10문항 · 생물학 10문항)

**방법**: 답을 가리고 먼저 직접 써 본다. 그다음 `<details>`를 열어 비교한다. 채점 기준과 권장 경로는 [README.md](README.md#h-07a89dec)를 참고한다.

---

## A. ML

**A1.** 길이 L인 시퀀스에서 표준 self-attention의 시간 복잡도와 attention map 메모리 복잡도는?
<details><summary>답</summary>
attention 연산의 시간은 O(L²·d), attention map 메모리는 O(L²)이다. 레이어 전체로는 Q·K·V 선형 투영 비용 O(L·d²)가 더해져 O(L²d + Ld²)가 되며, 긴 시퀀스 L ≫ d에서는 L² 항이 지배한다. FlashAttention 계열은 정확한(exact) attention을 유지하면서 메모리 접근을 줄여 메모리 사용을 선형에 가깝게 만들지만, 연산량 자체는 여전히 L²에 비례한다.
</details>

**A2.** BERT식 masked language modeling(MLM)과 자기회귀(causal) 언어모델의 차이와, 각각이 자연스럽게 맞는 생물학 과제는?
<details><summary>답</summary>
MLM은 가려진 토큰을 양방향 문맥으로 예측하며, 임베딩 추출과 변이 효과의 masked-marginal 점수화 등에 맞는다. 자기회귀는 앞 토큰만 보고 다음 토큰을 예측하며, 서열 생성과 전체 서열의 log-likelihood 점수화에 맞는다. (예: ESM-2는 MLM, ProGen·Evo 계열은 자기회귀이다. reading_list를 참고한다)
</details>

**A3.** Perplexity의 정의와, 토크나이저가 다른 두 모델의 perplexity를 직접 비교하면 안 되는 이유는?
<details><summary>답</summary>
perplexity는 exp(토큰당 평균 음의 로그우도)이다. 토큰 단위가 다르면(단일 뉴클레오타이드 vs k-mer vs BPE) "토큰 하나"가 담는 정보량이 달라 값을 직접 비교할 수 없다. 비교하려면 염기/잔기 단위로 환산(bits per base 등)해야 한다. 단, 환산만으로 항상 공정해지지는 않는다. 같은 서열·같은 평가 범위·같은 확률 정의여야 하며, MLM의 masked/pseudo-likelihood와 자기회귀 모델의 정규화된 서열 likelihood는 서로 다른 양이다.
</details>

**A4.** LoRA는 무엇을 고정하고 무엇을 학습하는가?
<details><summary>답</summary>
사전학습 가중치 W는 고정하고, 저랭크 행렬 B(d×r), A(r×k)의 곱 ΔW = BA만 학습한다(r ≪ d). 학습 파라미터 수와 optimizer 메모리가 크게 준다.
</details>

**A5.** Linear probe와 fine-tuning은 각각 무엇을 평가/최적화하며, 사전학습 표현의 품질을 비교할 때 왜 probe를 자주 쓰는가?
<details><summary>답</summary>
Probe는 표현을 고정하고 그 위에 선형 층만 학습하므로, 표현에 정보가 선형적으로 담겼는지를 측정한다. 공정한 표현 비교에 유리하고 하이퍼파라미터 영향이 적다. Fine-tuning은 전체(또는 일부) 가중치를 갱신해 최종 성능을 높이지만 과적합·재학습 비용·모델 간 비교 변동이 크다.
</details>

**A6.** 생물학 서열 데이터에서 random train/test split이 성능을 과대평가하는 이유는?
<details><summary>답</summary>
서열은 진화적으로 서로 관련(상동)되어 서로 독립적인 표본이 아니다. random split이면 테스트 서열의 가까운 친척이 훈련 세트에 들어 있어 모델이 새 가족이 아닌 "아는 가족의 새 구성원"에서 점수를 얻는다(기억만으로도 높은 점수가 가능). 새 가족에 대한 일반화가 목표일 때 과대평가이며, 알려진 가족의 새 구성원 예측이 목적이라면 그 목적에는 맞는 평가일 수 있다. 이 과정 모듈 01에서 직접 실험한다.
</details>

**A7.** Pearson과 Spearman 상관의 차이, 변이 효과 예측 벤치마크에서 Spearman을 자주 쓰는 이유는?
<details><summary>답</summary>
Pearson은 선형 관계를, Spearman은 순위(단조 관계)를 기반으로 한다. 실험 측정값(fitness 등)의 척도·분포가 assay마다 달라 점수 척도가 일치하지 않기 때문에 순위만 보는 Spearman이 흔히 쓰인다.
</details>

**A8.** 과적합을 학습/검증 곡선에서 어떻게 알아보고 어떻게 완화하나?
<details><summary>답</summary>
train loss는 계속 내려가는데 validation loss가 오르기 시작하면 과적합이다. 조기 종료, 정규화(dropout/weight decay), 데이터 증가·증강, 모델 축소로 완화한다.
</details>

**A9.** InfoNCE 대조학습 손실의 직관은?
<details><summary>답</summary>
배치 안에서 양성 쌍(예: 이미지–캡션)의 유사도가 음성 쌍들보다 커지도록 하는 softmax 교차엔트로피다. 온도 τ가 분포의 뾰족함을 조절한다. (CLIP, CONCH 같은 모델의 기반이다)
</details>

**A10.** bf16과 fp16의 차이는?
<details><summary>답</summary>
bf16은 지수부가 fp32와 같아 표현 가능한 범위가 넓어 loss scaling이 대체로 필요 없지만 가수부 정밀도가 낮다. fp16은 가수부 정밀도는 더 높지만 범위가 좁아 overflow/underflow 방지용 loss scaling이 필요하다.
</details>

---

## B. 생물학

**B1.** 코돈은 몇 개이며 아미노산은 몇 종인가? 이 관계를 무엇이라 하는가?
<details><summary>답</summary>
코돈 64개(정지 코돈 3개 포함)가 아미노산 20종을 지정하며, 이를 유전암호의 중복성(degeneracy)이라 한다. 같은 아미노산을 지정하는 동의 코돈이 존재하므로 synonymous 변이가 가능하다.
</details>

**B2.** 5'-ATGC-3'의 reverse complement는? DNA 모델 설계에서 이것이 왜 중요한가?
<details><summary>답</summary>
5'-GCAT-3'이다. 두 서열은 같은 이중가닥 분자를 두 가닥에서 읽은 대체 표현이다. 그래서 일부 모델은 reverse-complement 데이터 증강 또는 RC equivariance를 구조에 넣는다(예: reading_list 03의 Caduceus). 단, 전사 방향이나 strand-specific 과제처럼 **가닥 방향이 중요한 과제**에서는 두 서열을 동일하게 취급하면 안 된다.
</details>

**B3.** 다중 서열 정렬(MSA)의 공진화 신호가 단백질 잔기 접촉(contact) 예측에 쓰이는 이유는?
<details><summary>답</summary>
공간적으로 접촉한 두 잔기는 한쪽이 변하면 다른 쪽이 보상 변이를 일으키는 경향이 있어, MSA에서 두 위치 사이에 통계적 의존성이 생긴다. 이 신호를 읽어 접촉을 추정한다. (Rao 2021: 언어모델의 attention map도 contact를 학습한다. reading_list 02에 있다)
</details>

**B4.** Deep mutational scanning(DMS)은 무엇을 측정하나?
<details><summary>답</summary>
단백질의 많은 변이체(대개 가능한 모든 단일 아미노산 치환)에 대해 기능 선택/스크리닝 검정으로 각 변이의 기능 점수(fitness)를 대규모로 측정하는 실험이다. ProteinGym 같은 변이 효과 벤치마크의 정답 데이터가 된다.
</details>

**B5.** Missense, synonymous, nonsense 변이를 구분하라.
<details><summary>답</summary>
missense는 아미노산이 바뀌는 변이다. synonymous는 코돈이 바뀌어도 아미노산은 같은 변이다. nonsense는 조기 정지 코돈이 생겨 단백질이 짧아지는 변이다.
</details>

**B6.** 단일세포 RNA-seq 카운트 데이터의 주요 특성과 분석상 어려움은?
<details><summary>답</summary>
주요 특성은 다음과 같다. 희소(0이 많음, dropout 포함), 세포별 라이브러리 크기 차이, 과분산된 카운트, 배치 효과. 그래서 정규화·로그 변환 또는 부정이항/ZINB 기반 확률 모델(예: reading_list 04의 scVI)이 쓰인다.
</details>

**B7.** Promoter와 enhancer의 차이는? 서열→발현 모델이 긴 문맥을 필요로 하는 이유와 연결해 설명하라.
<details><summary>답</summary>
Promoter는 전사 시작점 근처에서 기본 전사 개시를 돕는 조절 영역이다. Enhancer는 거리·방향에 비교적 덜 의존하며 멀리서 유전자 발현을 조절할 수 있는 조절 요소다. 원거리 enhancer의 영향까지 포착하려면 수십~수백 kb 문맥이 필요하다(이것이 Enformer, Borzoi 계열의 동기다. reading_list 03을 참고한다).
</details>

**B8.** AlphaFold의 pLDDT는 무엇을 뜻하나?
<details><summary>답</summary>
잔기별 예측 신뢰도 점수다(0~100). 낮은 값은 예측이 불확실하거나 내재적 무질서 영역일 수 있다. 전체 구조의 정확도(예: TM-score)와는 다른 지표다.
</details>

**B9.** Homolog, ortholog, paralog를 구분하라.
<details><summary>답</summary>
homolog는 공통 조상을 공유하는 서열이다(상위 개념). ortholog는 종 분화로 갈라진 상동 유전자다. paralog는 유전자 중복으로 갈라진 상동 유전자다.
</details>

**B10.** GWAS에서 유의한 변이가 곧 원인(causal) 변이가 아닌 이유는? 변이 효과 예측 모델은 여기서 어떤 역할을 기대받나?
<details><summary>답</summary>
연관 불균형(LD) 때문에 원인 변이 주변의 많은 변이가 함께 유의하게 나타난다. 모델이 변이의 기능적 영향을 예측해 후보를 좁히는(fine-mapping 보조) 용도로 기대되지만, 모델 점수가 곧 인과를 뜻하지는 않는다.
</details>

---

## 채점 메모
| 영역 | 맞은 개수 | 메모(약한 주제) |
|---|---|---|
| ML (A) | / 10 | |
| 생물학 (B) | / 10 | |
