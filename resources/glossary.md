# 용어집

작성: 2026-11-06. 이 과정(모듈 00~07)에서 쓰는 용어를 한국어 설명으로 정리했다. 모델·논문별 세부 사실(크기, 데이터 규모 등)은 적지 않았고 `reading_list.md`와 각 모듈 README에 있다. 표의 마지막 열은 처음 또는 가장 자세히 다룬 모듈이다. 정의는 이 과정에서 쓰는 의미 기준이며, 분야에 따라 다르게 쓰이면 비고에 적었다.

## 1. 모델 · 학습

| 용어 | English | 설명 | 모듈 |
|---|---|---|---|
| 파운데이션 모델 | foundation model (FM) | 대규모 데이터로 사전학습한 뒤 여러 다운스트림 과제에 쓰는 모델. "FM"이라 불려도 단순 baseline을 항상 넘는 것은 아니다 | 00, 04 |
| 사전학습 | pretraining | 라벨 없는(또는 약한 라벨의) 대규모 데이터로 일반 표현을 먼저 학습하는 단계 | 01 |
| 자기지도 학습 | self-supervised learning | 입력 자체에서 학습 신호를 만든다(가린 토큰 맞히기, 다음 토큰 예측 등) | 01 |
| 마스크 언어모델 | masked language model (MLM) | 일부 토큰을 가리고 맞히는 목적함수. ESM-2, DNABERT, Nucleotide Transformer가 쓴다 | 01 |
| 자기회귀 | autoregressive | 앞 토큰으로 다음 토큰을 예측. 생성에 적합. ProGen, Evo 계열 | 01 |
| 대조학습 | contrastive learning | 짝이 맞는 쌍은 가깝게, 아닌 쌍은 멀게 학습. CLIP, 서열–텍스트 정렬 | 05 |
| InfoNCE | InfoNCE | 배치 안의 다른 항목을 음성으로 보고 "같은 인덱스가 정답"인 분류를 푸는 대조 손실. 정답이 모두 같게 보이면 초기 손실이 ln(배치 크기) | 05 |
| 확산 모델 | diffusion model | 잡음을 단계적으로 제거하며 생성. RFdiffusion, AlphaFold 3의 구조 생성부 | 02 |
| 토크나이저 | tokenizer | 서열을 모델 입력 단위로 자르는 규칙 | 01 |
| k-mer | k-mer | 길이 k의 연속 부분서열. 겹치게/겹치지 않게 쓸 수 있다 | 01 |
| BPE | byte-pair encoding | 자주 같이 나오는 조각을 반복 병합해 어휘를 만드는 토큰화. DNABERT-2가 채택(초기 DNABERT는 겹치는 k-mer) | 01, 03 |
| 임베딩 | embedding | 서열·세포·분자를 나타내는 벡터. 모델의 어느 층·어떻게 풀링했는지에 따라 달라진다 | 01 |
| 평균 풀링 | mean pooling | 토큰별 벡터를 평균해 서열 하나의 벡터로 만듦. 패딩과 특수 토큰(`<cls>`, `<eos>`)을 뺐는지 확인 | 02 |
| 선형 probe | linear probe | 동결한 임베딩 위에 선형 모델(Ridge 등)만 학습해 표현이 정보를 담았는지 보는 평가 | 02 |
| 동결 | frozen | 사전학습 가중치를 갱신하지 않음 | 02, 06 |
| 미세조정 | fine-tuning | 사전학습 가중치까지 과제 데이터로 갱신. 과적합·망각 위험 | 06 |
| few-shot | few-shot | 과제 예제를 소수만 제공(프롬프트 또는 소량 학습). 예제가 없으면 zero-shot, 전부 쓰면 일반 지도학습. 이 과정에서는 다루지 않는 설정 | — |
| 분포 이동 | distribution shift | 학습과 평가(배포) 데이터의 분포가 다른 상태. 새 위치·새 가족·새 기증자로의 분할이 이를 흉내 낸다 | 04, 07 |
| 매개변수 효율 미세조정 | parameter-efficient fine-tuning (PEFT) | 일부 소수 매개변수만 학습(LoRA 등) | 06 |
| LoRA | low-rank adaptation | 동결 가중치에 저랭크 분해 행렬을 더해 그 행렬만 학습. 학습 매개변수·옵티마이저 메모리를 줄이지만 순전파·역전파 계산은 남는다 | 06 |
| 스케일링 법칙 | scaling law | 모델 크기·데이터·연산량과 손실의 경험적 관계. 도메인(서열)마다 다를 수 있다 | 01 |
| 어텐션 | attention | 토큰 간 가중 합으로 문맥을 섞는 연산. 표준 구현은 L×L 점수 행렬을 만들어 시간·메모리가 L²로 는다(FlashAttention류는 점수 행렬을 저장하지 않아 메모리를 줄이지만 연산량은 여전히 L²†) | 01 |
| FlashAttention | FlashAttention | 정확한 어텐션을 타일 단위로 계산해 메모리 입출력을 줄이는 CUDA 구현†. Mac MPS에는 없다 | 06 |
| 상태공간 모델 / 장거리 합성곱 | state-space model (SSM) / long convolution | 긴 서열을 어텐션보다 낮은 비용으로 처리하는 계열. Mamba는 SSM, Hyena는 암묵적 장거리 합성곱이며 HyenaDNA·Caduceus 등 DNA 모델이 장거리 문맥에 쓴다 | 01, 03 |
| 혼합 정밀도 | mixed precision | 반정밀도(bf16/fp16) 연산 + fp32 가중치 사본. 이득은 주로 속도·활성값 메모리 | 06 |
| 그래디언트 체크포인팅 | gradient checkpointing | 일부 활성값을 저장하지 않고 역전파에서 다시 계산해 메모리를 시간과 맞바꿈 | 06 |
| 데이터 병렬 | data parallel (DDP) | 모델을 복제하고 데이터를 나눈 뒤 기울기를 평균 | 06 |
| ZeRO / FSDP | ZeRO / FSDP | 매개변수·기울기·옵티마이저 상태를 장치에 나눠 저장해 한 장치의 메모리를 줄이는 방법† | 06 |
| zero-shot | zero-shot | 해당 과제용 학습 없이 사전학습 모델의 출력만으로 예측 | 02 |
| 헤드 | head | 임베딩 위에 얹는 작은 예측 모듈(선형층 등) | 02 |
| 리비전 고정 | revision pinning | 모델/데이터를 저장소의 특정 커밋 해시로 고정해 재현성 확보 | 05, 07 |

## 2. 평가 · 통계 · 연구 방법

| 용어 | English | 설명 | 모듈 |
|---|---|---|---|
| 누출 | leakage | 평가에 쓰는 정보가 학습 과정(분할, 전처리, 설정 선택)에 새어 들어간 상태 | 01, 07 |
| 상동성 분할 | homology split | 서열 동일도 임계값이나 가족 기준으로 train/test를 가르는 분할 | 01 |
| 위치 블록 분할 | position-block split | 연속한 위치 구간을 한 그룹으로 묶어 분할(같은 위치의 다른 치환 누출을 막음) | 07 |
| scaffold 분할 | scaffold split | 분자의 Bemis–Murcko 골격이 같은 분자를 같은 쪽에 두는 분할. 골격이 달라도 분자가 비슷할 수 있다 | 05 |
| 시간 분할 | time split | 발표/측정 시점으로 train과 test를 나눔 | 05 |
| 기증자 단위 분할 | donor-level split | 같은 기증자의 세포가 train과 test에 걸치지 않게 가름 | 04 |
| LODO | leave-one-donor-out | 기증자 한 명을 test로 두는 교차검증 | 04 |
| 오염 | contamination | 평가 데이터가 사전학습 코퍼스에 들어 있음 | 01 |
| baseline | baseline | 비교 기준 방법. 이 과정에서는 ① 파라미터를 학습하지 않는 것, ② 단순 특징, ③ 무작위 초기화 모델 | 07 |
| 무작위 초기화 대조군 | random-init control | 같은 구조에서 사전학습 없이 초기화한 모델. 사전학습의 기여를 분리. 시드마다 새로 만든다 | 04, 07 |
| 음성 대조 | negative control | 라벨/입력을 섞어 신호가 없을 때의 점수를 봄. 진단이지 누출 부재의 증명이 아니다 | 05, 07 |
| 선택 효과 | selection effect | 후보 중 train 점수가 가장 높은 것을 고르는 절차 자체가 만드는 성능 차이 | 06 |
| 짝지은 비교 | paired comparison | 같은 분할·같은 항목에서 두 방법의 점수 차이를 구해 비교 | 04, 07 |
| 그룹 부트스트랩 | group bootstrap | 상관 단위(그룹)를 복원추출해 구간을 구함. test 군집이 적으면 무의미 | 07 |
| 조건부 구간 | conditional interval | 학습·초기화를 고정하고 test만 다시 뽑아 얻은 구간. 학습 변동은 반영하지 않는다 | 07 |
| Spearman 상관 | Spearman correlation | 순위 상관. 상수 예측에서는 정의되지 않음(NaN) | 02 |
| AUROC | ROC-AUC | 무작위로 고른 양성이 음성보다 높게 점수 매겨질 확률(동점은 1/2로 셈). 순위 성능이며 확률 보정(calibration)과 별개. 폴드 모델 점수는 폴드별로 구해 평균 | 05 |
| AUPRC | average precision (AP) / PR-AUC | 양성이 드문 분류에 적합. `average_precision_score`는 계단식 적분인 AP로 PR 곡선의 사다리꼴 면적과 약간 다르다. 무작위 기준은 대략 양성 비율(유병률에 의존) | 03 |
| macro-F1 | macro-averaged F1 | 클래스별 F1의 단순 평균. 불균형 데이터에서 클래스를 동등하게 본다 | 04 |
| Recall@k | recall@k | 상위 k개 안에 정답이 들어 있는 비율(정답이 질의당 하나일 때; 여러 개면 찾은 정답 수/전체 정답 수와 "하나라도 찾았는가"의 hit rate를 구분). 갤러리 크기와 동점 규칙에 의존 | 05 |
| 갤러리 | gallery | 검색에서 정답을 골라야 하는 후보 집합 | 05 |
| ties-against-truth | ties-against-truth | 정답과 점수가 같은 후보를 정답보다 앞선 것으로 세는 보수적 동점 규칙 | 05 |
| 사전 등록 | pre-registration | 분할·지표·주 비교·결정 규칙을 결과를 만들기 전에 고정(커밋)해 기록 | 07 |
| 결정 규칙 | decision rule | "지지/지지되지 않음"을 정하는 조건. 코드의 함수로 고정 | 07 |
| 민감도 분석 | sensitivity analysis | 한 번에 한 요인만 바꿔 결론이 얼마나 변하는지 봄. 설정을 고르는 데 쓰지 않는다 | 05, 07 |
| 효과 크기 | effect size | 차이의 크기(통계적 유의성과 별개). 의미 있는 최소 차이를 사전에 정한다 | 07 |
| red flag | red flag | 루브릭에서 해당 차원의 점수를 자동으로 낮추는 절차상 결함 | 07 |
| 재현성 | reproducibility | 같은 코드·데이터로 같은 결과를 얻는 것. 환경·시드·해시를 기록 | 07 |
| DOME | DOME | 생물학 지도 ML 검증 보고 권고(데이터, 최적화, 모델, 평가) | 01 |

## 3. 단백질

| 용어 | English | 설명 | 모듈 |
|---|---|---|---|
| 아미노산 서열 | amino-acid sequence | 20종 표준 아미노산의 열. 단일 잔기 토큰화가 기본 | 01 |
| MSA | multiple sequence alignment | 상동 서열을 정렬한 것. AlphaFold 2, EVE 같은 모델의 입력/학습 정보 | 02 |
| DMS | deep mutational scanning | 많은 변이의 효과를 한 번에 측정하는 실험. 노이즈가 있다 | 02 |
| 적합도 | fitness | DMS에서 측정한 변이의 기능/성장 점수. 임상 병원성과 목표가 다르다 | 02 |
| 상위성 | epistasis | 변이 효과가 단순 합이 아니라 서로 의존함 | 02 |
| masked marginal | masked marginal | 변이 위치를 가렸을 때 변이 아미노산과 야생형의 log 확률 차이로 점수를 매김. 마스크 없이 야생형 서열 1회로 모든 위치를 점수화하는 wild-type marginal(싸지만 야생형 잔기가 입력에 보임), 변이/야생형 서열 각각의 의사 로그우도(pseudo-log-likelihood, PLL) 비율(비싸지만 다중 변이로 확장 가능)과 구별한다 | 02 |
| contact map | contact map | 잔기 쌍의 3차원 근접 여부 행렬. 어텐션에서 읽어내는 연구가 있다. 예측 평가에는 상위 L(또는 L/5)개 예측의 정밀도(precision@L/5)를 흔히 쓴다(모듈 02 노트북 03) | 02 |
| pLDDT | predicted LDDT | 구조 예측의 잔기별 신뢰도(0~100) | 02 |
| PAE / pAE | predicted aligned error | 한 잔기를 기준으로 정렬했을 때 다른 잔기 위치의 예상 오차(잔기 쌍 단위) | 02 |
| pTM / ipTM | predicted TM-score / interface pTM | 전체/계면의 구조 정확도 예측 점수 | 02 |
| inverse folding | inverse folding | 백본 구조에서 서열을 설계. ProteinMPNN, ESM-IF1 | 02 |
| de novo 설계 | de novo design | 자연에 없는 단백질을 새로 설계. 이중 용도 검토 대상 | 02, 06 |
| 임상 병원성 예측 | clinical variant pathogenicity prediction | 사람 변이가 질병과 연관되는지 예측. AlphaMissense, EVE | 02 |

## 4. DNA · RNA · 유전체

| 용어 | English | 설명 | 모듈 |
|---|---|---|---|
| 참조 게놈 | reference genome | 좌표 기준이 되는 게놈 조립본(예: hg38). 좌표계를 맞추지 않으면 위치가 어긋난다 | 03 |
| 프로모터 | promoter | 전사 개시 부근의 조절 서열 | 03 |
| 인핸서 | enhancer | 먼 거리에서 전사를 높이는 조절 서열 | 03 |
| TSS | transcription start site | 전사 시작 위치 | 03 |
| 염색질 접근성 | chromatin accessibility | DNase/ATAC 신호로 측정하는 열린 염색질 | 03 |
| ChIP / CAGE | ChIP / CAGE | 단백질–DNA 결합(히스톤 표지 포함) / 전사 시작을 재는 기능유전체 실험 | 03 |
| sequence-to-function | sequence-to-function | DNA 서열에서 기능 트랙(발현, 접근성 등)을 직접 예측하는 지도학습 모델. Enformer, Borzoi, AlphaGenome | 03 |
| 트랙 / bin | track / bin | 서열을 일정 폭 구간(bin)으로 나눈 신호 예측 한 줄 | 03 |
| LLR | log-likelihood ratio | log p(alt\|문맥) − log p(ref\|문맥). DNA LM의 zero-shot 변이 점수 | 03 |
| ISM | in silico mutagenesis | 한 염기를 바꿔 예측 변화를 보는 방식 | 03 |
| phyloP / phastCons | phyloP / phastCons | 종 간 보존 점수. 학습 없이도 강한 기준선 | 03 |
| CADD | CADD | 여러 주석을 통합한 변이 해로움 점수 | 03 |
| 멘델 / 복합 형질 | Mendelian / complex traits | 단일 유전자 질환 vs 다수 변이가 기여하는 형질 | 03 |
| 매칭된 대조군 | matched controls | 변이 유형, TSS 거리 등을 맞춘 비인과 변이. 단순 특징의 정보를 줄이려는 설계 | 03 |
| 개인 간 cis 변이 효과 | personal cis-variant effect | 한 유전자 근처 변이가 개인의 발현 차이를 만드는 효과. 참조 게놈의 유전자 간 차이와 다른 과제 | 03 |
| RNA 기반 FM | RNA foundation model | RNA 서열에서 구조·기능을 예측하는 사전학습 모델(RNA-FM, RiNALMo 등) | 05 |

## 5. 단일세포

| 용어 | English | 설명 | 모듈 |
|---|---|---|---|
| 카운트 행렬 | count matrix | 세포 × 유전자 UMI 카운트 | 04 |
| HVG | highly variable genes | 세포 간 변동이 큰 유전자 선택 | 04 |
| 순위값 인코딩 | rank value encoding | 깊이 정규화 → 유전자별 코퍼스 중앙값으로 나눔 → 값 내림차순의 유전자 ID 열. Geneformer | 04 |
| 희소성 | sparsity (zeros) | 단일세포 카운트의 대부분이 0인 성질. 생물학적 무발현과 기술적 검출 실패가 섞여 있다† | 04, 05 |
| 배치 효과 | batch effect | 실험 조건 차이에서 오는 비생물학적 변동 | 04 |
| 배치 보정 | batch correction / integration | 배치 신호를 줄이는 방법(Harmony, scVI 등) | 04 |
| ASW | average silhouette width | 군집 분리도 지표. scib에서 배치 혼합/유형 보존에 쓰임 | 04 |
| 보존 vs 혼합 | bio-conservation vs batch mixing | 세포 유형 구조를 지키면서 배치를 섞는 두 목표. 함께 읽는다 | 04 |
| 섭동 예측 | perturbation prediction | 유전자 교란 후 전사체 변화를 예측하는 과제 | 04 |
| scVI | scVI | 변분 오토인코더 기반 단일세포 모델. 확률적 학습이라 실행마다 수치가 조금 달라진다 | 04 |

## 6. 분자 · 멀티모달

| 용어 | English | 설명 | 모듈 |
|---|---|---|---|
| SMILES | SMILES | 분자를 문자열로 쓰는 표기. 같은 분자도 표기가 여러 가지일 수 있다 | 05 |
| ECFP / Morgan 지문 | ECFP / Morgan fingerprint | 원자 주변 구조를 해시한 비트(카운트) 지문. 강한 baseline | 05 |
| Tanimoto 유사도 | Tanimoto similarity | 비트 지문 두 개의 교집합/합집합 비율(0~1). 카운트 지문에는 최소값 합/최대값 합 형태의 확장을 쓴다† | 05 |
| Murcko 골격 | Bemis–Murcko scaffold | 고리계와 연결 사슬만 남긴 분자 골격 | 05 |
| Butina 군집 | Butina clustering | Tanimoto 거리 기반 분자 군집. 부트스트랩의 상관 단위로 씀 | 05 |
| 모달리티 | modality | 서열, 구조, 텍스트, 이미지처럼 데이터 종류 | 05 |
| 모달리티 간극 | modality gap | 대조학습 공유 공간에서 서로 다른 모달리티의 임베딩이 분리된 영역에 놓이는 현상 | 05 |
| 거짓 음성 | false negative (in contrastive) | 배치 안에서 음성으로 취급했지만 실제로는 맞는 짝인 항목 | 05 |
| 전체 슬라이드 이미지 | whole-slide image (WSI) | 병리 조직 슬라이드 전체 이미지 | 05 |
| 에이전트 | agent | LLM이 도구를 호출하며 다단계 과제를 수행하는 구성 | 05 |

## 7. 해석 · 책임성 · 안전

| 용어 | English | 설명 | 모듈 |
|---|---|---|---|
| 희소 오토인코더 | sparse autoencoder (SAE) | 임베딩을 과완비 희소 코드로 재구성해 해석 가능한 특징을 찾으려는 방법 | 06 |
| TopK SAE | TopK SAE | 활성이 큰 k개만 남기는 희소 제약. 희소도를 직접 제어 | 06 |
| 다의성 / superposition | polysemanticity / superposition | 뉴런이 여러 개념에 반응하는 현상을, 신경망이 뉴런 수보다 많은 특징을 겹쳐 표현하기 때문이라 보는 가설 | 06 |
| 죽은 특징 | dead latent | 전 데이터에서 거의 켜지지 않는 SAE 특징 | 06 |
| 선택 효과 대조 | selection-effect control | 훈련하지 않은 무작위 사전에서도 같은 선택 절차를 적용해 보는 대조 | 06 |
| 이중 용도 | dual use | 유익한 연구가 오용될 수도 있는 성질. 공개 범위를 검토 | 06 |
| 접근 승인(gated) 모델 | gated model | 약관 동의가 필요한 모델·데이터. 에이전트가 대신 동의하지 않는다 | 05 |
| pickle / `trust_remote_code` | pickle / trust_remote_code | 코드 실행 경로가 될 수 있는 직렬화/원격 코드. 고정한 리비전과 검토 후에만 | 05 |
| 모델 카드 | model card | 모델의 용도, 성능, 한계를 적는 문서 | 07 |
| 데이터시트 | datasheet for datasets | 데이터셋의 동기·구성·수집 과정·권장 용도 문서 | 07 |
| 라이선스 근거 | license evidence | 라이선스를 어디서 확인했는지(파일·API 필드). 확인 못 하면 "미확인" | 06 |

## 8. 이 저장소의 표기

| 표기 | 의미 |
|---|---|
| 🟢 🟡 🔴 | 실행 등급: Mac 로컬 / Colab GPU / 클러스터 |
| † | 논문 본문·공식 문서에서 확인하지 못한 서술 |
| ⚠️ 미검증 | 사용 전 직접 확인할 것 |
| ✅ PubMed / bioRxiv / Web / Crossref | 서지 확인에 쓴 출처 (reading_list 범례) |
| ★ ◆ ○ | 필수 / 심화 / 선택 읽기 |
