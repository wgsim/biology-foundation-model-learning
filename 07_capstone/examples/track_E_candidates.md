# 트랙 E 후보 논문 조사 (2026-11-09)

[캡스톤 README §3](../README.md#3-트랙-e-논문-재현-프로토콜)의 4가지 선정 조건을 [reading_list](../../resources/reading_list.md)의 [비판]·[벤치마크] 논문에 적용해 **시험 실행에 쓸 후보를 좁힌** 기록이다.

> **이 기록의 성격**
> - **후보 조사이지 재현이 아니다.** 새 다운로드·설치·계산은 없다. 공개 저장소·모델·데이터 페이지(GitHub API, Hugging Face API, README)를 읽었고, 읽은 파일은 저장하지 않았다.
> - **확인한 것과 ⚠️(확인하지 못한 것)을 구분**했다. 논문 본문·표는 읽지 않았다(주장 문장과 수치는 학습자가 3-pass로 직접 확인할 것).
> - 학습자의 트랙 E는 본인이 고른 논문이다. 아래는 *작성자가 시험 실행에 쓸 후보*이며 학습자에게 주는 정답이 아니다.

> **후속**: 1순위 E-1은 [시험 실행](dry_runs/track_E_traitgym.md)했다(공개 지표 CSV 22개 재현, 승인 후 2,058 B 다운로드).

## 4가지 조건
① 핵심 주장이 한 문장 · ② 코드·데이터·가중치 공개와 라이선스 확인 · ③ 계산이 16~20시간 안(🟢 Mac / 🟡 Colab) · ④ 재현할 **주장 단위**(표 한 칸 또는 그림 하나)를 특정할 수 있다.

## 후보 비교
| 후보 | ② 코드·데이터·라이선스 (확인한 것) | ③ 계산 | ④ 주장 단위 | 판정 |
|---|---|---|---|---|
| **E-1 TraitGym** (Benegas et al. 2025, bioRxiv 10.1101/2025.02.11.637758) | 저장소 `songlab-cal/TraitGym` **MIT**(API 확인, 5.3 MB, 2026-06 갱신), HF 데이터셋 `songlab/TraitGym` **MIT**(카드 확인). 점수 파일뿐 아니라 **지표 CSV도 공개**: `mendelian_traits_matched_9/AUPRC_by_chrom_weighted_average/all/`에 모델별 파일 81개(합 14 KB 이하). 이 저장소 캐시의 스냅샷(`1fde19555fe8…`)이 API의 현재 리비전과 같다 | 🟢 점수 17종이 이미 캐시에 있어 지표 재계산은 CPU 분 단위. 다른 점수는 파일을 더 받아야 한다(데이터셋 전체 약 7.1 GB(HF 파일 목록의 크기 합), 필요한 것만) | `…/AUPRC_by_chrom_weighted_average/all/<모델>.csv`의 `score ± se` 한 칸(예: 파일 한 개는 `model,metric,score,se` 헤더에 한 줄) | **1순위 권장** |
| **E-2 ProteinGym** (Notin et al. 2023, bioRxiv 10.1101/2023.12.07.570727) | 저장소 `OATML-Markslab/ProteinGym` **MIT**(93.7 MB). 공개 지표표 `benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv`(141 KB)에 assay별 점수와 ESM2(8M·35M·150M·650M·3B·15B)·ESM-1v 등 열이 있다(헤더 확인). 데이터와 ESM-2 8M~650M은 캐시에 있다 | 🟢~🟡 650M 점수화는 assay 수·길이에 비례(**측정하지 않음** ⚠️) | assay별 Spearman 한 칸(예: DN7A × ESM2 650M) 또는 유형별 평균 | 2순위. **개별 DMS 데이터 라이선스는 미확인** ⚠️(템플릿과 같은 상태), ESM-2 점수화 방식(masked/wt marginals)·긴 서열 처리는 코드에서 확인 필요 ⚠️ |
| **E-3 Ahlmann-Eltze et al. 2025** (Nat Methods; 10.1038/s41592-025-02772-6) | 저장소 `const-ae/linear_perturbation_prediction-Paper` **MIT**(63.8 MB; Zenodo DOI 10.5281/zenodo.14832393에 보관). 분석은 **R 스크립트**, 벤치마크 스크립트가 따로 있다. 이 Mac에는 R(`/usr/local/bin/R`)이 있다 | ⚠️ 필요한 데이터의 크기·획득 경로를 확인하지 못했다. 딥러닝 모델 재실행은 GPU·긴 시간일 가능성(미확인) | 선형 baseline과 모델의 오차 비교(그림 한 개 또는 표 한 칸) | 더 조사 필요. 과정의 주제("FM이 항상 이기지 않는다")와 가장 가깝지만 지금은 판정할 수 없다 |
| **E-4 Kedzierska et al. 2025** (Genome Biol; 10.1186/s13059-025-03574-x) | 저장소 `microsoft/zero-shot-scfoundation` **MIT**. README: **Linux + NVIDIA GPU + Docker**에서 개발·시험, scGPT가 flash-attention(엄격한 GPU 요구)에 의존, Geneformer·scGPT 특정 커밋에 고정, 데이터는 figshare 한 파일(크기 ⚠️) | ✗ 이 Mac(MPS)·Colab T4에서는 맞지 않을 가능성이 높다(호환 GPU 목록은 확인하지 못함 ⚠️) | 클러스터링·배치 지표 표 | 제외 권장(GPU 클러스터가 생기면 재검토) |
| **E-5 Praski et al. 2025** (arXiv, 분자 임베딩 벤치마크) | 코드 저장소를 **찾지 못했다**(GitHub 저장소 검색 한 번에서 결과 없음 — 철저한 조사는 아님) ⚠️ | — | — | 제외 |

## 권장: E-1 TraitGym (L1 + L3)
- **이유**: (1) 계산이 거의 없다 — 점수가 캐시에 있다. (2) 공개 지표 CSV가 "정답"이라 L1 재현의 기준이 분명하다. (3) 모듈 03 노트북의 † 표시를 풀 수 있다 — 노트북은 염색체 가중 AUPRC를 직접 구현하며 "공식 정의와 다를 수 있다†"고 적었다.
- **사전 점검(한 개 값)**: 모듈 03 노트북 02의 `auprc_chrom_w`가 GPN-MSA |LLR|에서 0.654였고, 공개 CSV(`GPN-MSA_absLLR.plus.score`)는 AUPRC 0.6542(se 0.0450)다. 일치한다. 나머지 16개 점수는 비교하지 않았다 — 이것이 시험 실행의 L1 내용이다.
- **L1 재현**: 캐시의 점수 17종으로 `AUPRC_by_chrom_weighted_average`를 직접 계산해 공개 `score`와 비교. **허용 오차는 결과를 보기 전에 정한다**: 계산이 결정적이라 시드 변동이 없으므로(프로토콜 3의 "시드 변동"은 해당 없음) *공개 se*와 반올림 수준(소수 4자리)을 기준으로 삼는 안을 제안한다. 파일 이름의 부호(`.plus`/`.minus`)와 `score` 열 규칙을 따른다.
- **L3 변형 검증 (하나)**: 공개된 consequence 부분집합 CSV(`3_prime_UTR_variant/` 등)에서 순위가 유지되는가, 또는 match_group 단위 부트스트랩 대 염색체 단위 비교.
- **필요한 승인(실행하려면)**: 지표 CSV 17개(각 100 B 안팎) + (L3용) 부분집합 CSV 몇 개 — 합 100 KB 이하, 설치 없음. Snakemake 워크플로(`workflow/`)는 읽기만 하고 파이썬으로 지표 정의를 옮긴다(`snakemake`는 설치되어 있지 않다).
- **위험 ⚠️**: (a) 공개 CSV는 현재 데이터셋 릴리스 기준이라 프리프린트 표와 버전이 다를 수 있다(저장소에 `mendelian_traits_v21_`·`v22_matched_9` 폴더가 따로 있다). (b) 지표 정의의 세부(염색체 가중치, NaN 처리)는 `workflow/`를 읽어야 확정된다. (c) 논문의 주장 문장은 이 조사에서 읽지 않았다 — 학습자 절차(3-pass 읽기)로 직접 정해야 한다.

## 이 조사에서 하지 않은 것
- 논문 본문·그림·표 읽기, 주장 문장 확정, 재현 실행(위 L1 한 값 비교 외).
- E-3의 데이터 크기·획득 경로, E-4의 호환 GPU 목록, E-2의 점수화 코드 확인(⚠️로 남김).
- 교차 리뷰.
