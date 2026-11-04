# Biology Foundation Model 학습 로드맵

생물학 foundation model(FM)을 **중급 → 전문가 수준**까지 혼자 학습하기 위한 자료.
대상: ML·생물학 기초는 이미 있는 학습자. 형식: Markdown 강의노트 + Jupyter 실습.

## 실행 등급
| 표시 | 환경 |
|---|---|
| 🟢 | Mac 로컬 (CPU/MPS) |
| 🟡 | Google Colab GPU (T4급) |
| 🔴 | 기관 GPU 클러스터 |

환경 설정은 [SETUP.md](SETUP.md) 참고.

## 18주 일정 (주 8~10시간 기준)
| 주차 | 모듈 | 핵심 목표 |
|---|---|---|
| 0 | [00_orientation](00_orientation/) | 자가진단, 논문 읽기·재현 방법 |
| 1–2 | [01_foundations](01_foundations/) | 서열 토크나이징, 목적함수, 평가 함정(데이터 누수) |
| 3–6 | [02_protein](02_protein/) | PLM, 구조 예측, 설계, 변이 효과 예측 |
| 7–9 | [03_dna_genome](03_dna_genome/) | DNA/RNA 언어모델, sequence-to-function |
| 10–12 | [04_single_cell](04_single_cell/) | 단일세포 FM과 baseline 대비 비판적 평가 |
| 13–14 | [05_multimodal_other](05_multimodal_other/) | 분자·이미지·멀티모달·생물학 LLM |
| 15–16 | [06_advanced_research](06_advanced_research/) | 파인튜닝, 해석, 효율화, 벤치마크 설계 |
| 17–18 | [07_capstone](07_capstone/) | 프로젝트 또는 논문 재현 |

## 수준 체크포인트
- **중급 완료 (2~9주 후)**: 임베딩+probe, zero-shot 변이 점수화를 직접 구현하고 split 누수를 설명할 수 있다.
- **고급 (10~16주 후)**: FM과 단순 baseline을 공정하게 비교하는 평가를 설계할 수 있다.
- **전문가 (캡스톤)**: 논문 1편을 재현하거나 새 평가/적용 실험을 수행하고 한계를 서술할 수 있다.

## 각 모듈 구성
`README.md`(강의노트) · `notebooks/`(실습) · `exercises.md`(과제)

모듈마다 산출물 1개(단순 baseline 재현 / split 변경 효과 확인 / 동결 표현 vs 미세조정 비교 중 하나)와 데이터 누출 점검 항목을 포함한다. 자세한 내용은 [resources/reading_list.md](resources/reading_list.md)의 운영 메모 참고.

## 진행 상태
- [x] 스캐폴드
- [x] 1차 문헌 검증 및 `resources/reading_list.md` (v2: Codex·agy 교차 검토 반영)
- [x] 모듈 00 오리엔테이션
- [x] 모듈 01 Foundations (강의노트·과제·노트북 3종, 실행 확인)
- [ ] 모듈 02–07 작성

> 최신 모델·논문 정보는 작성 시 문헌 DB로 확인하며, 확인되지 않은 항목은 `⚠️ 미검증`으로 표시한다.
