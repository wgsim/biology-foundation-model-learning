# Biology Foundation Model 학습 로드맵

생물학 foundation model(FM)을 **중급에서 전문가 수준**까지 혼자 학습하기 위한 자료다.
대상은 ML·생물학 기초가 이미 있는 학습자이고, 형식은 Markdown 강의노트와 Jupyter 실습이다.

## English summary
A self-study curriculum on biology foundation models (protein, DNA/genome, single-cell, multimodal) for learners who already have intermediate ML and biology backgrounds. It runs 18 weeks at 8–10 hours per week and is split into 8 modules (00–07) with lecture notes, 19 Jupyter notebooks, exercises and a capstone. The prose is in Korean. Code, model names and paper titles are in English.

- **Theme**: evaluation rigor. Each module has a deliverable and a data-leakage check. The course repeatedly compares foundation models with simple baselines, measures leakage across splits, and uses negative controls.
- **Compute**: the notebooks run on a Mac (CPU/MPS) or on Colab (T4-class GPU). DDP and SLURM templates are included but have not been run on a cluster.
- **Capstone**: a two-week plan with five tracks (A–E; E reproduces a paper), templates for the proposal, report, rubric and checklist, and dry-run records. One dry run recomputes 22 published TraitGym metric values from cached model scores. All of them match within a tolerance fixed before the results were seen. This checks the score-to-metric step, not the models themselves.
- **Status**: written with an AI coding tool (Claude Code) and cross-reviewed by other AI models (OpenAI Codex, Google Gemini). There was no human expert review and no learner pilot, so treat the content as a study log rather than a validated course. Statements that could not be verified are marked `⚠️` or `†`.
- **Start here**: [SETUP.md](SETUP.md) for the environment, then the module folders in the schedule below. `bash book/build.sh` builds a searchable web book.
- **License**: MIT for code, CC BY 4.0 for prose. See the "라이선스" section at the end.

## 실행 등급
| 표시 | 환경 |
|---|---|
| 🟢 | Mac 로컬 (CPU/MPS) |
| 🟡 | Google Colab GPU (T4급, 캡스톤 시험 실행 한 건만 A100) |
| 🔴 | 기관 GPU 클러스터 |

환경 설정은 [SETUP.md](SETUP.md)를 참고한다. 목차·검색이 있는 웹 책으로 보려면 SETUP.md §6(`bash book/build.sh`)을 따른다.

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

모듈마다 산출물 1개(단순 baseline 재현 / split 변경 효과 확인 / 동결 표현 vs 미세조정 비교 중 하나)와 데이터 누출 점검 항목을 포함한다. 자세한 내용은 [resources/reading_list.md](resources/reading_list.md)의 운영 메모를 참고한다. 용어는 [glossary](resources/glossary.md), 벤치마크는 [benchmarks](resources/benchmarks.md), 반복 패턴은 [cheatsheets](resources/cheatsheets/)에 있다.

## 진행 상태
- [x] 스캐폴드
- [x] 1차 문헌 검증 및 `resources/reading_list.md` (v2: Codex·agy 교차 검토 반영)
- [x] 모듈 00 오리엔테이션
- [x] 모듈 01 Foundations (강의노트·과제·노트북 3종, 실행 확인)
- [x] 모듈 02 Protein (강의노트·과제·노트북 4종: 01–03은 Mac에서, 04는 Colab T4에서 실행 확인)
- [x] 모듈 03 DNA·Genome (강의노트·과제·노트북 3종 모두 Mac에서 실행 확인, 노트북 03은 격리 환경)
- [x] 모듈 04 Single-cell (강의노트·과제·노트북 3종 모두 Mac에서 실행 확인, Colab T4에서도 01–03 실행 확인)
- [x] 모듈 05 Multimodal·기타 (강의노트·과제·노트북 2종 모두 Mac에서 실행 확인, Colab T4에서도 2종 모두 실행 확인, 시간은 Mac의 3~6배)
- [x] 모듈 06 Advanced (강의노트·과제·노트북 3종 Mac에서 실행 확인, Colab T4에서도 01–03 실행 확인. DDP 템플릿은 CPU 2프로세스로만 검증했고 SLURM 파일은 문법 점검만 했다. 클러스터는 미검증이다)
- [x] 모듈 07 Capstone: 제안서·보고서·루브릭·점검표·검토 프롬프트 템플릿을 만들었다. 실행되는 템플릿 노트북 1종은 Mac과 Colab T4에서 실행을 확인했다. 트랙 B~D는 MVP 하나씩을 캐시로 시험 실행했다([기록](07_capstone/examples/dry_runs/README.md)). 질문의 나머지 절반도 대부분 시험 실행했고([기록](07_capstone/examples/dry_runs/second_halves.md)), Geneformer V2-104M 미세조정만 Colab A100에서 학습했다. 트랙 E는 TraitGym 한 편을 시험 실행했다([기록](07_capstone/examples/dry_runs/track_E_traitgym.md)). 시험 실행은 모두 학습자 파일럿이 아니다. 새 모델은 scGPT whole-human 체크포인트 하나(약 207 MB, 소스 26개는 설치 없이 사용)이고, 새 데이터는 MoleculeNet CSV 3개(약 3.9 MB), TraitGym 지표 CSV 22개(2 KB), CELLxGENE H5AD 1개(150 MB)뿐이다.
- [x] resources 보강: [용어집](resources/glossary.md) · [벤치마크 지도](resources/benchmarks.md) · [치트시트 4종](resources/cheatsheets/) (코드 블록은 `.venv`에서 실행 확인). 전체 DOI·arXiv ID 125개와 상대 링크 147개를 점검했다. 이후 문서가 늘어 2026-11-10에 상대 링크 208개와 앵커를 다시 점검했고 깨진 것은 없었다. DOI는 다시 대조하지 않았다. 외부 URL 10개 중 4개는 자리표시자·디렉터리 주소다.

> 최신 모델·논문 정보는 작성 시 문헌 DB로 확인하며, 확인되지 않은 항목은 `⚠️ 미검증`으로 표시한다.

## 작성 방식과 한계
- 문서와 코드는 AI 코딩 도구(Anthropic의 Claude Code)로 작성했다. 커밋마다 `Co-Authored-By` 표기가 남아 있다. 각 모듈의 계획과 외부 다운로드는 저자가 승인한 뒤 진행했다.
- 교차 검토는 다른 AI 모델(OpenAI Codex, Google Gemini)이 했다. 사람 전문가의 검토와 학습자 파일럿은 하지 않았다.
- 오류가 남아 있을 수 있다. 인용한 수치와 주장은 원문으로 다시 확인해야 한다.

## 라이선스
- 코드는 MIT 라이선스(`LICENSE`), 문서는 CC BY 4.0(`LICENSE-docs.md`)이다.
- 이 저장소에는 외부 모델 가중치와 데이터가 들어 있지 않다. 노트북이 실행 중에 내려받는 모델과 데이터는 각 출처의 라이선스를 따른다. 확인한 근거와 확인하지 못한 항목은 [06 강의노트](06_advanced_research/README.md)의 "데이터·모델 라이선스와 연구 책임성" 절에 있다.
