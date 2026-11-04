# 모듈 02 과제

정답은 제공하지 않는다. **평가 기준**을 만족하는지 스스로 점검하고, 결과는 [실험 로그 템플릿](../00_orientation/experiment_log_template.md)으로 기록한다. 모든 지표는 시드/반복·신뢰구간 또는 표준편차와 함께 보고한다.

## 과제 1 — probe vs baseline vs split (모듈 산출물)  *(노트북 01)*
노트북 01에 **유형이 다른 assay를 하나 더** 추가하고(예: Expression 또는 Activity), 다음 중 최소 둘을 바꿔 결과가 어떻게 달라지는지 보고하라.
- 위치 블록 크기 5 / 10 / 20
- 훈련 샘플 수를 1/4로 줄이기
- ESM-2 임베딩 층 선택 (마지막 층 vs 중간 층)
- 평가 기준: ① baseline(B1, B2) 포함 표, ② random·position-block 두 split의 결과, ③ ESM 특징이 baseline을 이기는/못 이기는 조건에 대한 가설 2개 이상, ④ 한계(assay 수, 반복 수) 서술.

## 과제 2 — 모델 크기와 assay 유형  *(노트북 02)*
노트북 02에 assay를 3개 이상 추가하고(ProteinGym 참조표에서 길이 ≤ 250, 단일 치환 ≥ 1500개 기준), 모델 크기(8M→650M)에 따른 Spearman 변화를 그려라.
- 평가 기준: ① 위치 단위 부트스트랩 신뢰구간, ② 크기가 커져도 좋아지지 않는 assay를 찾아 가설 제시, ③ BLOSUM62 baseline 대비 향상 여부를 assay별로 서술, ④ "평균 하나로 모델을 판단하면 안 되는 이유"를 본인 결과로 설명.

## 과제 3 — 점수 변형과 비용  *(노트북 02)*
masked marginal과 wild-type marginal 외에 한 가지를 더 구현하라 (예: 변이 서열의 pseudo-log-likelihood 비율, 또는 여러 마스크 위치의 평균).
- 평가 기준: ① 세 방식의 Spearman·실행 시간 표, ② 비용 대비 성능 trade-off 서술, ③ ProteinGym 공식 점수표(웹)와 본인 수치를 비교하고 차이가 나는 이유(assay 수, 단일 치환만 사용, 버전 등) 서술.

## 과제 4 — contact  *(노트북 03)*
다른 PDB 구조를 하나 이상 추가하고(길이 100~250, 사용자가 선택, 출처·라이선스 표기), `MIN_SEP`과 접촉 기준(Cβ 8 Å)을 바꿔 결과를 비교하라.
- 평가 기준: ① 최소 두 설정의 precision@L/5 표(모델 크기별), ② 층별 정밀도 곡선 또는 head별 히트맵, ③ leave-one-protein-out head 선택의 낙관성 정도 서술, ④ attention 기반 점수가 지도 학습된 contact head와 다른 이유 서술.

## 과제 5 — pLDDT는 변이 효과를 알려 주는가?  *(노트북 04, Colab 🟡)*
ProteinGym 소형 assay(예: `DN7A_SACS2_Tsuboyama_2023_1JIC`, 55 aa)에서 단일 변이 약 200개를 샘플링해 ESMFold로 접고, **평균 pLDDT 변화(변이 − wt)와 DMS 점수의 Spearman**을 구하라.
- 평가 기준: ① 표본 크기·샘플링 방법·신뢰구간, ② 같은 변이에 대한 ESM-2 zero-shot 점수와의 비교, ③ pLDDT가 변이 효과 예측에 유용한지에 대한 결론과 근거.

## 과제 6 — 논문 노트
Lin et al. 2023 (ESM-2/ESMFold) **또는** Notin et al. (ProteinGym)을 [논문 노트 템플릿](../00_orientation/paper_note_template.md)으로 정리하라.
- 평가 기준: ① 체크리스트의 사전학습–평가 겹침, split, baseline 항목을 본문 근거와 함께 채움, ② 저자가 밝힌 한계 요약, ③ 이 모듈 노트북의 수치와 논문 수치가 다를 때 이유(설정 차이) 정리.

## 과제 7 (선택, 🔴 클러스터 또는 GPU) — 설계 self-consistency
ProteinMPNN 또는 ESM-IF1로 1UBQ(또는 다른 소형 구조) 백본에서 서열을 8개 설계하고, ESMFold로 재접어 **pLDDT와 백본 RMSD**를 보고하라. 도구 설치와 사용법은 각 공식 저장소의 README를 따른다(이 저장소에는 노트북이 없음).
- 평가 기준: ① 설계 서열 8개의 서열 동일도(wt 대비), ② 재접힘 RMSD·pLDDT 표, ③ "in silico self-consistency가 실험 성공을 보장하지 않는 이유"를 본인 결과와 함께 3문장 이상.

## 과제 8 (선택) — SAE feature
InterPLM 공개 자료(논문·저장소)를 열어 ESM-2 SAE feature 세 개를 골라 어떤 개념과 대응하는지 정리하고, 그 대응을 **인과적으로 검증하려면 어떤 실험이 필요한지** 설계안을 쓰라.
- 평가 기준: ① 선택한 feature의 근거(활성 서열 예시, 주석), ② 인과 검증 설계(개입 실험), ③ 한계.
