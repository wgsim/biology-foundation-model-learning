# 치트시트

모듈에서 실제로 쓴 패턴·계산을 한 장씩 정리했다. 각 파일의 `python` 블록은 저장소 환경(`.venv`)에서 **순서대로 실행해 확인**했다(블록 안의 `assert`가 통과해야 한다). `bash` 블록은 표기가 없으면 실행하지 않은 것이다. 새 모델·다른 라이브러리 버전에서는 다시 확인할 것.

| 파일 | 내용 | 관련 모듈 |
|---|---|---|
| [01_embedding_recipes.md](01_embedding_recipes.md) | 평균 임베딩(패딩·특수 토큰 제외), 층 선택, 변이 위치 임베딩, masked-marginal 점수, 흔한 실수 | 02 |
| [02_metrics_and_stats.md](02_metrics_and_stats.md) | Spearman·AUROC(폴드별)·AUPRC 기준선, 동점 처리, 짝지은 비교와 그룹 부트스트랩, 결정 규칙 함수 | 01~07 |
| [03_splits_and_leakage.md](03_splits_and_leakage.md) | 도메인별 분할 단위, 위치 블록·클러스터·scaffold·기증자 분할, 누출을 숫자로 재기 | 01, 02, 04, 05, 07 |
| [04_compute_and_memory.md](04_compute_and_memory.md) | 메모리 예산 공식, 측정값(Mac MPS), 환경별 가능/불가능, 장비 특이점 | 06 |

용어는 [glossary.md](../glossary.md), 벤치마크는 [benchmarks.md](../benchmarks.md), 논문은 [reading_list.md](../reading_list.md).
