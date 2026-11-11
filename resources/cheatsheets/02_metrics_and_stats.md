# 치트시트 02 — 지표와 통계 (짝지은 비교 · 그룹 부트스트랩 · 동점 처리)

모듈 01~07에서 실제로 틀렸거나 틀리기 쉬웠던 계산을 모았다. `python` 블록은 합성 데이터로 **실행해 확인했다**(2026-11-06, numpy·scipy·scikit-learn은 저장소 `.venv`). 합성 데이터의 숫자는 코드의 동작을 보이려는 것이며 생물학적 결과가 아니다.

```python
import warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
```

## 1. 지표를 고르는 표
| 과제 | 주 지표 | 같이 볼 것 | 함정 |
|---|---|---|---|
| 변이 효과(연속 값) | Spearman ρ (assay별) | 유형별 집계, 노이즈 상한 | 평균 하나로 순위를 매겨서는 안 된다 (모듈 02) |
| 이진 분류, 균형 | AUROC (**폴드별**) | 정확도, 보정 | 폴드를 풀링한 AUROC (아래 §3) |
| 이진 분류, 양성이 드묾 | AUPRC | 양성 비율(무작위 기준선) | AUPRC의 무작위 기준은 0.5가 아니라 대략 양성 비율 (§4) |
| 다중 클래스, 불균형 | macro-F1 | 클래스별 재현율 | 정확도는 다수 클래스에 끌린다 |
| 검색/정렬 | Recall@k, 순위 | 갤러리 크기, 동점 규칙 | 동점 처리 (§5) |
| 배치 보정(단일세포) | 보존 지표 **와** 혼합 지표 | 지표 간 일관성 | 한쪽만 높은 방법 (모듈 04) |

## 2. 상수 예측과 Spearman
```python
rng = np.random.default_rng(0)
y = rng.normal(size=50)
const = np.zeros(50)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                # ConstantInputWarning
    rho_const = spearmanr(const, y).statistic
assert np.isnan(rho_const)                 # 상수 예측의 Spearman은 정의되지 않는다 (NaN)
assert roc_auc_score(np.r_[0, 1, 0, 1], np.zeros(4)) == 0.5     # 반면 AUROC는 동점을 1/2로 세므로 상수 예측에 0.5를 준다 (두 지표의 상수 입력 동작이 다르다)
```
- 정의되지 않는 값을 **−1이나 0으로 바꿔 평균에 넣지 않는다**. NaN으로 기록하고 제외한 시드 수를 함께 보고한다 (모듈 07 템플릿: 가장 가까운 train 위치의 평균 예측이 새 위치에서 거의 상수가 되어 한 시드의 ρ가 NaN이었다). 하이퍼파라미터 선택용 내부 점수에서만 NaN을 가장 나쁜 값으로 대체한다.

## 3. AUROC — 폴드별로 구해 평균
```python
# 폴드마다 다른 상수를 내는 모델 두 개: 어떤 폴드에서도 변별력이 없다. (roc_auc_score는 상수 입력에 0.5를 준다)
y1 = np.array([0, 1] * 10); p1 = np.full(20, 0.2)
y2 = np.array([0, 1] * 10); p2 = np.full(20, 0.8)
per_fold = np.mean([roc_auc_score(y1, p1), roc_auc_score(y2, p2)])
pooled = roc_auc_score(np.r_[y1, y2], np.r_[p1, p2])
assert per_fold == 0.5
assert pooled == 0.5                       # 균형 + 양 폴드 같은 비율이라 우연히 0.5
# 폴드마다 양성 비율이 다르면 풀링은 상수 예측도 0.5에서 벗어난다.
y3 = np.array([0] * 18 + [1] * 2);  p3 = np.full(20, 0.2)     # 양성 10%, 낮은 점수
y4 = np.array([0] * 2 + [1] * 18);  p4 = np.full(20, 0.8)     # 양성 90%, 높은 점수
per_fold2 = np.mean([roc_auc_score(y3, p3), roc_auc_score(y4, p4)])
pooled2 = roc_auc_score(np.r_[y3, y4], np.r_[p3, p4])
assert per_fold2 == 0.5 and abs(pooled2 - 0.9) < 1e-9     # 상수 예측인데 풀링하면 0.9
print("폴드별 평균", per_fold2, "| 풀링", round(pooled2, 3))
```
- 폴드 모델마다 점수 눈금이 다를 수 있으므로 **폴드별 AUROC를 우선 보고**한다. 점수를 풀링한 out-of-fold AUROC는 *다른 추정 대상*(폴드 간 눈금 차이까지 섞인 값)이므로 쓰려면 그렇게 정의하고 폴드별 값과 함께 낸다. 모듈 05를 만들던 중 풀링한 AUROC가 0.5 아래(0.460)로 나온 적이 있다. 개발 중 관측이며, 최종 노트북은 폴드별 평균을 쓴다.

## 4. AUPRC의 무작위 기준선 ≈ 양성 비율
```python
rng = np.random.default_rng(1)
y = (rng.random(20000) < 0.10).astype(int)       # 양성 10%
ap = average_precision_score(y, rng.random(20000))
assert abs(ap - y.mean()) < 0.01
print("무작위 AUPRC", round(ap, 3), "양성 비율", round(y.mean(), 3))
```
- `average_precision_score`는 계단식 적분의 **average precision(AP)** 이며 PR 곡선의 사다리꼴 면적과는 약간 다르다. 양성 비율은 표본이 클 때의 *근사* 기준이고 유한 표본의 기대 AP와 정확히 같지는 않다. 양성이 적은 부분집합(예: 양성 21개)에서는 더 흔들린다. 그래서 무작위 점수의 **순열 분포**를 함께 보고한다 (모듈 03: TraitGym).

## 5. 검색 지표의 동점 처리 — 정답에 불리하게
```python
def rank_of_truth(scores, true_idx, decimals=6):
    """정답의 순위(1이 최고). 정답과 점수가 같은 다른 항목은 정답보다 앞선 것으로 센다 (ties-against-truth)."""
    s = np.round(np.asarray(scores, dtype=float), decimals)
    better = (s > s[true_idx]).sum()
    tied_others = ((s == s[true_idx]).sum()) - 1
    return int(better + tied_others + 1)

scores = np.array([0.9, 0.9, 0.9, 0.1])             # 모두 같은 점수를 낸 중복 항목들
assert rank_of_truth(scores, 0) == 3                # 엄격한 '>'만 쓰면 1위로 부풀려진다
strict = int((scores > scores[0]).sum() + 1)
assert strict == 1
print("엄격 '>' 순위", strict, "| ties-against-truth", rank_of_truth(scores, 0))
```
- 동점 규칙을 **미리 정해** 보고한다. 중복 항목(같은 텍스트·서열)이 갤러리에 있으면 엄격한 `>`는 최근접 이웃 baseline을 크게 부풀린다 (모듈 05: R@1이 1.5~2.3배). 점수를 반올림해 부동소수점 잡음에 의한 가짜 순서를 없앤다. 반올림 자릿수(`decimals`)는 **미리 정해 보고**한다. 반올림은 실제로 다른 점수를 동점으로 만들 수 있고 경계에서 갈릴 수도 있으므로, 다른 자릿수에서 결과가 바뀌는지 확인한다.

## 6. 짝지은 비교 + 그룹 부트스트랩
```python
def paired_delta_by_group(y, p1, p2, groups, metric=lambda a, b: spearmanr(a, b).statistic,
                          n_boot=1000, min_groups=8, seed=0):
    """Δ = metric(y, p1) − metric(y, p2). 상관 단위(groups)를 복원추출하는 조건부 구간.
    군집이 min_groups 미만이면 구간을 계산하지 않고 None을 돌려준다."""
    y, p1, p2, groups = map(np.asarray, (y, p1, p2, groups))
    delta = metric(y, p1) - metric(y, p2)          # 추정 대상: 풀링한 행 단위 지표의 차이 (군집이 크면 더 많이 반영된다)
    ug = np.unique(groups)
    if len(ug) < min_groups:
        return delta, None
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in pick])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = metric(y[ii], p1[ii]) - metric(y[ii], p2[ii])
        if np.isfinite(d):
            out.append(d)
    if len(out) < 0.9 * n_boot:                    # 정의되지 않은 재표본이 많으면 구간을 내지 않는다
        return delta, None
    return delta, (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

# 합성 자체 점검: 40개 군집, 군집마다 20개 행
rng = np.random.default_rng(2)
G = np.repeat(np.arange(40), 20)
y = rng.normal(size=800) + np.repeat(rng.normal(size=40), 20)
same_a = y + rng.normal(scale=1.0, size=800)
same_b = y + rng.normal(scale=1.0, size=800)
better = y + rng.normal(scale=0.3, size=800)
d0, ci0 = paired_delta_by_group(y, same_a, same_b, G)
d1, ci1 = paired_delta_by_group(y, better, same_b, G)
assert ci0[0] < 0 < ci0[1]                 # 같은 수준이면 구간이 0을 포함
assert ci1[0] > 0                          # 훨씬 정확하면 0을 포함하지 않음
assert paired_delta_by_group(y[:40], better[:40], same_b[:40], G[:40] // 20)[1] is None   # 군집 2개 → 계산 안 함
print("같은 수준:", round(d0, 3), [round(v, 3) for v in ci0], "| 더 정확:", round(d1, 3), [round(v, 3) for v in ci1])
```
- **조건부 구간**: 고정된 학습·시드를 두고 test 군집만 다시 뽑는다. 학습 변동·모델 초기화 변동은 반영하지 않는다고 적는다.
- 위 함수의 추정 대상은 *재표본된 행을 풀링한 지표*다(점수 눈금이 같은 한 모델의 예측을 같은 행에서 비교하므로 §3의 폴드 풀링 문제와는 다르다). 군집 크기가 매우 다르고 **군집별 지표의 평균**이 목표라면 군집마다 지표를 먼저 구해 평균하도록 바꾼다.
- **군집이 적으면 계산하지 않는다**: 군집 2개를 복원추출하면 가능한 재표본이 3가지뿐이다. 이때는 시드별 값과 일관성(이긴 시드 수)·사전 정한 효과 크기 규칙으로 판단한다 (모듈 07). `min_groups=8`은 이 과정의 **데모용 경계값**이다. 8개가 유효한 구간을 보장하지는 않으며, 군집 크기·의존성에 따라 더 많이 필요할 수 있다.
- 시드를 바꿔 같은 데이터를 다시 분할한 것은 test가 겹치므로 **독립 반복이 아니다**. 분할 민감도를 보는 것이다.
- "중앙값의 차이"와 "차이의 중앙값"은 다르다. 짝지은 비교는 후자(차이를 먼저 구한다).

## 7. 결정 규칙을 함수로
```python
def decide(deltas, ci=None, min_effect=0.02, min_wins=4):
    """시드별 Δ(모델 − baseline) → '지지' / '지지되지 않음'. 규칙은 결과를 보기 전에 정해 코드에 고정한다."""
    d = np.asarray([x for x in deltas if np.isfinite(x)])
    if len(d) == 0:
        return "판정 불가"
    ok = d.mean() >= min_effect and (d > 0).sum() >= min_wins
    if ci is not None:
        ok = ok and ci[0] > 0
    return "지지" if ok else "지지되지 않음"

assert decide([0.05, 0.04, 0.06, 0.03, 0.05]) == "지지"
assert decide([0.05, 0.04, -0.06, 0.03, -0.05]) == "지지되지 않음"
assert decide([np.nan, np.nan]) == "판정 불가"
```
- 임계값(0.02, 4/5)은 이 과정의 데모용이다. 실제 프로젝트에서는 **의미 있는 효과 크기**를 근거와 함께 사전에 정한다.

## 8. 보고 전 점검
- [ ] 지표 정의(폴드별/풀링, 동점 규칙, 갤러리 크기)를 적었다
- [ ] NaN을 값으로 바꾸지 않았다 (제외한 시드 수 보고)
- [ ] 비교는 같은 분할·같은 절차의 짝지은 차이다
- [ ] 구간은 무엇에 조건부인지, 군집 수가 충분했는지 적었다
- [ ] 무작위 기준선(AUPRC=양성 비율, 순열 분포)을 함께 보였다
- [ ] 음성 대조(라벨 섞기)는 **진단**일 뿐 누출 부재의 증명이 아니다 (평균이 0 근처인지와 시드 변동을 함께)
