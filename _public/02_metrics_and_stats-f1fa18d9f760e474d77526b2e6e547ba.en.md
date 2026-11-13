# Cheatsheet 02 — Metrics and Statistics (Paired Comparison · Group Bootstrap · Tie Handling)

> **English translation.** AI-assisted translation of the [Korean original](02_metrics_and_stats.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original. In the code blocks only the comments and message strings were translated; the executable code is unchanged.

This collects the calculations that were actually wrong, or easy to get wrong, in modules 01–07. The `python` blocks were **run and confirmed** on synthetic data (2026-11-06; numpy, scipy and scikit-learn from the repository `.venv`). The numbers from the synthetic data are meant to show how the code behaves, and are not biological results.

```python
import warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
```

## 1. A table for choosing the metric
| Task | Primary metric | What to look at together | Pitfall |
|---|---|---|---|
| Variant effect (continuous value) | Spearman ρ (per assay) | Per-type aggregation, the noise ceiling | You must not rank by a single average (module 02) |
| Binary classification, balanced | AUROC (**per fold**) | Accuracy, calibration | AUROC pooled over folds (§3 below) |
| Binary classification, rare positives | AUPRC | Positive fraction (random baseline) | The random baseline of AUPRC is not 0.5 but roughly the positive fraction (§4) |
| Multiclass, imbalanced | macro-F1 | Per-class recall | Accuracy is pulled toward the majority class |
| Retrieval/alignment | Recall@k, rank | Gallery size, tie rule | Tie handling (§5) |
| Batch correction (single-cell) | A preservation metric **and** a mixing metric | Consistency between metrics | A method that is high on only one (module 04) |

## 2. Constant prediction and Spearman
```python
rng = np.random.default_rng(0)
y = rng.normal(size=50)
const = np.zeros(50)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                # ConstantInputWarning
    rho_const = spearmanr(const, y).statistic
assert np.isnan(rho_const)                 # the Spearman of a constant prediction is undefined (NaN)
assert roc_auc_score(np.r_[0, 1, 0, 1], np.zeros(4)) == 0.5     # AUROC, on the other hand, counts ties as 1/2, so it gives a constant prediction 0.5 (the two metrics behave differently on constant input)
```
- **Do not replace an undefined value with −1 or 0 and put it in the average.** Record it as NaN and report the number of excluded seeds together (module 07 template: the mean prediction of the nearest train position became nearly constant at new positions, so the ρ of one seed was NaN). Replace NaN with the worst value only in the internal score used for hyperparameter selection.

## 3. AUROC — compute per fold and average
```python
# Two models that output a different constant in each fold: neither has any discriminative power in any fold. (roc_auc_score gives 0.5 for constant input)
y1 = np.array([0, 1] * 10); p1 = np.full(20, 0.2)
y2 = np.array([0, 1] * 10); p2 = np.full(20, 0.8)
per_fold = np.mean([roc_auc_score(y1, p1), roc_auc_score(y2, p2)])
pooled = roc_auc_score(np.r_[y1, y2], np.r_[p1, p2])
assert per_fold == 0.5
assert pooled == 0.5                       # balanced + the same ratio in both folds, so it is 0.5 by coincidence
# If the positive fraction differs by fold, pooling moves even a constant prediction away from 0.5.
y3 = np.array([0] * 18 + [1] * 2);  p3 = np.full(20, 0.2)     # 10% positives, low score
y4 = np.array([0] * 2 + [1] * 18);  p4 = np.full(20, 0.8)     # 90% positives, high score
per_fold2 = np.mean([roc_auc_score(y3, p3), roc_auc_score(y4, p4)])
pooled2 = roc_auc_score(np.r_[y3, y4], np.r_[p3, p4])
assert per_fold2 == 0.5 and abs(pooled2 - 0.9) < 1e-9     # a constant prediction, yet 0.9 when pooled
print("per-fold mean", per_fold2, "| pooled", round(pooled2, 3))
```
- The score scale can differ for each fold model, so **report the per-fold AUROC first**. An out-of-fold AUROC that pools the scores is a *different estimand* (a value that also mixes in the difference of scales between folds), so if you use it, define it that way and report it together with the per-fold values. While building module 05, a pooled AUROC once came out below 0.5 (0.460). That is an observation made during development, and the final notebook uses the per-fold mean.

## 4. The random baseline of AUPRC ≈ the positive fraction
```python
rng = np.random.default_rng(1)
y = (rng.random(20000) < 0.10).astype(int)       # 10% positives
ap = average_precision_score(y, rng.random(20000))
assert abs(ap - y.mean()) < 0.01
print("random AUPRC", round(ap, 3), "positive fraction", round(y.mean(), 3))
```
- `average_precision_score` is the **average precision (AP)** of a step-wise integral, and differs slightly from the trapezoidal area of the PR curve. The positive fraction is an *approximate* baseline for large samples and is not exactly equal to the expected AP in a finite sample. It fluctuates more in a subset with few positives (e.g., 21 positives). So report the **permutation distribution** of the random score together (module 03: TraitGym).

## 5. Tie handling in retrieval metrics — against the correct answer
```python
def rank_of_truth(scores, true_idx, decimals=6):
    """The rank of the correct answer (1 is the best). Other items with the same score as the correct one are counted as ahead of it (ties-against-truth)."""
    s = np.round(np.asarray(scores, dtype=float), decimals)
    better = (s > s[true_idx]).sum()
    tied_others = ((s == s[true_idx]).sum()) - 1
    return int(better + tied_others + 1)

scores = np.array([0.9, 0.9, 0.9, 0.1])             # duplicate items that all gave the same score
assert rank_of_truth(scores, 0) == 3                # using only a strict '>' inflates it to rank 1
strict = int((scores > scores[0]).sum() + 1)
assert strict == 1
print("strict '>' rank", strict, "| ties-against-truth", rank_of_truth(scores, 0))
```
- Decide the tie rule **in advance** and report it. If duplicate items (the same text or sequence) are in the gallery, a strict `>` greatly inflates the nearest-neighbor baseline (module 05: R@1 by 1.5–2.3 times). Rounding the scores removes fake orderings caused by floating-point noise. The number of rounding digits (`decimals`) is **decided and reported in advance**. Rounding can actually turn different scores into ties and may split at a boundary, so check whether the result changes at other numbers of digits.

## 6. Paired comparison + group bootstrap
```python
def paired_delta_by_group(y, p1, p2, groups, metric=lambda a, b: spearmanr(a, b).statistic,
                          n_boot=1000, min_groups=8, seed=0):
    """Δ = metric(y, p1) − metric(y, p2). A conditional interval that resamples the correlation unit (groups) with replacement.
    If there are fewer than min_groups clusters, it does not compute the interval and returns None."""
    y, p1, p2, groups = map(np.asarray, (y, p1, p2, groups))
    delta = metric(y, p1) - metric(y, p2)          # estimand: the difference of the pooled row-level metric (larger clusters are reflected more)
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
    if len(out) < 0.9 * n_boot:                    # if many resamples are undefined, do not give an interval
        return delta, None
    return delta, (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

# Synthetic self-check: 40 clusters, 20 rows per cluster
rng = np.random.default_rng(2)
G = np.repeat(np.arange(40), 20)
y = rng.normal(size=800) + np.repeat(rng.normal(size=40), 20)
same_a = y + rng.normal(scale=1.0, size=800)
same_b = y + rng.normal(scale=1.0, size=800)
better = y + rng.normal(scale=0.3, size=800)
d0, ci0 = paired_delta_by_group(y, same_a, same_b, G)
d1, ci1 = paired_delta_by_group(y, better, same_b, G)
assert ci0[0] < 0 < ci0[1]                 # at the same level, the interval includes 0
assert ci1[0] > 0                          # when much more accurate, it does not include 0
assert paired_delta_by_group(y[:40], better[:40], same_b[:40], G[:40] // 20)[1] is None   # 2 clusters → not computed
print("same level:", round(d0, 3), [round(v, 3) for v in ci0], "| more accurate:", round(d1, 3), [round(v, 3) for v in ci1])
```
- **Conditional interval**: with the training and seeds fixed, only the test clusters are redrawn. State that the variation from training and from model initialization is not reflected.
- The estimand of the function above is *the metric pooled over the resampled rows* (it compares the predictions of a model whose score scale is the same, on the same rows, so it differs from the fold-pooling problem of §3). If the cluster sizes differ greatly and the goal is **the mean of the per-cluster metrics**, change it so that the metric is computed per cluster first and then averaged.
- **If there are few clusters, do not compute it**: resampling 2 clusters with replacement gives only 3 possible resamples. In that case, judge from the per-seed values, the consistency (the number of seeds won) and the pre-set effect-size rule (module 07). `min_groups=8` is a **demonstration boundary value** of this course. 8 does not guarantee a valid interval, and more may be needed depending on the cluster sizes and the dependence.
- Re-splitting the same data with a different seed has overlapping test sets, so it is **not an independent repeat**. It looks at the sensitivity to the split.
- The "difference of medians" and the "median of differences" are different. A paired comparison is the latter (compute the differences first).

## 7. The decision rule as a function
```python
def decide(deltas, ci=None, min_effect=0.02, min_wins=4):
    """Per-seed Δ (model − baseline) → 'supported' / 'not supported'. Fix the rule in code before seeing the results."""
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
- The thresholds (0.02, 4/5) are for the demonstration in this course. In a real project, decide a **meaningful effect size** in advance, with grounds.

In the code above, the strings `"지지"`, `"지지되지 않음"` and `"판정 불가"` are the verdict labels used by the repository's scripts: "supported", "not supported" and "cannot judge". They were left untranslated because the repository's scripts compare against them.

## 8. Checks before reporting
- [ ] You wrote down the metric definition (per-fold/pooled, tie rule, gallery size)
- [ ] You did not replace NaN with a value (report the number of excluded seeds)
- [ ] The comparison is a paired difference on the same split with the same procedure
- [ ] You wrote what the interval is conditional on, and whether there were enough clusters
- [ ] You showed the random baseline together (AUPRC = positive fraction, permutation distribution)
- [ ] The negative control (shuffling labels) is only a **diagnostic** and not proof that there is no leakage (together with whether the mean is near 0 and the seed variation)

<!-- i18n-source: resources/cheatsheets/02_metrics_and_stats.md blob:eeb96f1f8dbaec376a7386f58675518126367d19 -->
