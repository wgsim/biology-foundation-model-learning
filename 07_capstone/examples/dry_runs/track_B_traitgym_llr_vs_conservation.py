# 트랙 B 시험 실행: TraitGym 캐시로 "DNA LM의 LLR이 보존 점수보다 나은가?"를 템플릿 구조로 평가
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/track_B_traitgym_llr_vs_conservation.py` (캐시만 사용, 다운로드 없음)
# 템플릿 대응: §0 사전 등록 → §1 출처 → §2 데이터 점검 → §3 단위/누출 → §4 점수(baseline·모델) → §5 평가 → §6 민감도 → 대조
import hashlib, importlib.metadata as md, json, os, subprocess, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
T0 = time.time()

# ---------- 0. 사전 등록 (결과를 보기 전에) ----------
RUN_NAME = "dry_B_traitgym_llr_vs_conservation"
DATASET = "mendelian_traits_matched_9"
# (파일, 열, 부호) — 부호는 데이터를 보기 전에 정한다: LLR은 −, 보존 점수는 +
SCORES = {
    "NT LLR": ("NucleotideTransformer_LLR", "score", -1),
    "Evo2-7B LLR": ("evo2_7b_LLR", "score", -1),
    "phyloP-241m": ("phyloP-241m", "score", +1),
    "phyloP-100v": ("phyloP-100v", "score", +1),
    "phastCons-43p": ("phastCons-43p", "score", +1),
}
PRIMARY = dict(model="NT LLR", baseline="phyloP-241m")
SECONDARY = dict(model="Evo2-7B LLR", baseline="phyloP-241m")
MIN_POS_CHROM = 5          # 염색체 단위 Δ를 계산하려면 양성이 이 수 이상
MIN_EFFECT = 0.02          # 실질적 의미가 있는 최소 Δ(AUPRC)
MIN_WINS = 10              # Δ > 0이어야 하는 염색체 수
MIN_GROUPS = 8             # 구간 계산에 필요한 match_group 수(여기서는 항상 충족)
BOOT, N_PERM = 500, 1000


def decide(deltas, ci):
    deltas = np.asarray(deltas, float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < MIN_WINS:
        return "판정 불가(유효 단위 부족)", float("nan"), 0
    mean_d, wins = float(np.mean(deltas)), int(np.sum(deltas > 0))
    ok = mean_d >= MIN_EFFECT and wins >= MIN_WINS and (ci is None or ci[0] > 0)
    return ("지지" if ok else "지지되지 않음"), mean_d, wins


# ---------- 1. 출처 ----------
ROOT = Path.cwd()
snaps = sorted(p for p in (ROOT / ".cache/hf/hub/datasets--songlab--TraitGym/snapshots").iterdir() if p.is_dir())
assert len(snaps) == 1, f"TraitGym 스냅샷이 {len(snaps)}개다 — 사용할 리비전을 명시해야 한다: {[p.name for p in snaps]}"
snap = snaps[0]
D = snap / DATASET


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


files = {"test.parquet": D / "test.parquet", **{k: D / "features" / f"{v[0]}.parquet" for k, v in SCORES.items()}}
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
PROV = dict(run=RUN_NAME, repo_commit=commit, dataset_revision=snap.name,
            sha256={k: sha256(p)[:16] + "…" for k, p in files.items()},
            packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "pyarrow"]},
            license="TraitGym: MIT (모듈 03 노트북 02의 기록)", primary=PRIMARY, secondary=SECONDARY)
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME                  # git 제외 폴더: 실행 기록은 여기에만 쓴다
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))
print("provenance:", json.dumps(PROV, ensure_ascii=False)[:400], "...")

# ---------- 2. 데이터 점검 ----------
tg = pd.read_parquet(D / "test.parquet")
y = tg.label.values.astype(int)
groups, chrom = tg.match_group.values, tg.chrom.astype(str).values
S = {}
for name, (f, c, sign) in SCORES.items():
    x = pd.read_parquet(D / "features" / f"{f}.parquet")
    assert len(x) == len(tg), f"{f}: 행 수가 변이표와 다름"
    S[name] = sign * x[c].values.astype(float)
assert all(not np.isnan(v).any() for v in S.values()), "NaN 있는 점수"
assert tg.groupby("match_group").label.sum().eq(1).all(), "그룹당 양성이 1개가 아니다"
assert tg.groupby("match_group").chrom.nunique().eq(1).all(), "그룹이 여러 염색체에 걸친다"
assert not tg.duplicated(["chrom", "pos", "ref", "alt"]).any(), "변이 중복"
assert tg.match_group.nunique() >= MIN_GROUPS, f"match_group이 {MIN_GROUPS}개 미만이라 군집 부트스트랩 구간을 믿을 수 없다"
print(f"변이 {len(tg)} | 양성 {y.sum()} | 그룹 {len(np.unique(groups))} | 염색체 {len(np.unique(chrom))}")

# ---------- 3. 단위 ----------
# 분할이 없다(zero-shot): 군집 단위 = match_group(부트스트랩), 일반화 단위 = 염색체(짝지은 Δ)
chrom_pos = pd.Series(y).groupby(chrom).sum()
CHROMS = [c for c in chrom_pos.index if chrom_pos[c] >= MIN_POS_CHROM]
print(f"염색체 단위 Δ에 쓸 염색체 {len(CHROMS)}개 (양성 ≥ {MIN_POS_CHROM}):", CHROMS)


# ---------- 5. 평가 ----------
def auprc(m, s=None):
    return average_precision_score(y[m], (S[s] if isinstance(s, str) else s)[m])


def chrom_delta(model, base, sm=None):
    sm = S[model] if sm is None else sm
    out = []
    for c in CHROMS:
        m = chrom == c
        out.append(average_precision_score(y[m], sm[m]) - average_precision_score(y[m], S[base][m]))
    return np.array(out)


def boot_delta(model, base, sm=None, B=BOOT, seed=0):
    sm = S[model] if sm is None else sm
    ug = np.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in ug}
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(B):
        ii = np.concatenate([idx[k] for k in rs.choice(ug, len(ug), replace=True)])
        vals.append(average_precision_score(y[ii], sm[ii]) - average_precision_score(y[ii], S[base][ii]))
    return np.percentile(vals, [2.5, 97.5])


G_BASE = np.lexsort((np.arange(len(y)), groups))      # match_group 블록 순서로 정렬한 인덱스


def perm_null(score, n_perm=N_PERM, seed=0):
    """점수 *값*을 match_group 안에서 섞어 풀링 AUPRC의 귀무분포를 만든다 (점수의 전체 분포·동점 구조는 그대로 유지)."""
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(n_perm):
        shuf = np.lexsort((rs.random_sample(len(y)), groups))   # 각 그룹 블록 안의 무작위 순열
        sc = np.empty(len(y))
        sc[G_BASE] = score[shuf]
        vals.append(average_precision_score(y, sc))
    return np.array(vals)


allm = np.ones(len(y), bool)
S["TSS 거리만"] = -tg.tss_dist.abs().values.astype(float)               # 비학습 대조: TSS 거리만
NULL = {name: perm_null(S[name], seed=i) for i, name in enumerate(S)}   # 점수별 귀무분포
rows = []
for name in S:
    a, nl = auprc(allm, name), NULL[name]
    rows.append(dict(score=name, auprc=a, flipped=auprc(allm, -S[name]), null_mean=nl.mean(), null_q95=np.percentile(nl, 95), perm_p=(1 + np.sum(nl >= a)) / (1 + len(nl))))
TAB = pd.DataFrame(rows)
print("\n점수별 귀무분포(그룹 내 점수 값 섞기 1,000회):")
print(TAB.round(3).to_string(index=False))
null = NULL[PRIMARY["model"]]                                           # 음성 대조의 비교 기준(주 비교 모델의 귀무분포)

print("\n--- 주 비교와 보조 비교 (Δ = AUPRC_model − AUPRC_baseline) ---")
RESULT = {}
for tag, cmp_ in [("PRIMARY", PRIMARY), ("SECONDARY", SECONDARY)]:
    d_ch = chrom_delta(cmp_["model"], cmp_["baseline"])
    ci = boot_delta(cmp_["model"], cmp_["baseline"])
    dec, mean_d, wins = decide(d_ch, tuple(ci))
    pooled = auprc(allm, cmp_["model"]) - auprc(allm, cmp_["baseline"])
    RESULT[tag] = dict(decision=dec, mean_delta_chrom=mean_d, wins=wins, n_chrom=len(d_ch), pooled_delta=pooled, ci=ci.round(3).tolist())
    print(f"{tag}: {cmp_['model']} vs {cmp_['baseline']} | 풀링 Δ {pooled:+.3f} 95%[{ci[0]:+.3f},{ci[1]:+.3f}] | 염색체 평균 Δ {mean_d:+.3f}, Δ>0 {wins}/{len(d_ch)} → {dec}")

# 음성 대조: 모델 점수를 match_group 안에서 섞는다 → 풀링 Δ가 (귀무 평균 − baseline)에 가깝고, 모델 AUPRC는 귀무 근처여야 한다
rs = np.random.RandomState(1)
gi = [np.where(groups == k)[0] for k in np.unique(groups)]
shuf = np.empty(len(y))
for idx in gi:
    shuf[idx] = rs.permutation(S[PRIMARY["model"]][idx])
print(f"\n음성 대조(모델 점수를 그룹 내 섞기): AUPRC {auprc(allm, shuf):.3f} (귀무 평균 {null.mean():.3f}) | 염색체 평균 Δ vs {PRIMARY['baseline']}: {np.mean(chrom_delta(PRIMARY['model'], PRIMARY['baseline'], shuf)):+.3f}")
# 양성 대조: 결정 규칙이 '지지'를 낼 수 있는지 — Evo2-7B LLR(AUPRC 0.38)은 TSS 거리만(≈ 귀무) 점수보다 훨씬 높으므로 '지지'여야 한다
d_pos = chrom_delta("Evo2-7B LLR", "TSS 거리만")
ci_pos = boot_delta("Evo2-7B LLR", "TSS 거리만")
dec_pos, mean_pos, wins_pos = decide(d_pos, tuple(ci_pos))
pooled_pos = auprc(allm, "Evo2-7B LLR") - auprc(allm, "TSS 거리만")
print(f"양성 대조(Evo2-7B LLR vs TSS 거리만): 풀링 Δ {pooled_pos:+.3f} 95%[{ci_pos[0]:+.3f},{ci_pos[1]:+.3f}] | 염색체 평균 Δ {mean_pos:+.3f}, Δ>0 {wins_pos}/{len(d_pos)} → {dec_pos}")
d_ref = chrom_delta("phyloP-100v", "phyloP-241m")                      # 참고: 같은 계열 보존 점수끼리의 차이
print(f"참고(phyloP-100v vs phyloP-241m): 염색체 평균 Δ {np.mean(d_ref):+.3f}, Δ>0 {int((d_ref > 0).sum())}/{len(d_ref)}")

# ---------- 6. 민감도: 한 번에 한 요인 ----------
print("\n--- 민감도 (주 비교, 한 번에 하나) ---")
for b in ["phyloP-100v", "phastCons-43p"]:
    d = chrom_delta(PRIMARY["model"], b)
    ci = boot_delta(PRIMARY["model"], b)
    dec, mean_d, wins = decide(d, tuple(ci))
    print(f"baseline={b}: 풀링 Δ {auprc(allm, PRIMARY['model']) - auprc(allm, b):+.3f} 95%[{ci[0]:+.3f},{ci[1]:+.3f}] | 염색체 평균 Δ {mean_d:+.3f} {wins}/{len(d)} → {dec}")
for mp in [3, 10]:
    CH = [c for c in chrom_pos.index if chrom_pos[c] >= mp]
    d = np.array([average_precision_score(y[chrom == c], S[PRIMARY['model']][chrom == c]) - average_precision_score(y[chrom == c], S[PRIMARY['baseline']][chrom == c]) for c in CH])
    print(f"MIN_POS_CHROM={mp}: 염색체 {len(CH)}개, 평균 Δ {d.mean():+.3f}, Δ>0 {int((d > 0).sum())}/{len(d)}")
print(f"\n총 {time.time() - T0:.0f}초")
