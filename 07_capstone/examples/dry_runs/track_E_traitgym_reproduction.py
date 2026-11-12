# 트랙 E 시험 실행: TraitGym(Benegas et al. 2025) 논문 재현 — L1(공개 지표 CSV의 숫자 재현) + L3(순위가 지표 정의·부분집합을 바꿔도 유지되는가)
# 주장 단위(L1): `mendelian_traits_matched_9/AUPRC_by_chrom_weighted_average/{all, 일부 부분집합}/<모델>.csv`의 `score ± se`
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/track_E_traitgym_reproduction.py`
#   입력: 모듈 03이 받아 둔 TraitGym 점수 캐시 + `.cache/traitgym_published/`(공개 지표 CSV 22개, 합 2,058 B. 리비전을 고정해 승인 후 내려받았다)
import hashlib, importlib.metadata as md, json, subprocess, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path.cwd()

# ---------- 0. 사전 등록 (작성자는 모듈 03 노트북 02의 결과와 workflow 코드의 정의를 *읽은 뒤* 이 규칙을 적었다) ----------
RUN_NAME = "dry_E_traitgym_l1_l3"
REV = "1fde19555fe8c0a55b1382bdf4c6f7082209f566"          # 캐시와 공개 CSV가 모두 이 리비전이어야 한다
DATASET = "mendelian_traits_matched_9"
# L1: 공개 score와 재계산 score의 절대 차이 허용 오차. 계산이 결정적이므로 시드 변동이 없다(프로토콜 3의 시드 변동은 해당 없음).
# 공개 se(0.005~0.10, 가장 작은 se도 EPS의 5배 이상)보다 작은 값으로, 반올림이나 라이브러리 버전 차이는 허용하고 정의의 차이는 잡도록 *결과를 보기 전에* 정했다.
EPS = 1e-3
SE_REL = 0.15                                              # se는 numpy로 재구현한 부트스트랩이라 polars 난수열과 같지 않으므로 통계적 일치만 본다(판정에 쓰지 않음)
BOOT_SE = 1000                                             # 공개 코드의 bootstrap_se와 같은 횟수
SUBSETS = ["3_prime_UTR_variant", "5_prime_UTR_variant", "non_coding_transcript_exon_variant", "nonexonic_AND_distal", "nonexonic_AND_proximal"]
SUBSET_MODEL = "GPN_final_LLR.minus.score"                  # 공개 부분집합 CSV를 내려받은 모델 (캐시에 점수가 있다)
# L3: 공개 `all` 순위와 변형 순위의 Kendall τ ≥ TAU_MIN이면 '순위 유지'. 양성이 MIN_POS 미만인 부분집합은 순위가 잡음이라 판정하지 않는다.
TAU_MIN, MIN_POS = 0.80, 30
BOOT_RANK = 200                                            # 순위 잡음(match_group 부트스트랩) 반복 수
SEED = 0
NON_EXONIC = ["intergenic_variant", "intron_variant", "upstream_gene_variant", "downstream_gene_variant"]
CRE = ["PLS", "pELS", "dELS", "DNase-H3K4me3", "CTCF-only"]
NONEXONIC_FULL = NON_EXONIC + CRE + [f"{c}_flank" for c in CRE]


# ---------- 1. 출처 ----------
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


snaps = sorted(p for p in (ROOT / ".cache/hf/hub/datasets--songlab--TraitGym/snapshots").iterdir() if p.is_dir())
assert [p.name for p in snaps] == [REV], f"캐시의 TraitGym 리비전이 {REV}가 아니다: {[p.name for p in snaps]}"
D = snaps[0] / DATASET
PUB = ROOT / ".cache" / "traitgym_published"
pub_files = sorted(PUB.glob("*/*.csv"))
assert len(pub_files) == 22, f"공개 지표 CSV가 22개가 아니다: {len(pub_files)}"
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            dataset_revision=REV, published_sha256={f"{p.parent.name}/{p.name}": sha256(p)[:16] + "…" for p in pub_files},
            published_bytes=sum(p.stat().st_size for p in pub_files), test_parquet_sha256=sha256(D / "test.parquet")[:16] + "…",
            source=f"https://huggingface.co/datasets/songlab/TraitGym/resolve/{REV}/{DATASET}/AUPRC_by_chrom_weighted_average/",
            license="TraitGym 저장소·HF 데이터셋: MIT (API/카드 확인)", eps=EPS, tau_min=TAU_MIN, min_pos=MIN_POS,
            packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "scipy"]})
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))

# ---------- 2. 데이터 점검 ----------
tg = pd.read_parquet(D / "test.parquet")
y = tg.label.values.astype(int)
chrom = tg.chrom.astype(str).values
assert tg.groupby("match_group").label.sum().eq(1).all() and tg.groupby("match_group").chrom.nunique().eq(1).all()
print(f"변이 {len(tg)} | 양성 {y.sum()} | match_group {tg.match_group.nunique()} | 염색체 {tg.chrom.nunique()} | 캐시 데이터셋 리비전 {REV[:10]} (공개 CSV는 같은 리비전의 URL에서 받았다; 스크립트가 확인하는 것은 캐시 스냅샷 이름뿐)")

pub = {}
for p in pub_files:
    d = pd.read_csv(p)
    assert list(d.columns) == ["model", "metric", "score", "se"] and len(d) == 1 and d.metric.iloc[0] == "AUPRC", p
    pub[(p.parent.name, d.model.iloc[0])] = (float(d.score.iloc[0]), float(d.se.iloc[0]))
ALL_MODELS = sorted(m for (s, m) in pub if s == "all")
assert len(ALL_MODELS) == 17


def load_score(model):
    """공개 파일 이름 `<features>.<sign>.<feature>` → 캐시된 특징 열. 부호가 minus면 점수를 뒤집는다 (공개 코드 rule unsupervised_pred와 같음)."""
    for sign in ("plus", "minus"):
        tag = f".{sign}."
        if tag in model:
            feats, col = model.split(tag, 1)
            x = pd.read_parquet(D / "features" / f"{feats}.parquet", columns=[col])[col].values.astype(float)
            assert len(x) == len(tg) and not np.isnan(x).any()
            return -x if sign == "minus" else x
    raise ValueError(model)


S = {m: load_score(m) for m in ALL_MODELS}


# ---------- 3. 지표: 공개 workflow의 정의를 옮긴 것 ----------
def by_chrom_table(yy, ss, cc):
    """rule get_metric_by_block: 염색체마다 (n, AUPRC). 공개 코드는 V[chrom].unique() 전체를 쓰고 양성 없는 염색체를 따로 처리하지 않는다.
    이 경우 sklearn(1.9.1에서 확인)은 경고와 함께 AP = 0.0을 돌려주고 그 염색체도 n으로 가중되어 평균에 들어간다. 부분집합(교집합)에서 일어날 수 있어 그대로 따른다."""
    rows = []
    for c in pd.unique(cc):
        m = cc == c
        rows.append((m.sum(), average_precision_score(yy[m], ss[m])))
    return np.array(rows)


def weighted_avg(tab):
    """rule get_metric_by_block_weighted_average: 변이 수 가중 평균."""
    return float((tab[:, 0] / tab[:, 0].sum() * tab[:, 1]).sum())


def se_chrom_boot(tab, n_boot=BOOT_SE):
    """bootstrap_se: 염색체 표의 행을 복원추출해 가중 평균의 표준편차(ddof=1). 공개 코드는 polars 시드 i를 쓰고 여기서는 numpy 시드 i를 쓴다."""
    vals = []
    for i in range(n_boot):
        ix = np.random.RandomState(i).randint(0, len(tab), len(tab))
        vals.append(weighted_avg(tab[ix]))
    return float(np.std(vals, ddof=1))


# ---------- 4. L1: 공개 숫자 재현 ----------
print(f"\n=== L1: 공개 AUPRC_by_chrom_weighted_average 재현 (허용 오차 |Δ| ≤ {EPS}) ===")
rows = []
for m in ALL_MODELS:
    tab = by_chrom_table(y, S[m], chrom)
    sc, se = weighted_avg(tab), se_chrom_boot(tab)
    ps, pse = pub[("all", m)]
    rows.append(dict(model=m, 공개=ps, 재계산=sc, 차이=sc - ps, 공개_se=pse, 재계산_se=se, se_비=se / pse))
L1 = pd.DataFrame(rows).set_index("model")
print(L1.round(4).to_string())
ok_all = (L1["차이"].abs() <= EPS)
print(f"score: {int(ok_all.sum())}/17 허용 오차 안 | 최대 |차이| {L1['차이'].abs().max():.2e} | se: {int(((L1['se_비'] - 1).abs() <= SE_REL).sum())}/17이 ±{int(SE_REL * 100)}% 안 (참고: numpy 재구현이라 정확히 같을 수 없다)")

print("\n--- L1 (부분집합): GPN_final_LLR, 공개 부분집합 CSV 5개 ---")


def subset_mask(name):
    if name in ("3_prime_UTR_variant", "5_prime_UTR_variant", "non_coding_transcript_exon_variant"):
        return (tg.consequence == name).values
    nonex = tg.consequence.isin(NONEXONIC_FULL).values
    target = len(tg[tg.match_group == tg.match_group.iloc[0]])
    which = name.split("_AND_")[1]                       # proximal / distal: 공개 코드는 음성은 모두 두고 *양성*만 tss_dist(≤ / > 1,000)로 거른 뒤, 크기가 완전한 match_group만 남긴다
    keep = (~tg.label.values.astype(bool)) | ((tg.tss_dist.values <= 1000) if which == "proximal" else (tg.tss_dist.values > 1000))
    sizes = pd.Series(keep).groupby(tg.match_group.values).transform("sum").values
    prox = keep & (sizes == target)
    return nonex & prox                                   # `_AND_`: 두 부분집합의 교집합 (교집합 뒤 그룹 완결성은 다시 맞추지 않는다)


SUBMASK = {s: subset_mask(s) for s in SUBSETS}
TARGET = int((tg.match_group == tg.match_group.iloc[0]).sum())
for s in SUBSETS:                                         # 교집합 뒤 match_group이 온전한가 (공개 코드는 교집합 뒤 다시 맞추지 않는다)
    mm = SUBMASK[s]
    g = pd.DataFrame({"g": tg.match_group.values[mm], "y": y[mm]}).groupby("g").agg(n=("y", "size"), pos=("y", "sum"))
    print(f"  [{s}] match_group {len(g)}개 중 크기 {TARGET}·양성 1개로 온전한 것 {int(((g.n == TARGET) & (g.pos == 1)).sum())}개, 양성이 없는 것 {int((g.pos == 0).sum())}개")
rows = []
for s in SUBSETS:
    m = SUBMASK[s]
    tab = by_chrom_table(y[m], S[SUBSET_MODEL][m], chrom[m])
    sc = weighted_avg(tab)
    ps, pse = pub[(s, SUBSET_MODEL)]
    rows.append(dict(subset=s, 변이수=int(m.sum()), 양성=int(y[m].sum()), 공개=ps, 재계산=sc, 차이=sc - ps, 공개_se=pse))
L1S = pd.DataFrame(rows).set_index("subset")
print(L1S.round(4).to_string())
ok_sub = (L1S["차이"].abs() <= EPS)
print(f"부분집합 score: {int(ok_sub.sum())}/5 허용 오차 안 | 최대 |차이| {L1S['차이'].abs().max():.2e}")
VERDICT = "재현됨" if (ok_all.all() and ok_sub.all()) else "재현되지 않음"
print(f"\n=== L1 판정(사전 규칙: 22개 값 모두 |Δ| ≤ {EPS}): {VERDICT} ===")

# ---------- 5. 대조 (규칙이 실제로 실패할 수 있는가) ----------
print("\n=== 대조 ('실패 검출' 대조: 일부러 틀린 입력·정의에서 검사가 실패하는가. L1이 통과한 것이 '통과 가능' 쪽의 확인이다) ===")
flip = {m: -S[m] for m in ALL_MODELS}                       # 부호를 뒤집은 점수: 공개 값과 달라야 한다
d_flip = np.array([weighted_avg(by_chrom_table(y, flip[m], chrom)) - pub[("all", m)][0] for m in ALL_MODELS])
print(f"실패 검출 ①(부호 뒤집기): 부호를 뒤집은 점수 17개 중 |Δ| > {EPS}인 모델 {int((np.abs(d_flip) > EPS).sum())}/17, 최소 |Δ| {np.abs(d_flip).min():.3f}")
rs = np.random.RandomState(SEED)
gi = [np.where(tg.match_group.values == g)[0] for g in tg.match_group.unique()]
shuf = {}
for m in ALL_MODELS:
    sc = np.empty(len(y))
    for ix in gi:
        sc[ix] = S[m][ix][rs.permutation(len(ix))]
    shuf[m] = weighted_avg(by_chrom_table(y, sc, chrom))
d_sh = np.array([shuf[m] - pub[("all", m)][0] for m in ALL_MODELS])
print(f"실패 검출 ②(match_group 안에서 점수 섞기): 섞은 점수의 가중 AUPRC 평균 {np.mean(list(shuf.values())):.3f} (양성 비율 {y.mean():.3f}), 공개 값과 |Δ| > {EPS}인 모델 {int((np.abs(d_sh) > EPS).sum())}/17")
alt = {m: average_precision_score(y, S[m]) for m in ALL_MODELS}
d_pool = np.array([alt[m] - pub[("all", m)][0] for m in ALL_MODELS])
macro = {m: float(by_chrom_table(y, S[m], chrom)[:, 1].mean()) for m in ALL_MODELS}
d_macro = np.array([macro[m] - pub[("all", m)][0] for m in ALL_MODELS])
print(f"실패 검출 ③(정의가 다른 지표는 공개 값과 달라야 한다): 풀링 AUPRC |Δ| 최대 {np.abs(d_pool).max():.3f}(> {EPS}인 모델 {int((np.abs(d_pool) > EPS).sum())}/17), 염색체 동일 가중 평균 |Δ| 최대 {np.abs(d_macro).max():.3f}(> {EPS}인 모델 {int((np.abs(d_macro) > EPS).sum())}/17)")

# ---------- 6. L3: 순위가 유지되는가 ----------
print(f"\n=== L3: 공개 `all` 순위 vs 변형 순위 (Kendall τ ≥ {TAU_MIN}이면 '순위 유지'; 양성 < {MIN_POS}이면 판정하지 않음) ===")
PUB_SCORE = np.array([pub[("all", m)][0] for m in ALL_MODELS])
order_pub = [ALL_MODELS[i] for i in np.argsort(-PUB_SCORE)]
print("공개 순위(상위 5):", ", ".join(f"{m.split('.')[0]} {pub[('all', m)][0]:.3f}" for m in order_pub[:5]), "| 하위 3:", ", ".join(f"{m.split('.')[0]} {pub[('all', m)][0]:.3f}" for m in order_pub[-3:]))


def rank_scores(mask, fn):
    return np.array([fn(y[mask], S[m][mask], chrom[mask]) for m in ALL_MODELS])


VARIANTS = {
    "재계산 가중 평균(L1과 같음)": (np.ones(len(y), bool), lambda yy, ss, cc: weighted_avg(by_chrom_table(yy, ss, cc))),
    "풀링 AUPRC": (np.ones(len(y), bool), lambda yy, ss, cc: average_precision_score(yy, ss)),
    "염색체 동일 가중 평균": (np.ones(len(y), bool), lambda yy, ss, cc: float(by_chrom_table(yy, ss, cc)[:, 1].mean())),
}
for s in SUBSETS:
    VARIANTS[f"부분집합 {s} · 풀링"] = (SUBMASK[s], lambda yy, ss, cc: average_precision_score(yy, ss))        # 사전 지정: 부분집합은 염색체당 양성이 너무 적어 풀링 AUPRC로 순위를 매긴다 (기준 순위는 가중 평균이므로 부분집합과 지표가 함께 바뀐다)
for s in SUBSETS:
    VARIANTS[f"부분집합 {s} · 가중 평균(리뷰 후 추가)"] = (SUBMASK[s], lambda yy, ss, cc: weighted_avg(by_chrom_table(yy, ss, cc)))   # 교차 리뷰의 지적 뒤 추가: 기준과 같은 지표로 부분집합만 바꾼 변형
grp_all = tg.match_group.values
L3 = []
for name, (mask, fn) in VARIANTS.items():
    sc_v = rank_scores(mask, fn)
    tau = kendalltau(PUB_SCORE, sc_v).statistic
    npos = int(y[mask].sum())
    # 순위 잡음: 같은 변형을 match_group 부트스트랩으로 다시 계산한 순위와 점추정 순위의 τ (표본 변동만으로 τ가 이만큼 내려간다)
    rs = np.random.RandomState(1000 + len(L3))
    ug = np.unique(grp_all[mask])
    idx = {g: np.where((grp_all == g) & mask)[0] for g in ug}
    taus = []
    for _ in range(BOOT_RANK):
        ii = np.concatenate([idx[g] for g in rs.choice(ug, len(ug), replace=True)])
        sc_b = np.array([fn(y[ii], S[m][ii], chrom[ii]) if len(np.unique(chrom[ii])) > 0 else np.nan for m in ALL_MODELS])
        taus.append(kendalltau(sc_v, sc_b).statistic)
    noise = float(np.nanmean(taus))
    top1 = ALL_MODELS[int(np.argmax(sc_v))].split(".")[0]
    verdict = "판정 안 함(양성 부족)" if npos < MIN_POS else ("순위 유지" if tau >= TAU_MIN else "순위 유지되지 않음")
    L3.append({"변형": name, "변이수": int(mask.sum()), "양성": npos, "τ_vs_공개": round(float(tau), 3), "순위잡음_τ": round(noise, 3), "1위": top1, "판정": verdict})
print(pd.DataFrame(L3).set_index("변형").to_string())
print("(순위잡음_τ: 같은 변형을 match_group 부트스트랩으로 다시 계산한 순위와 그 변형의 점추정 순위 사이의 평균 τ = 변형 *내부*의 순위 안정성이다. τ_vs_공개가 어떤 분포에서 나오는지(귀무분포)는 아니다.)")
print(f"\n총 {time.time() - T0:.0f}초")
