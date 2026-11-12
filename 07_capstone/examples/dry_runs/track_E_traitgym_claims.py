# 트랙 E 시험 실행 2단계: 논문 본문(bioRxiv v2 전문)에서 읽은 주장을 캐시된 점수로 직접 점검한다
# 논문: Benegas, Eraslan, Song, "Benchmarking DNA Sequence Models for Causal Regulatory Variant Prediction in Human Genetics" (bioRxiv 10.1101/2025.02.11.637758)
# 점검 대상 = 본문에서 *문장으로* 말한 것 중 Mendelian(`mendelian_traits_matched_9`) zero-shot 점수로 확인할 수 있는 것. 그림·표의 수치는 읽지 못했다.
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/track_E_traitgym_claims.py` (캐시와 `.cache/traitgym_published/`만 사용)
import hashlib, importlib.metadata as md, json, subprocess, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path.cwd()

# ---------- 0. 사전 등록 (작성자는 논문 본문을 읽은 뒤, 아래 점검의 결과를 보기 전에 이 문턱을 적었다. 단 K1·K2의 모델 순서는 앞선 L1 표에서 이미 보았고 K6의 τ(0.956)도 앞선 L3에서 보았다) ----------
RUN_NAME = "dry_E_traitgym_claims"
REV = "1fde19555fe8c0a55b1382bdf4c6f7082209f566"
DATASET = "mendelian_traits_matched_9"
BOOT = 1000                      # match_group 부트스트랩 횟수 (Δ의 구간)
SEED = 0
# 논문 본문 주장 (한국어 요약) → 점검 규칙
#  K1 "정렬 기반 zero-shot 점수(CADD·phyloP·GPN-MSA)가 가장 좋고 alignment-free 중 최고는 Evo2 40B이며, Evo2는 정렬 기반보다 뒤처진다"
#      → CADD, phyloP-241m, GPN-MSA LLR 각각이 evo2_40b_LLR보다 높다: 짝지은 부트스트랩 Δ의 95% 구간 하한 > 0
#      → 그리고 evo2_40b_LLR가 우리 캐시의 *비정렬 기반* 점수 11개(Evo2 1B·7B, NT LLR·|LLR|, HyenaDNA, Caduceus, GPN_final, Borzoi, Enformer, Sei) 중 최고 (SpeciesLM·GPN-Promoter 체크포인트는 캐시에 없다)
#  K2 "Evo2는 규모에 따라 뚜렷이 좋아진다": Δ(7B−1B), Δ(40B−7B) 둘 다 구간 하한 > 0
#  K3 "규모 효과는 proximal non-exonic·ncRNA에서 크고 distal non-exonic(enhancer 유사)에서는 크지 않다": 1B→40B 이득이
#      proximal ≥ 0.10 이고 ncRNA ≥ 0.10 이고 distal ≤ 0.05 (가중 평균 AUPRC, 공개 코드의 부분집합 정의)
#  K4 "distal non-exonic은 Mendelian 양성의 약 10%": 양성 중 비율이 [0.095, 0.105]
#  K5 "기준(무작위) AUPRC는 양성 비율 0.1": |338/3380 − 0.1| ≤ 0.001
#  K6 "전체 풀링 AUPRC로 바꿔도 zero-shot 결과가 비슷하다(보충 그림 S7)": 가중 평균 순위와 풀링 순위의 Kendall τ ≥ 0.90
#  (K5·K6은 이미 본 값의 확인이다: K5의 338/3,380은 L1에서, K6의 τ는 L3에서 이미 알고 있었다. 사전 예측이 아니다)
#  K7 (논문 B.4의 서술 vs 공개 코드) "zero-shot의 se는 양성·음성 집합 안에서 100회 부트스트랩": 그 방식으로 다시 계산한 se와 공개 CSV se의 비의 중앙값이 [0.85, 1.15]이면 '같은 방법으로 보임'
TAU_MIN, K3_PROX, K3_NCRNA, K3_DISTAL = 0.90, 0.10, 0.10, 0.05
ALIGN_BASED = ["CADD.plus.RawScore", "phyloP-241m.plus.score", "GPN-MSA_LLR.minus.score"]
ALIGN_FREE = ["evo2_1b_base_LLR.minus.score", "evo2_7b_LLR.minus.score", "evo2_40b_LLR.minus.score", "NucleotideTransformer_LLR.minus.score", "NucleotideTransformer_absLLR.plus.score",
              "HyenaDNA_LLR.minus.score", "Caduceus_LLR.minus.score", "GPN_final_LLR.minus.score", "Borzoi_L2_L2.plus.all", "Enformer_L2_L2.plus.all", "Sei.plus.seqclass_max_absdiff"]    # SpeciesLM·GPN-Promoter 별도 체크포인트는 캐시에 없다
NON_EXONIC = ["intergenic_variant", "intron_variant", "upstream_gene_variant", "downstream_gene_variant"]
CRE = ["PLS", "pELS", "dELS", "DNase-H3K4me3", "CTCF-only"]
NONEXONIC_FULL = NON_EXONIC + CRE + [f"{c}_flank" for c in CRE]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


snaps = sorted(p for p in (ROOT / ".cache/hf/hub/datasets--songlab--TraitGym/snapshots").iterdir() if p.is_dir())
assert [p.name for p in snaps] == [REV]
D = snaps[0] / DATASET
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "provenance.json").write_text(json.dumps(dict(
    run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(), dataset_revision=REV,
    test_parquet_sha256=sha256(D / "test.parquet")[:16] + "…", paper="bioRxiv 10.1101/2025.02.11.637758 v2 (본문 텍스트를 브라우저로 읽음; 저장소에 저장하지 않음)",
    packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "scipy"]}), ensure_ascii=False, indent=1))

tg = pd.read_parquet(D / "test.parquet")
y = tg.label.values.astype(int)
chrom = tg.chrom.astype(str).values
grp = tg.match_group.values


def load_score(model):
    for sign in ("plus", "minus"):
        tag = f".{sign}."
        if tag in model:
            feats, col = model.split(tag, 1)
            x = pd.read_parquet(D / "features" / f"{feats}.parquet", columns=[col])[col].values.astype(float)
            assert len(x) == len(tg) and not np.isnan(x).any()
            return -x if sign == "minus" else x
    raise ValueError(model)


MODELS = sorted(set(ALIGN_BASED + ALIGN_FREE + ["phastCons-43p.plus.score", "phyloP-100v.plus.score", "GPN-MSA_absLLR.plus.score", "Borzoi_L2_L2.plus.all",
                                                "Enformer_L2_L2.plus.all", "Sei.plus.seqclass_max_absdiff", "NucleotideTransformer_absLLR.plus.score"]))
S = {m: load_score(m) for m in MODELS}
PUB = {}
for p in sorted((ROOT / ".cache" / "traitgym_published" / "all").glob("*.csv")):
    d = pd.read_csv(p)
    PUB[d.model.iloc[0]] = (float(d.score.iloc[0]), float(d.se.iloc[0]))
assert set(MODELS) <= set(PUB)


def wavg_rows(yy, ss, cc):
    """공개 코드의 가중 평균 AUPRC: 염색체별 AP를 변이 수로 가중 (양성 없는 염색체는 AP 0.0)."""
    tot, acc = 0, 0.0
    for c in pd.unique(cc):
        m = cc == c
        acc += m.sum() * average_precision_score(yy[m], ss[m])
        tot += m.sum()
    return acc / tot


def subset_mask(name):
    if name == "ncRNA":
        return (tg.consequence == "non_coding_transcript_exon_variant").values
    nonex = tg.consequence.isin(NONEXONIC_FULL).values
    target = len(tg[tg.match_group == tg.match_group.iloc[0]])
    keep = (~tg.label.values.astype(bool)) | ((tg.tss_dist.values <= 1000) if name == "proximal" else (tg.tss_dist.values > 1000))
    sizes = pd.Series(keep).groupby(grp).transform("sum").values
    return nonex & keep & (sizes == target)


SUB = {n: subset_mask(n) for n in ["proximal", "ncRNA", "distal"]}
print(f"변이 {len(tg)} | 양성 {y.sum()} | 데이터셋 리비전 {REV[:10]}")

# ---------- 부트스트랩: match_group 단위, 모든 모델에 같은 재표본 ----------
KEY = ["CADD.plus.RawScore", "phyloP-241m.plus.score", "GPN-MSA_LLR.minus.score", "evo2_1b_base_LLR.minus.score", "evo2_7b_LLR.minus.score", "evo2_40b_LLR.minus.score"]
ug = np.unique(grp)
gidx = {g: np.where(grp == g)[0] for g in ug}
rs = np.random.RandomState(SEED)
DRAWS = [np.concatenate([gidx[g] for g in rs.choice(ug, len(ug), replace=True)]) for _ in range(BOOT)]
t = time.time()
BM = {m: np.array([wavg_rows(y[ii], S[m][ii], chrom[ii]) for ii in DRAWS]) for m in KEY}
print(f"부트스트랩(전체, 모델 {len(KEY)}개 × {BOOT}) {time.time() - t:.0f}s")
PT = {m: wavg_rows(y, S[m], chrom) for m in KEY}


def ci(a, b):
    d = BM[a] - BM[b]
    return PT[a] - PT[b], np.percentile(d, [2.5, 97.5])


short = lambda m: m.split(".")[0]
print("\n=== K1: 정렬 기반 zero-shot > Evo2 40B (짝지은 match_group 부트스트랩) ===")
k1 = []
for m in ALIGN_BASED:
    d, (lo, hi) = ci(m, "evo2_40b_LLR.minus.score")
    k1.append(lo > 0)
    print(f"  {short(m):18s} − evo2_40b: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] ({'하한 > 0' if lo > 0 else '하한 ≤ 0'})")
free_scores = {m: PT[m] if m in PT else wavg_rows(y, S[m], chrom) for m in ALIGN_FREE}
best_free = max(free_scores, key=free_scores.get)
print(f"  비정렬 기반 점수(우리 캐시에 있는 {len(ALIGN_FREE)}종): " + ", ".join(f"{short(m)} {v:.3f}" for m, v in sorted(free_scores.items(), key=lambda kv: -kv[1])))
K1 = all(k1) and best_free == "evo2_40b_LLR.minus.score"
print(f"  → K1: {'지지' if K1 else '지지되지 않음'} (우리 캐시의 점수·zero-shot 종류 안에서; 비정렬 기반 최고 = {short(best_free)}; SpeciesLM·GPN-Promoter 체크포인트는 캐시에 없어 비교하지 못했다)")

print("\n=== K2: Evo2 규모 효과 ===")
d71, (l71, h71) = ci("evo2_7b_LLR.minus.score", "evo2_1b_base_LLR.minus.score")
d407, (l407, h407) = ci("evo2_40b_LLR.minus.score", "evo2_7b_LLR.minus.score")
print(f"  7B − 1B: {d71:+.3f} [{l71:+.3f}, {h71:+.3f}] | 40B − 7B: {d407:+.3f} [{l407:+.3f}, {h407:+.3f}]")
K2 = l71 > 0 and l407 > 0
print(f"  → K2: {'지지' if K2 else '지지되지 않음'}")

print("\n=== K3: 규모 효과의 부분집합 차이 (1B → 40B, 가중 평균 AUPRC) ===")
gain = {}
for name, mask in SUB.items():
    ug_s = np.unique(grp[mask])
    gi = {g: np.where((grp == g) & mask)[0] for g in ug_s}
    rs3 = np.random.RandomState(100 + len(gain))
    gb = []
    for _ in range(BOOT):
        ii = np.concatenate([gi[g] for g in rs3.choice(ug_s, len(ug_s), replace=True)])
        gb.append(wavg_rows(y[ii], S["evo2_40b_LLR.minus.score"][ii], chrom[ii]) - wavg_rows(y[ii], S["evo2_1b_base_LLR.minus.score"][ii], chrom[ii]))
    g0 = wavg_rows(y[mask], S["evo2_40b_LLR.minus.score"][mask], chrom[mask]) - wavg_rows(y[mask], S["evo2_1b_base_LLR.minus.score"][mask], chrom[mask])
    gain[name] = g0
    lo, hi = np.percentile(gb, [2.5, 97.5])
    g7 = wavg_rows(y[mask], S["evo2_40b_LLR.minus.score"][mask], chrom[mask]) - wavg_rows(y[mask], S["evo2_7b_LLR.minus.score"][mask], chrom[mask])
    print(f"  (추가: {name}의 7B → 40B 이득 {g7:+.3f}; evo2_1b_base는 컨텍스트 8k의 base 체크포인트라 1B→40B에는 크기 말고도 체크포인트 종류 차이가 섞여 있을 수 있다)")
    print(f"  {name:9s} 변이 {int(mask.sum()):4d} 양성 {int(y[mask].sum()):3d} | 1B {wavg_rows(y[mask], S['evo2_1b_base_LLR.minus.score'][mask], chrom[mask]):.3f} → 40B {wavg_rows(y[mask], S['evo2_40b_LLR.minus.score'][mask], chrom[mask]):.3f} | 이득 {g0:+.3f} [{lo:+.3f}, {hi:+.3f}]")
K3 = gain["proximal"] >= K3_PROX and gain["ncRNA"] >= K3_NCRNA and gain["distal"] <= K3_DISTAL
print(f"  → K3 (proximal ≥ {K3_PROX}, ncRNA ≥ {K3_NCRNA}, distal ≤ {K3_DISTAL}): {'지지' if K3 else '지지되지 않음'}")

print("\n=== K4·K5: 데이터셋 구성 ===")
share = float(y[SUB['distal']].sum() / y.sum())
print(f"  distal non-exonic 양성 {int(y[SUB['distal']].sum())}/{int(y.sum())} = {share:.3f} → K4: {'지지' if 0.095 <= share <= 0.105 else '지지되지 않음'}")
print(f"  양성 비율 {y.mean():.4f} → K5(양성 비율만 확인; 이미 본 값): {'일치' if abs(y.mean() - 0.1) <= 0.001 else '불일치'}. 참고: 점수를 match_group 안에서 섞었을 때 이 지표(염색체 가중 AUPRC)의 평균은 0.124였다(앞선 실패 검출 ②) — 무작위 점수의 실제 기대 AUPRC가 양성 비율 0.1과 정확히 같지는 않다")

print("\n=== K6: 풀링 AUPRC로 바꿔도 순위가 비슷한가 (보충 그림 S7의 주장) ===")
wa = np.array([wavg_rows(y, S[m], chrom) for m in MODELS])
po = np.array([average_precision_score(y, S[m]) for m in MODELS])
tau = kendalltau(wa, po).statistic
print(f"  {len(MODELS)}개 점수의 Kendall τ(가중 평균 vs 풀링) = {tau:.3f} → K6(이미 본 값의 확인; 캐시된 17개 점수의 순위 유사성에 한정): {'일치' if tau >= TAU_MIN else '불일치'} (τ ≥ {TAU_MIN})")

print("\n=== K7: 공개 CSV의 se와 논문 B.4의 zero-shot 서술(양성·음성 집합 안에서 100회 부트스트랩)의 비교 — 이 서술을 우리가 해석해 구현한 세 가지와 비교 ===")
print("  (a) 전체에서 양성·음성을 따로 복원추출한 뒤 염색체 가중 AUPRC, (b) 염색체 안에서 양성·음성을 따로 복원추출한 뒤 염색체 가중 AUPRC, (c) 전체 양성·음성 복원추출 + 풀링 AUPRC(공개 `get_metric` 규칙; 다른 지표)")
rs7 = np.random.RandomState(7)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
RES_A = [np.concatenate([rs7.choice(pos, len(pos), replace=True), rs7.choice(neg, len(neg), replace=True)]) for _ in range(100)]
cp = {c: (np.where((chrom == c) & (y == 1))[0], np.where((chrom == c) & (y == 0))[0]) for c in pd.unique(chrom)}
RES_B = [np.concatenate([np.concatenate([rs7.choice(p_, len(p_), replace=True), rs7.choice(n_, len(n_), replace=True)]) for p_, n_ in cp.values()]) for _ in range(100)]
rows = []
for m in MODELS:
    se_a = float(np.std([wavg_rows(y[ii], S[m][ii], chrom[ii]) for ii in RES_A], ddof=1))
    se_b = float(np.std([wavg_rows(y[ii], S[m][ii], chrom[ii]) for ii in RES_B], ddof=1))
    se_c = float(np.std([average_precision_score(y[ii], S[m][ii]) for ii in RES_A], ddof=1))
    rows.append((short(m), PUB[m][1], se_a, se_b, se_c))
R7 = pd.DataFrame(rows, columns=["model", "공개 CSV se", "(a) 전체 양·음 복원추출", "(b) 염색체 안 양·음 복원추출", "(c) 풀링 AUPRC"]).set_index("model")
print(R7.round(4).to_string())
for col in ["(a) 전체 양·음 복원추출", "(b) 염색체 안 양·음 복원추출", "(c) 풀링 AUPRC"]:
    ratio = R7[col] / R7["공개 CSV se"]
    med = float(ratio.median())
    print(f"  {col}: 공개 CSV se 대비 비의 중앙값 {med:.2f} (최소 {ratio.min():.2f}, 최대 {ratio.max():.2f}) → {'비슷함' if 0.85 <= med <= 1.15 else '다름'} (사전 문턱 0.85~1.15)")
print("  → K7: 해석한 세 구현 모두 공개 CSV se와 모델별 크기가 다르면 '논문 서술 해석과 공개 CSV는 일치하지 않는다'이다. 논문이 실제로 어떻게 계산했는지(방법 동일성)는 이 비교로 판단할 수 없다. 공개 CSV의 se가 염색체 표 부트스트랩이라는 것은 workflow 코드를 읽어 확인했고 L1에서 재구현이 0.98~1.07배였다.")
print(f"\n총 {time.time() - T0:.0f}초")
