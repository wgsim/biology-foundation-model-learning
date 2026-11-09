# 트랙 C 질문의 다른 절반: "라벨 입도(거친/세부)에 따라 zero-shot 임베딩의 순위가 바뀌는가?" — 템플릿 구조로
# zero-shot = 동결 임베딩 + kNN(k=15, 코사인) head, 기증자 단위 leave-one-donor-out (모듈 04/02와 같은 설정)
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_C_label_granularity_zero_shot.py` (캐시만 사용, 다운로드 없음)
import hashlib, importlib.metadata as md, itertools, json, subprocess, time, warnings
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path.cwd()
SC = ROOT / ".cache" / "singlecell"

# ---------- 0. 사전 등록 (작성자는 트랙 C MVP에서 일부 결과를 이미 보았다: 로지스틱 head의 세부 라벨 결과, kNN 거친 라벨 결과) ----------
RUN_NAME = "h2_C_label_granularity_zero_shot"
K_NN = 15
REPS = ["PCA(HVG,50)", "V2 cls L-2", "V2 mean L-2", "V1 mean L-2", "V1 random-init"]   # 순위를 매길 표현(전부 동결, kNN)
GRAN = ["coarse", "cxg", "fine"]                                                          # 라벨 입도: 거친(13) < cellxgene(?) < 세부(20)
# 질문: 입도를 바꾸면 표현들의 순위가 바뀌는가?  '바뀐다'(= 지지)의 정의: 어떤 표현 쌍 (a, b)에 대해
#   Δ_거친 = F1(a) − F1(b)와 Δ_세부의 부호가 반대이고, 둘 다 기증자 부트스트랩 95% 구간이 0을 포함하지 않는다.
MIN_EFFECT, MIN_WINS, BOOT = 0.02, 9, 1000        # 양성 대조(사전학습 효과)용 기증자 단위 결정 규칙
SEED = 0


def decide(deltas, ci):
    deltas = np.asarray(deltas, float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < MIN_WINS:
        return "판정 불가(유효 단위 부족)", float("nan"), 0
    mean_d, wins = float(np.mean(deltas)), int(np.sum(deltas > 0))
    ok = mean_d >= MIN_EFFECT and wins >= MIN_WINS and (ci is None or ci[0] > 0)
    return ("지지" if ok else "지지되지 않음"), mean_d, wins


# ---------- 1. 출처 ----------
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            sha256={f: sha256(SC / f)[:16] + "…" for f in ["pbmc_sub.h5ad", "geneformer_embeddings.npz", "wilk2020_pbmc.h5ad"]},
            packages={k: md.version(k) for k in ["numpy", "scikit-learn", "scanpy", "anndata"]}, k_nn=K_NN, reps=REPS, granularities=GRAN)
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))

# ---------- 2. 데이터 점검 (트랙 C 스크립트와 같은 절차) ----------
sub = ad.read_h5ad(SC / "pbmc_sub.h5ad")
E = dict(np.load(SC / "geneformer_embeddings.npz"))
assert all(v.shape[0] == sub.n_obs for v in E.values())
obs = sub.obs
donors = obs.donor.values
dn = sorted(set(donors))


def balanced_subsample(o, group="donor_id", cap=800, seed=0):
    """노트북 04/01 cell 7의 하위표본 추출을 재현한다 (캐시의 pbmc_sub에는 cell.type.fine이 없다)."""
    rng = np.random.default_rng(seed)
    idx = []
    for _, g in o.groupby(group, observed=True).indices.items():
        idx += list(rng.choice(g, size=min(cap, len(g)), replace=False))
    return np.sort(np.array(idx))


wobs = ad.read_h5ad(SC / "wilk2020_pbmc.h5ad", backed="r").obs[["donor_id", "cell.type.coarse", "cell_type", "cell.type.fine"]]
w = wobs.iloc[balanced_subsample(wobs)]
assert (w.donor_id.astype(str).values == obs.donor.values).all() and (w["cell.type.coarse"].astype(str).values == obs.label.values).all() \
    and (w.cell_type.astype(str).values == obs.cell_type_cxg.values).all(), "재현한 하위표본이 캐시와 다르다"
assert len(set(donors)) == 13 and all((donors == d).sum() == 800 for d in dn)
LAB = {}
for nm, arr in {"coarse": obs.label.values, "cxg": obs.cell_type_cxg.values, "fine": w["cell.type.fine"].astype(str).values}.items():
    cls = sorted(set(arr))
    LAB[nm] = (np.array([cls.index(v) for v in arr]), cls)
print(f"세포 {sub.n_obs} | 기증자 {len(dn)} | 입도별 클래스 수: " + ", ".join(f"{g} {len(LAB[g][1])}" for g in GRAN))
# 클래스가 기증자 일부에만 있는 경우를 보고한다 (지표는 pooled confusion에서 계산하므로 문제는 아니지만 해석에 필요)
for g in GRAN:
    y, cls = LAB[g]
    per_donor = np.array([[np.sum((donors == d) & (y == c)) for c in range(len(cls))] for d in dn])
    print(f"  {g}: 클래스 최소 세포 수(전체) {np.bincount(y).min()}, 일부 기증자에 없는 클래스 {int((per_donor == 0).any(0).sum())}개, 기증자당 최소 {per_donor.min()}")

# ---------- 4. 특징 ----------
a = sub.copy()
sc.pp.filter_genes(a, min_cells=3)
sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3")
hvg = a.var.highly_variable.values
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
b = a[:, hvg].copy()
sc.pp.scale(b, max_value=10)
sc.pp.pca(b, n_comps=50, random_state=SEED)
Z_PCA = b.obsm["X_pca"].copy()                       # 모듈 04/02와 같은 방식: HVG·스케일·PCA를 모든 세포로 비지도 적합(transductive; PCA에 유리한 설정)


def l2(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def split_static(Z, te):
    """훈련 기증자 평균으로 중심화 → 행 L2 정규화 (노트북 04/02의 fold_prep)"""
    mu = Z[~te].mean(0)
    return l2(Z[~te] - mu), l2(Z[te] - mu)


FEATS = {
    "PCA(HVG,50)": Z_PCA.astype(np.float64),
    "V2 cls L-2": E["v2_cls_L-2"].astype(np.float64),
    "V2 mean L-2": E["v2_mean_L-2"].astype(np.float64),
    "V1 mean L-2": E["v1_mean_L-2"].astype(np.float64),
    "V1 random-init": E["v1rand_mean_L-2"].astype(np.float64),
}


def knn_pred(Ztr, ytr, Zte, ncls, k=K_NN):
    S = Zte @ Ztr.T
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    votes = np.zeros((len(Zte), ncls))
    rows = np.arange(len(Zte))
    for j in range(k):
        np.add.at(votes, (rows, ytr[idx[:, j]]), 1.0 + 1e-3 * S[rows, idx[:, j]])
    return votes.argmax(1)


def one_fold(feat, gran, d, shuffle_labels=False):
    y, cls = LAB[gran]
    te = donors == d
    Ztr, Zte = split_static(FEATS[feat], te)
    ytr = y[~te].copy()
    if shuffle_labels:
        ytr = np.random.default_rng(SEED).permutation(ytr)             # 음성 대조: 훈련 라벨을 섞는다
    pred = knn_pred(Ztr, ytr, Zte, len(cls))
    cm = np.zeros((len(cls), len(cls)))
    np.add.at(cm, (y[te], pred), 1)
    return cm, balanced_accuracy_score(y[te], pred)


def macro_f1(cm):
    tp = np.diag(cm)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)
    return float(np.nanmean(f1))


RUNS = {}


def run(feat, gran, shuffle=False):
    out = Parallel(n_jobs=8)(delayed(one_fold)(feat, gran, d, shuffle) for d in dn)
    RUNS[(feat, gran, shuffle)] = dict(cms=np.stack([o[0] for o in out]), bal=np.array([o[1] for o in out]))


for g in GRAN:
    for f in REPS:
        run(f, g)
    run("V2 cls L-2", g, shuffle=True)
print(f"실행 {time.time() - T0:.0f}s")

# ---------- 5. 평가 ----------
BI = np.random.default_rng(SEED).integers(0, len(dn), size=(BOOT, len(dn)))      # 기증자 재표본 (모든 비교에서 같은 인덱스 → 쌍 부트스트랩)


def f1_boot(feat, gran, shuffle=False):
    cms = RUNS[(feat, gran, shuffle)]["cms"]
    return np.array([macro_f1(cms[i].sum(0)) for i in BI])


def f1_pooled(feat, gran, shuffle=False):
    return macro_f1(RUNS[(feat, gran, shuffle)]["cms"].sum(0))


print("\n[입도별 풀링 macro-F1 (기증자 부트스트랩 95% 구간)]  — 표현을 행, 입도를 열로")
BOOTF = {(f, g): f1_boot(f, g) for f in REPS for g in GRAN}
hdr = "".join(f"{g}({len(LAB[g][1])}종)".rjust(26) for g in GRAN)
print(f"{'표현':16s}{hdr}")
for f in REPS:
    cells = []
    for g in GRAN:
        lo, hi = np.percentile(BOOTF[(f, g)], [2.5, 97.5])
        cells.append(f"{f1_pooled(f, g):.3f} [{lo:.3f},{hi:.3f}]".rjust(26))
    print(f"{f:16s}{''.join(cells)}")
print(f"{'(음성 대조) V2 라벨 섞기':16s}" + "".join(f"{f1_pooled('V2 cls L-2', g, True):.3f}".rjust(26) for g in GRAN))
print("\n[입도별 순위 (풀링 macro-F1 내림차순)]")
RANK = {}
for g in GRAN:
    RANK[g] = sorted(REPS, key=lambda f: -f1_pooled(f, g))
    print(f"  {g:7s}: " + " > ".join(RANK[g]))

print("\n[표현 쌍의 Δ(a − b)와 입도에 따른 변화] Δ는 풀링 macro-F1 차이, 구간은 기증자 부트스트랩(같은 재표본으로 쌍)")
FLIPS = []
rows = []
for a_, b_ in itertools.combinations(REPS, 2):
    d = {g: f1_pooled(a_, g) - f1_pooled(b_, g) for g in GRAN}
    ci = {g: np.percentile(BOOTF[(a_, g)] - BOOTF[(b_, g)], [2.5, 97.5]) for g in GRAN}
    sig = {g: (ci[g][0] > 0) or (ci[g][1] < 0) for g in GRAN}
    flip = d["coarse"] * d["fine"] < 0 and sig["coarse"] and sig["fine"]
    did = (BOOTF[(a_, "fine")] - BOOTF[(b_, "fine")]) - (BOOTF[(a_, "coarse")] - BOOTF[(b_, "coarse")])
    dci = np.percentile(did, [2.5, 97.5])
    if flip:
        FLIPS.append((a_, b_))
    rows.append((a_, b_, d, ci, flip, d["fine"] - d["coarse"], dci))
    print(f"  {a_:14s} − {b_:14s} | " + " | ".join(f"{g} {d[g]:+.3f} [{ci[g][0]:+.3f},{ci[g][1]:+.3f}]" for g in GRAN)
          + f" | 세부−거친 {d['fine'] - d['coarse']:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}]" + (" ← 부호가 뒤집히고 둘 다 구간이 0을 제외" if flip else ""))

dec = "지지(순위가 바뀜)" if FLIPS else "지지되지 않음(구간까지 고려한 순위 뒤집힘 없음)"
print(f"\n=== 주 질문: 입도에 따라 zero-shot 순위가 바뀌는가? → {dec} | 뒤집힌 쌍: {FLIPS}")
same_order = all(RANK[g] == RANK['coarse'] for g in GRAN)
print(f"점추정 순위가 세 입도에서 같은가: {same_order}")

print("\n=== 대조 ===")
for g in GRAN:
    d_bal = RUNS[("V1 mean L-2", g, False)]["bal"] - RUNS[("V1 random-init", g, False)]["bal"]
    ci = np.percentile(BOOTF[("V1 mean L-2", g)] - BOOTF[("V1 random-init", g)], [2.5, 97.5])
    dec_, mean_d, wins = decide(d_bal, tuple(ci))
    print(f"양성 대조 [{g}] 사전학습 효과 V1 − V1 무작위 초기화: 풀링 F1 Δ {f1_pooled('V1 mean L-2', g) - f1_pooled('V1 random-init', g):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}], 균형정확도 평균 Δ {mean_d:+.3f}, Δ>0 {wins}/{len(d_bal)} → {dec_}")

print("\n=== 라벨 위계 점검: 세부 라벨이 거친 라벨 안에 포함되는가 (입도 비교의 전제) ===")
fine_cls, coarse_cls = LAB["fine"][1], LAB["coarse"][1]
n_mixed = 0
for i, fc in enumerate(fine_cls):
    cs = np.bincount(LAB["coarse"][0][LAB["fine"][0] == i], minlength=len(coarse_cls))
    if cs.max() < cs.sum():
        n_mixed += 1
        print(f"  세부 '{fc}'는 거친 라벨 하나에 속하지 않는다: {dict((coarse_cls[j], int(c)) for j, c in enumerate(cs) if c)}")
print(f"  거친 라벨 하나에 포함되지 않는 세부 라벨: {n_mixed}/{len(fine_cls)}")
print(f"\n총 {time.time() - T0:.0f}초")
