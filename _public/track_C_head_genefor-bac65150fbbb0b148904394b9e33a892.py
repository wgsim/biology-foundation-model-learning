# 트랙 C 시험 실행: Geneformer 임베딩 위의 *지도* head vs PCA head — LODO(기증자 단위), 템플릿 구조로
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/track_C_head_geneformer_vs_pca.py` (캐시만 사용, 다운로드 없음)
import hashlib, importlib.metadata as md, json, os, subprocess, sys, time, warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path.cwd()                        # 저장소 루트에서 실행: .venv/bin/python -I <이 파일>
SC = ROOT / ".cache" / "singlecell"

# ---------- 0. 사전 등록 ----------
RUN_NAME = "dry_C_head_geneformer_vs_pca"
C_LR = 1.0                     # 로지스틱 회귀 규제(고정, test로 고르지 않는다)
PRIMARY = dict(model="V2 cls L-2", baseline="PCA(HVG,50)", labels="coarse", head="logreg")
MIN_EFFECT, MIN_WINS, BOOT = 0.02, 9, 1000      # 기증자 13명 중 Δ>0인 기증자 ≥ 9
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
            sha256={f: sha256(SC / f)[:16] + "…" for f in ["pbmc_sub.h5ad", "geneformer_embeddings.npz", "wilk2020_pbmc.h5ad"]},   # wilk2020: 세부 라벨 복원에 쓴다
            packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "scanpy", "anndata"]},
            primary=PRIMARY, C_LR=C_LR)
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME                  # git 제외 폴더: 실행 기록은 여기에만 쓴다
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))
print("provenance:", json.dumps(PROV, ensure_ascii=False)[:300], "...")

# ---------- 2. 데이터 점검 ----------
sub = ad.read_h5ad(SC / "pbmc_sub.h5ad")
E = dict(np.load(SC / "geneformer_embeddings.npz"))
assert all(v.shape[0] == sub.n_obs for v in E.values())
obs = sub.obs
donors = obs.donor.values
dn = sorted(set(donors))


def balanced_subsample(o, group="donor_id", cap=800, seed=0):
    """노트북 04/01 cell 7의 하위표본 추출을 그대로 재현한다 (캐시의 pbmc_sub에는 cell.type.fine이 없다)."""
    rng = np.random.default_rng(seed)
    idx = []
    for _, g in o.groupby(group, observed=True).indices.items():
        idx += list(rng.choice(g, size=min(cap, len(g)), replace=False))
    return np.sort(np.array(idx))


wobs = ad.read_h5ad(SC / "wilk2020_pbmc.h5ad", backed="r").obs[["donor_id", "cell.type.coarse", "cell_type", "cell.type.fine"]]
w = wobs.iloc[balanced_subsample(wobs)]
assert (w.donor_id.astype(str).values == obs.donor.values).all() and (w["cell.type.coarse"].astype(str).values == obs.label.values).all() \
    and (w.cell_type.astype(str).values == obs.cell_type_cxg.values).all(), "재현한 하위표본이 캐시와 다르다 — 세부 라벨을 붙일 수 없다"
fine = w["cell.type.fine"].astype(str).values
LAB = {}
for nm, arr in {"coarse": obs.label.values, "fine": fine}.items():
    cls = sorted(set(arr))
    LAB[nm] = (np.array([cls.index(v) for v in arr]), cls)
print(f"세포 {sub.n_obs} | 기증자 {len(dn)} | 거친 {len(LAB['coarse'][1])}종, 세부 {len(LAB['fine'][1])}종 | 임베딩 {sorted(E)[:3]}…")
assert len(set(donors)) == 13 and all((donors == d).sum() == 800 for d in dn)

# ---------- 4. 특징 ----------
a = sub.copy()
sc.pp.filter_genes(a, min_cells=3)
sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3")
hvg = a.var.highly_variable.values
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
LOGHVG = np.asarray(a[:, hvg].X.todense() if hasattr(a.X, "todense") else a[:, hvg].X, dtype=np.float32)       # fold별 PCA용(스케일 전)
b = a[:, hvg].copy()
sc.pp.scale(b, max_value=10)
sc.pp.pca(b, n_comps=50, random_state=SEED)
Z_PCA = b.obsm["X_pca"].copy()                       # 노트북 04/02와 같은 방식. 주 비교의 PCA 기준: HVG 선택·스케일·PCA를 *모든* 세포(held-out 기증자 포함)로 비지도 적합한다(transductive, 라벨은 쓰지 않음). 이는 PCA에 유리한 쪽의 설정이다. fold별 적합은 민감도에서 따로 본다
print(f"HVG {int(hvg.sum())} | PCA {Z_PCA.shape} | 준비 {time.time() - T0:.0f}s")


def fold_pca(te):
    """held-out 기증자를 *빼고* 스케일·PCA를 적합 (HVG 선택은 전체 세포로 했다는 한계는 남는다)."""
    tr = ~te
    mu, sd = LOGHVG[tr].mean(0), LOGHVG[tr].std(0) + 1e-8
    f = lambda M: np.clip((M - mu) / sd, -10, 10)
    p = PCA(n_components=50, random_state=SEED, svd_solver="randomized").fit(f(LOGHVG[tr]))
    Ztr, Zte = p.transform(f(LOGHVG[tr])), p.transform(f(LOGHVG[te]))
    mu_ = Ztr.mean(0)                                           # 주 비교(split_static)와 같은 후처리: 훈련 평균으로 중심화한 뒤 행 L2 정규화한다
    return l2(Ztr - mu_), l2(Zte - mu_)


def l2(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def split_static(Z, te):
    """훈련 기증자 평균으로 중심화한 뒤 행 L2 정규화한다 (노트북 04/02의 fold_prep)"""
    mu = Z[~te].mean(0)
    return l2(Z[~te] - mu), l2(Z[te] - mu)


FEATS = {
    "PCA(HVG,50)": lambda te: split_static(Z_PCA, te),
    "PCA(HVG,50) fold별": lambda te: fold_pca(te),
    "V2 cls L-2": lambda te: split_static(E["v2_cls_L-2"].astype(np.float64), te),
    "V1 mean L-2": lambda te: split_static(E["v1_mean_L-2"].astype(np.float64), te),
    "V1 random-init": lambda te: split_static(E["v1rand_mean_L-2"].astype(np.float64), te),
}


def knn_pred(Ztr, ytr, Zte, ncls, k=15):
    S = Zte @ Ztr.T
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    votes = np.zeros((len(Zte), ncls))
    rows = np.arange(len(Zte))
    for j in range(k):
        np.add.at(votes, (rows, ytr[idx[:, j]]), 1.0 + 1e-3 * S[rows, idx[:, j]])
    return votes.argmax(1)


def one_fold(feat, head, lab, d, shuffle_labels=False):
    y, cls = LAB[lab]
    te = donors == d
    Ztr, Zte = FEATS[feat](te)
    ytr = y[~te].copy()
    if shuffle_labels:
        ytr = np.random.default_rng(SEED).permutation(ytr)            # 음성 대조: 훈련 라벨을 섞는다
    if head == "knn":
        pred = knn_pred(Ztr, ytr, Zte, len(cls))
    else:
        sc_ = StandardScaler().fit(Ztr)
        clf = LogisticRegression(C=C_LR, max_iter=300).fit(sc_.transform(Ztr), ytr)
        pred = clf.predict(sc_.transform(Zte))
    cm = np.zeros((len(cls), len(cls)))
    np.add.at(cm, (y[te], pred), 1)
    return d, cm, balanced_accuracy_score(y[te], pred)


RUNS = {}


def run(feat, head, lab, shuffle=False):
    key = (feat, head, lab, shuffle)
    t = time.time()
    out = Parallel(n_jobs=8)(delayed(one_fold)(feat, head, lab, d, shuffle) for d in dn)
    cms = np.stack([o[1] for o in out])
    bal = np.array([o[2] for o in out])
    RUNS[key] = dict(cms=cms, bal=bal)
    print(f"  {feat:22s} {head:6s} {lab:6s} shuffle={shuffle!s:5s} 풀링 macro-F1 {macro_f1(cms.sum(0)):.3f} | 기증자별 균형정확도 평균 {bal.mean():.3f} | {time.time() - t:.0f}s", flush=True)


def macro_f1(cm):
    tp = np.diag(cm)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)
    return float(np.nanmean(f1))


BI = np.random.default_rng(SEED).integers(0, len(dn), size=(BOOT, len(dn)))


def compare(k1, k0):
    r1, r0 = RUNS[k1], RUNS[k0]
    d_bal = r1["bal"] - r0["bal"]                                     # 기증자별 Δ
    f1d = np.array([macro_f1(r1["cms"][i].sum(0)) - macro_f1(r0["cms"][i].sum(0)) for i in BI])
    ci = np.percentile(f1d, [2.5, 97.5])
    dec, mean_d, wins = decide(d_bal, tuple(ci))
    pooled = macro_f1(r1["cms"].sum(0)) - macro_f1(r0["cms"].sum(0))
    return f"풀링 macro-F1 Δ {pooled:+.3f} 95%[{ci[0]:+.3f},{ci[1]:+.3f}] | 기증자별 균형정확도 평균 Δ {mean_d:+.3f}, Δ>0 {wins}/{len(d_bal)} → {dec}"


# ---------- 5. 평가 ----------
print("\n[주 비교 세트: 거친 라벨 13종, 로지스틱 head]")
for f in ["PCA(HVG,50)", "V2 cls L-2", "V1 mean L-2", "V1 random-init"]:
    run(f, "logreg", "coarse")
run("V2 cls L-2", "logreg", "coarse", shuffle=True)
print("\n[민감도: 한 번에 하나]")
run("PCA(HVG,50) fold별", "logreg", "coarse")
run("PCA(HVG,50)", "knn", "coarse")
run("V2 cls L-2", "knn", "coarse")
run("PCA(HVG,50)", "logreg", "fine")
run("V2 cls L-2", "logreg", "fine")

P = lambda f, h="logreg", l="coarse", s=False: (f, h, l, s)
print("\n=== 주 비교 ===")
print("PRIMARY V2 cls L-2 vs PCA(HVG,50) [logreg, coarse]:", compare(P("V2 cls L-2"), P("PCA(HVG,50)")))
print("\n=== 대조 ===")
print("사전학습 효과 V1 mean L-2 vs V1 random-init:", compare(P("V1 mean L-2"), P("V1 random-init")))
print("V1 mean L-2 vs PCA:", compare(P("V1 mean L-2"), P("PCA(HVG,50)")))
cmS = RUNS[P("V2 cls L-2", s=True)]
print(f"음성 대조(훈련 라벨 섞기): 풀링 macro-F1 {macro_f1(cmS['cms'].sum(0)):.3f}, 기증자별 균형정확도 평균 {cmS['bal'].mean():.3f} (13종 균등 추측 ≈ {1 / 13:.3f})")
print("\n=== 민감도 (주 비교와 한 요인만 다름) ===")
print("PCA를 fold별로 적합 (PCA 기준을 바꿈): V2 cls L-2 vs PCA fold별:", compare(P("V2 cls L-2"), P("PCA(HVG,50) fold별")))
print("head = kNN(k=15) [zero-shot 방식]:", compare(P("V2 cls L-2", "knn"), P("PCA(HVG,50)", "knn")))
print("라벨 = 세부 20종:", compare(P("V2 cls L-2", "logreg", "fine"), P("PCA(HVG,50)", "logreg", "fine")))
print(f"\n총 {time.time() - T0:.0f}초")
