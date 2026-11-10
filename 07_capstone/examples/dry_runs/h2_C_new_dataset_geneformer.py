# 트랙 C 질문의 다른 절반 (C②): "새 데이터셋에서도 Geneformer V2 임베딩 위의 지도 head가 PCA head보다 기증자 단위로 나은가?"
# 새 데이터셋(승인 후 내려받음): CELLxGENE "Cross-sectional" — HIV 공동감염 내장리슈마니아증 연구(Commun Biol 2024; doi 10.1038/s42003-024-06225-2), PBMC 17,308세포, 기증자 10명, 5' 10x
# 파이프라인: 모듈 04/01의 rank value encoding·임베딩 추출을 그대로 옮겨(노트북 함수는 import할 수 없어 복사) 새 데이터에 적용 → 트랙 C MVP(logreg head vs PCA, LODO) 재현
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_C_new_dataset_geneformer.py` (약 25~40분, Mac MPS)
import hashlib, importlib.metadata as md, io, json, pickle, subprocess, time, warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from huggingface_hub import hf_hub_download
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from transformers import BertConfig, BertModel
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
T0 = time.time()
ROOT = Path.cwd()
SC = ROOT / ".cache" / "singlecell"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------- 0. 사전 등록 (작성자는 Wilk 데이터셋의 같은 비교 결과 — V2 < PCA — 를 이미 보았다; 이 데이터셋의 결과는 보기 전에 정했다) ----------
RUN_NAME = "h2_C_new_dataset_geneformer"
H5AD = SC / "cxg_cross_sectional.h5ad"
H5AD_SHA256 = "98a84e43ce2f531209bfdf5e640151c6fa27c676ccdec3b349cdeb4b0a7ce188"
CAP = 1000                         # 기증자당 최대 세포 수 (Wilk는 800; 이 데이터는 기증자 10명이고 773~3,711세포)
MIN_CLASS_CELLS = 100              # 하위표본에서 이 수 미만인 세포 유형은 평가에서 뺀다 (LODO에서 기증자 10명에게 나뉘면 평가가 불가능한 크기)
PRIMARY = dict(model="V2 cls L-2", baseline="PCA(HVG,50)", head="logreg", labels="cell_type(작은 유형 제외)", unit="donor")
MIN_EFFECT, BOOT = 0.02, 1000      # 기증자별 균형정확도 Δ의 평균 ≥ 0.02, 풀링 macro-F1 Δ의 기증자 부트스트랩 구간 하한 > 0
MIN_WIN_FRAC = 0.7                 # Δ>0인 기증자 비율 ≥ 0.7 (기증자 10명이면 7명)
C_LR, K_NN, SEED = 1.0, 15, 0
REV = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"      # 모듈 04/01이 쓴 Geneformer 저장소 커밋 (캐시에 있다)
REPO = "ctheodoris/Geneformer"
FILES = {
    "v1": dict(model="Geneformer-V1-10M", tok="geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl", med="geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl",
               map="geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl", max_len=2048, special=False),
    "v2": dict(model="Geneformer-V2-104M", tok="geneformer/token_dictionary_gc104M.pkl", med="geneformer/gene_median_dictionary_gc104M.pkl",
               map="geneformer/ensembl_mapping_dict_gc104M.pkl", max_len=4096, special=True),
}


def decide(deltas, ci):
    deltas = np.asarray(deltas, float)
    mean_d, wins = float(np.mean(deltas)), int(np.sum(deltas > 0))
    ok = mean_d >= MIN_EFFECT and wins >= np.ceil(MIN_WIN_FRAC * len(deltas)) and (ci is None or ci[0] > 0)
    return ("지지" if ok else "지지되지 않음"), mean_d, wins


# ---------- 1. 출처 ----------
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


assert sha256(H5AD) == H5AD_SHA256, "H5AD의 sha256이 내려받을 때 기록한 값과 다르다"
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(), device=str(DEVICE),
            data=dict(file=H5AD.name, sha256=H5AD_SHA256, bytes=H5AD.stat().st_size, source="https://datasets.cellxgene.cziscience.com/829fdf60-add0-4ca3-9b2f-dd86b3083885.h5ad",
                      collection="Persistent T cell unresponsiveness associated with chronic visceral leishmaniasis in HIV-coinfected patients (doi 10.1038/s42003-024-06225-2)",
                      license="같은 연구의 Zenodo 기록(10.5281/zenodo.10887674)은 CC BY 4.0; CELLxGENE 파일 자체의 라이선스 표기는 확인하지 못함 — 재배포하지 않는다"),
            geneformer_repo_revision=REV, cap=CAP, min_class_cells=MIN_CLASS_CELLS, primary=PRIMARY,
            packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "scanpy", "anndata", "torch", "transformers", "scipy"]})
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))
print("device:", DEVICE)

# ---------- 2. 데이터 점검과 하위표본 ----------
adata = ad.read_h5ad(H5AD)
X_raw = adata.raw.X.tocsr()
ids = adata.raw.var_names
sample = X_raw.data[:200000]
assert np.allclose(sample, np.round(sample)) and not ids.duplicated().any() and ids.str.startswith("ENSG").all(), "raw.X가 원시 카운트가 아니거나 유전자 ID가 Ensembl이 아니다"
print(f"원본 {adata.shape} | raw.X 정수형 | 세포당 총 카운트 중앙값 {np.median(np.asarray(X_raw.sum(1)).ravel()):.0f}")
obs0 = adata.obs
print(pd.concat([obs0.groupby("donor_id", observed=True).size().rename("세포"), obs0.groupby("donor_id", observed=True).disease.agg(lambda s: "|".join(sorted(set(map(str, s))))).rename("질병")], axis=1).to_string())
assert int(obs0.groupby("donor_id", observed=True).disease.nunique().max()) == 1, "기증자 안에서 질병이 섞여 있다"
print("→ 질병 상태는 기증자와 완전히 겹친다(교란), 모든 임베딩 비교는 기증자 단위로만 해석한다.")


def balanced_subsample(o, group="donor_id", cap=CAP, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = []
    for _, g in o.groupby(group, observed=True).indices.items():
        idx += list(rng.choice(g, size=min(cap, len(g)), replace=False))
    return np.sort(np.array(idx))


sel = balanced_subsample(obs0)
obs = obs0.iloc[sel][["donor_id", "disease", "cell_type", "is_primary_data"]].copy()
obs.columns = ["donor", "disease", "cell_type", "is_primary"]
for c in ["donor", "disease", "cell_type"]:
    obs[c] = obs[c].astype(str)
counts = X_raw[sel].tocsr()
n_counts = np.asarray(counts.sum(1)).ravel()
vc = obs.cell_type.value_counts()
small = vc[vc < MIN_CLASS_CELLS].index.tolist()
keep = ~obs.cell_type.isin(small).values
print(f"하위표본 {counts.shape[0]}세포 (기증자당 최대 {CAP}) | 세포 유형 {len(vc)}종 중 {MIN_CLASS_CELLS}세포 미만 {len(small)}종 제외({int((~keep).sum())}세포): {small}")
sel_k = np.where(keep)[0]
obs, counts, n_counts = obs.iloc[sel_k].reset_index(drop=True), counts[sel_k], n_counts[sel_k]
donors = obs.donor.values
dn = sorted(set(donors))
cls = sorted(set(obs.cell_type))
y_all = np.array([cls.index(v) for v in obs.cell_type])
print(f"평가 세포 {len(obs)} | 기증자 {len(dn)}명({', '.join(f'{d}:{(donors == d).sum()}' for d in dn)}) | 세포 유형 {len(cls)}종 | 비-primary 세포 비율 {1 - obs.is_primary.astype(bool).mean():.2f}")

# ---------- 3. Rank value encoding (노트북 04/01 §3과 같은 구현) ----------
# Geneformer의 유전자 사전은 pickle(.pkl)이다. 모듈 04/01과 같이 *허용 목록 unpickler*로만 읽는다: numpy dtype·scalar 외의 전역 객체는 차단한다. 파일은 고정한 저장소 리비전에서 받은 것이다(근사 방어; 신뢰할 수 없는 pickle에는 쓰지 말 것).
class SafeUnpickler(pickle.Unpickler):
    ALLOWED = {("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global: {module}.{name}")


def load_pkl(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()


def fetch(rel):
    return hf_hub_download(REPO, rel, revision=REV)


class RankTokenizer:
    def __init__(self, version):
        cfg = FILES[version]
        self.cfg = cfg
        self.tok = load_pkl(fetch(cfg["tok"]))
        self.med = load_pkl(fetch(cfg["med"]))
        mp = load_pkl(fetch(cfg["map"]))
        self.map = {k: v for k, v in mp.items() if v in self.tok}
        self.genes = [g for g in self.tok if g.startswith("ENS")]
        self.gidx = {g: i for i, g in enumerate(self.genes)}
        self.token_ids = np.array([self.tok[g] for g in self.genes])
        self.median = np.array([self.med[g] for g in self.genes])
        self.max_len, self.special = cfg["max_len"], cfg["special"]

    def collapse(self, counts, var_ids):
        mapped = pd.Series(np.asarray(var_ids)).str.upper().map(self.map)
        keep_ = np.where(mapped.notna().values)[0]
        cols = [self.gidx[mapped.iloc[i]] for i in keep_]
        G = sp.csr_matrix((np.ones(len(keep_)), (np.arange(len(keep_)), cols)), shape=(len(keep_), len(self.genes)))
        X = (counts[:, keep_] @ G).tocsr()
        X.eliminate_zeros()
        return X, len(keep_)

    def __call__(self, counts, var_ids, n_counts, target_sum=10_000):
        Xc, n_keep = self.collapse(counts, var_ids)
        Xn = (sp.diags(target_sum / n_counts) @ Xc) @ sp.diags(1.0 / self.median)
        Xn = Xn.tocsr()
        L = self.max_len - (2 if self.special else 0)
        out = []
        for i in range(Xn.shape[0]):
            s, e = Xn.indptr[i], Xn.indptr[i + 1]
            v, ix = Xn.data[s:e], Xn.indices[s:e]
            order = np.argsort(-v, kind="stable")
            t = self.token_ids[ix[order]][:L]
            if self.special:
                t = np.concatenate([[self.tok["<cls>"]], t, [self.tok["<eos>"]]])
            out.append(t.astype(np.int64))
        return out, n_keep


var_ids = adata.raw.var_names.values
tk = {v: RankTokenizer(v) for v in FILES}
tokens, stats = {}, {}
for v, t in tk.items():
    tokens[v], n_keep = t(counts, var_ids, n_counts)
    L = np.array([len(x) for x in tokens[v]])
    assert L.min() > (2 if t.special else 0), "어휘 유전자가 하나도 없는 세포가 있다"
    stats[v] = dict(어휘와_연결된_열=n_keep, 길이_중앙값=float(np.median(L)), 절단된_세포_비율=float((L >= t.max_len).mean()))
print("토큰화:", json.dumps(stats, ensure_ascii=False))

# ---------- 4. 임베딩 (노트북 04/01 §4와 같은 구현; 층 -2 평균·<cls>) ----------


@torch.no_grad()
def embed(model, token_lists, special, cls_id=None, eos_id=None, batch=32, layers=(-2,)):
    n, H = len(token_lists), model.config.hidden_size
    order = np.argsort([len(t) for t in token_lists])
    out = {("mean", l): np.zeros((n, H), np.float32) for l in layers}
    cls_out = {l: np.zeros((n, H), np.float32) for l in layers} if special else {}
    for b in range(0, n, batch):
        bi = order[b:b + batch]
        ml = max(len(token_lists[i]) for i in bi)
        ids_ = torch.zeros(len(bi), ml, dtype=torch.long)
        for j, i in enumerate(bi):
            ids_[j, :len(token_lists[i])] = torch.from_numpy(token_lists[i])
        ids_ = ids_.to(DEVICE)
        att = ids_ != 0
        hs = model(input_ids=ids_, attention_mask=att.long(), output_hidden_states=True).hidden_states
        gm = att.clone()
        if special:
            gm &= (ids_ != cls_id) & (ids_ != eos_id)
        m = gm.unsqueeze(-1).float()
        for l in layers:
            out[("mean", l)][bi] = ((hs[l] * m).sum(1) / m.sum(1)).float().cpu().numpy()
            if special:
                cls_out[l][bi] = hs[l][:, 0].float().cpu().numpy()
    res = dict(out)
    for l, v in cls_out.items():
        res[("cls", l)] = v
    return res


def load_model(version, random_init=False):
    path = Path(fetch(FILES[version]["model"] + "/config.json")).parent
    fetch(FILES[version]["model"] + "/model.safetensors")
    if random_init:
        torch.manual_seed(SEED)
        return BertModel(BertConfig.from_pretrained(path), add_pooling_layer=False).to(DEVICE).eval()
    return BertModel.from_pretrained(path, add_pooling_layer=False).to(DEVICE).eval()


EMB_FILE = OUT / "emb.npz"
KEY = hashlib.sha256(("|".join([H5AD_SHA256, REV, str(CAP), str(MIN_CLASS_CELLS), str(SEED)]) + "".join(obs.cell_type) + "".join(donors)).encode()).hexdigest()[:24]
emb = None
if EMB_FILE.exists():
    z = np.load(EMB_FILE)
    if "key" in z.files and str(z["key"]) == KEY:
        emb = {k: z[k] for k in z.files if k != "key"}
        print(f"캐시된 임베딩 사용: {sorted(emb)}")
if emb is None:
    emb = {}
    t0 = time.time()
    m1 = load_model("v1")
    test = tokens["v1"][:8]
    alone = np.vstack([embed(m1, [t], False, batch=1)[("mean", -2)] for t in test])
    batch_ = embed(m1, test, False, batch=8)[("mean", -2)]
    print(f"패딩 불변성: 단독 vs 배치 최대 절대차 {np.abs(alone - batch_).max():.2e} (임베딩 크기 중앙값 {np.median(np.abs(batch_)):.2f})")
    emb["v1_mean_L-2"] = embed(m1, tokens["v1"], False)[("mean", -2)]
    del m1
    print(f"V1-10M {time.time() - t0:.0f}s", flush=True)
    m1r = load_model("v1", random_init=True)
    emb["v1rand_mean_L-2"] = embed(m1r, tokens["v1"], False)[("mean", -2)]
    del m1r
    print(f"V1 무작위 초기화 {time.time() - t0:.0f}s", flush=True)
    m2 = load_model("v2")
    r = embed(m2, tokens["v2"], True, cls_id=tk["v2"].tok["<cls>"], eos_id=tk["v2"].tok["<eos>"])
    emb["v2_cls_L-2"], emb["v2_mean_L-2"] = r[("cls", -2)], r[("mean", -2)]
    del m2
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    print(f"V2-104M {time.time() - t0:.0f}s", flush=True)
    np.savez(EMB_FILE, key=np.array(KEY), **emb)
assert all(np.isfinite(v).all() for v in emb.values()), "임베딩에 NaN/inf가 있다"

# ---------- 5. PCA baseline (트랙 C와 같은 절차: HVG 2,000 → 정규화·로그 → 스케일 → PCA 50; 모든 세포로 비지도 적합 = transductive) ----------
a = ad.AnnData(X=counts.astype(np.float32), obs=obs.copy())
sc.pp.filter_genes(a, min_cells=3)
sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3")
hvg = a.var.highly_variable.values
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
LOGHVG = np.asarray(a[:, hvg].X.todense() if hasattr(a.X, "todense") else a[:, hvg].X, dtype=np.float32)
b = a[:, hvg].copy()
sc.pp.scale(b, max_value=10)
sc.pp.pca(b, n_comps=50, random_state=SEED)
Z_PCA = b.obsm["X_pca"].copy()
print(f"HVG {int(hvg.sum())} | PCA {Z_PCA.shape} | 준비 {time.time() - T0:.0f}s")


def l2(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def split_static(Z, te):
    mu = Z[~te].mean(0)
    return l2(Z[~te] - mu), l2(Z[te] - mu)


def fold_pca(te):
    tr = ~te
    mu, sd = LOGHVG[tr].mean(0), LOGHVG[tr].std(0) + 1e-8
    f = lambda M: np.clip((M - mu) / sd, -10, 10)
    p = PCA(n_components=50, random_state=SEED, svd_solver="randomized").fit(f(LOGHVG[tr]))
    Ztr, Zte = p.transform(f(LOGHVG[tr])), p.transform(f(LOGHVG[te]))
    mu_ = Ztr.mean(0)
    return l2(Ztr - mu_), l2(Zte - mu_)


def fold_pca_strict(te):
    """유전자 필터(min_cells=3)·HVG(seurat_v3)·스케일·PCA를 모두 훈련 fold의 세포만으로 적합한다(위 fold_pca는 HVG를 전체 세포로 고른 값을 그대로 쓴다)."""
    tr = ~te
    a_tr = ad.AnnData(X=counts[tr].astype(np.float32), var=pd.DataFrame(index=np.arange(counts.shape[1]).astype(str)))
    sc.pp.filter_genes(a_tr, min_cells=3)
    kept = a_tr.var_names.astype(int).values
    sc.pp.highly_variable_genes(a_tr, n_top_genes=2000, flavor="seurat_v3")
    cols = kept[a_tr.var.highly_variable.values]
    tot = np.asarray(counts[:, kept].sum(1)).ravel()
    Lg = np.log1p(counts[:, cols].toarray() / tot[:, None] * 1e4).astype(np.float32)
    mu, sd = Lg[tr].mean(0), Lg[tr].std(0) + 1e-8
    f = lambda M: np.clip((M - mu) / sd, -10, 10)
    p = PCA(n_components=50, random_state=SEED, svd_solver="randomized").fit(f(Lg[tr]))
    Ztr, Zte = p.transform(f(Lg[tr])), p.transform(f(Lg[te]))
    mu_ = Ztr.mean(0)
    return l2(Ztr - mu_), l2(Zte - mu_)


FEATS = {
    "PCA(HVG,50)": lambda te: split_static(Z_PCA, te),
    "PCA(HVG,50) fold별": lambda te: fold_pca(te),
    "PCA(HVG,50) 엄격 fold별": fold_pca_strict,
    "V2 cls L-2": lambda te: split_static(emb["v2_cls_L-2"].astype(np.float64), te),
    "V2 mean L-2": lambda te: split_static(emb["v2_mean_L-2"].astype(np.float64), te),
    "V1 mean L-2": lambda te: split_static(emb["v1_mean_L-2"].astype(np.float64), te),
    "V1 random-init": lambda te: split_static(emb["v1rand_mean_L-2"].astype(np.float64), te),
}


def knn_pred(Ztr, ytr, Zte, ncls, k=K_NN):
    S = Zte @ Ztr.T
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    votes = np.zeros((len(Zte), ncls))
    rows = np.arange(len(Zte))
    for j in range(k):
        np.add.at(votes, (rows, ytr[idx[:, j]]), 1.0 + 1e-3 * S[rows, idx[:, j]])
    return votes.argmax(1)


def one_fold(feat, head, d, shuffle_labels=False):
    te = donors == d
    Ztr, Zte = FEATS[feat](te)
    ytr = y_all[~te].copy()
    if shuffle_labels:
        ytr = np.random.default_rng(SEED).permutation(ytr)             # 훈련 라벨만 섞는다
    if head == "knn":
        pred = knn_pred(Ztr, ytr, Zte, len(cls))
    else:
        scl = StandardScaler().fit(Ztr)
        pred = LogisticRegression(C=C_LR, max_iter=3000).fit(scl.transform(Ztr), ytr).predict(scl.transform(Zte))
    cm = np.zeros((len(cls), len(cls)))
    np.add.at(cm, (y_all[te], pred), 1)
    return cm, balanced_accuracy_score(y_all[te], pred)


def macro_f1(cm):
    tp = np.diag(cm)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)
    return float(np.nanmean(f1))


RUNS = {}


def run(feat, head, shuffle=False):
    t = time.time()
    out = Parallel(n_jobs=8)(delayed(one_fold)(feat, head, d, shuffle) for d in dn)
    RUNS[(feat, head, shuffle)] = dict(cms=np.stack([o[0] for o in out]), bal=np.array([o[1] for o in out]))
    print(f"  {feat:20s} {head:6s} shuffle={shuffle!s:5s} 풀링 macro-F1 {macro_f1(RUNS[(feat, head, shuffle)]['cms'].sum(0)):.3f} | 기증자별 균형정확도 평균 {RUNS[(feat, head, shuffle)]['bal'].mean():.3f} | {time.time() - t:.0f}s", flush=True)


print("\n[주 비교 세트: 로지스틱 head]")
for f in ["PCA(HVG,50)", "V2 cls L-2", "V2 mean L-2", "V1 mean L-2", "V1 random-init"]:
    run(f, "logreg")
run("V2 cls L-2", "logreg", shuffle=True)
print("\n[민감도]")
run("PCA(HVG,50) fold별", "logreg")
run("PCA(HVG,50) 엄격 fold별", "logreg")
run("PCA(HVG,50)", "knn")
run("V2 cls L-2", "knn")

BI = np.random.default_rng(SEED).integers(0, len(dn), size=(BOOT, len(dn)))


def compare(k1, k0):
    r1, r0 = RUNS[k1], RUNS[k0]
    d_bal = r1["bal"] - r0["bal"]
    f1d = np.array([macro_f1(r1["cms"][i].sum(0)) - macro_f1(r0["cms"][i].sum(0)) for i in BI])
    ci = np.percentile(f1d, [2.5, 97.5])
    dec, mean_d, wins = decide(d_bal, tuple(ci))
    pooled = macro_f1(r1["cms"].sum(0)) - macro_f1(r0["cms"].sum(0))
    return f"풀링 macro-F1 Δ {pooled:+.3f} 95%[{ci[0]:+.3f},{ci[1]:+.3f}] | 기증자별 균형정확도 평균 Δ {mean_d:+.3f}, Δ>0 {wins}/{len(d_bal)} → {dec}"


P = lambda f, h="logreg", s=False: (f, h, s)
print("\n=== 주 비교 ===")
print(f"PRIMARY V2 cls L-2 vs PCA(HVG,50) [logreg, {len(cls)}종, 기증자 {len(dn)}명]:", compare(P("V2 cls L-2"), P("PCA(HVG,50)")))
print("\n=== 대조 ===")
print("양성 대조(사전학습 효과) V1 mean L-2 vs V1 random-init:", compare(P("V1 mean L-2"), P("V1 random-init")))
print("참고 V1 mean L-2 vs PCA:", compare(P("V1 mean L-2"), P("PCA(HVG,50)")))
cmS = RUNS[P("V2 cls L-2", s=True)]
print(f"음성 대조(훈련 라벨 섞기): 풀링 macro-F1 {macro_f1(cmS['cms'].sum(0)):.3f}, 기증자별 균형정확도 평균 {cmS['bal'].mean():.3f} ({len(cls)}종 균등 추측 ≈ {1 / len(cls):.3f})")
print("\n=== 민감도 (주 비교와 한 요인만 다름) ===")
print("PCA를 fold별로 적합(HVG는 전체 세포로 선택):", compare(P("V2 cls L-2"), P("PCA(HVG,50) fold별")))
print("PCA를 엄격하게 fold별 적합(필터·HVG·스케일·PCA 모두 훈련 fold만):", compare(P("V2 cls L-2"), P("PCA(HVG,50) 엄격 fold별")))
print("head = kNN(k=15) [zero-shot 방식]:", compare(P("V2 cls L-2", "knn"), P("PCA(HVG,50)", "knn")))
print("V2 평균 풀링 vs PCA:", compare(P("V2 mean L-2"), P("PCA(HVG,50)")))
print("\n기증자별 균형정확도 (PCA | V2 cls):")
print("  " + " ".join(f"{d}:{RUNS[P('PCA(HVG,50)')]['bal'][i]:.2f}|{RUNS[P('V2 cls L-2')]['bal'][i]:.2f}" for i, d in enumerate(dn)))
print(f"\n총 {time.time() - T0:.0f}초")
