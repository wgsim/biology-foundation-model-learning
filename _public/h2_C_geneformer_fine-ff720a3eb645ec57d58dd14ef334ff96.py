# 트랙 C 질문의 다른 절반 (C③): "Geneformer를 *미세조정*하면 동결 임베딩 위 head나 PCA head보다 기증자 단위로 나은가?"
# 모델: Geneformer V1-10M (Mac MPS에서 현실적인 유일한 크기다. V2-104M은 512토큰에서 약 17분/에폭(9,000세포)이라 13-fold LODO는 약 7시간이 걸려 하지 않았다)
# 데이터: 모듈 04의 Wilk 2020 PBMC 하위표본(캐시), 기증자 13명 leave-one-donor-out, 거친 라벨 13종. 분할은 트랙 C MVP·동결 임베딩 결과와 같다
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_C_geneformer_finetune.py` (캐시만 사용, 약 2시간)
import hashlib, importlib.metadata as md, json, pickle, subprocess, time, warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
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

# ---------- 0. 사전 등록 (작성자는 동결 임베딩 결과를 이미 보았다: V1 mean L-2 + logreg macro-F1 0.724, PCA 0.856; 미세조정 결과는 보기 전에 정했다) ----------
RUN_NAME = "h2_C_geneformer_finetune"
MAX_LEN = 512                           # 미세조정 입력 길이(상위 순위 유전자 512개). 동결 비교도 같은 길이로 다시 계산한다
EPOCHS, BS, LR_BACKBONE, LR_HEAD, WD, WARM = 2, 32, 5e-5, 1e-3, 0.01, 0.1       # 고정 설정; 튜닝하지 않았다(검증 기반 선택 없음)
PRIMARY = dict(model="V1-10M 미세조정(512토큰)", baselines=["PCA(HVG,50)+logreg", "V1 동결(512토큰)+logreg"], unit="donor(13)")
MIN_EFFECT, MIN_WINS, BOOT = 0.02, 9, 1000       # 트랙 C MVP와 같은 규칙: 기증자별 균형정확도 Δ 평균 ≥ 0.02, Δ>0 ≥ 9/13, 풀링 macro-F1 Δ 구간 하한 > 0
CTRL_DONORS_RAND = 5                    # 무작위 초기화 미세조정(양성 대조)은 비용 때문에 앞의 5개 기증자 fold만 돌린다
CTRL_DONORS_SHUF = 2                    # 훈련 라벨을 섞은 미세조정(음성 대조)은 2개 fold만 돌린다
C_LR, SEED = 1.0, 0
REV = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"
REPO = "ctheodoris/Geneformer"
V1 = dict(model="Geneformer-V1-10M", tok="geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl", med="geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl",
          map="geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl", max_len=2048)


def decide(deltas, ci):
    deltas = np.asarray(deltas, float)
    assert len(deltas) == 13, "MIN_WINS=9는 기증자 13명 기준이다"
    mean_d, wins = float(np.mean(deltas)), int(np.sum(deltas > 0))
    ok = mean_d >= MIN_EFFECT and wins >= MIN_WINS and (ci is None or ci[0] > 0)
    return ("지지" if ok else "지지되지 않음"), mean_d, wins


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
# 재개용 캐시는 설정·데이터 지문(KEY)이 붙은 디렉터리에 둔다. 하이퍼파라미터·토큰 길이·모델 리비전·데이터가 바뀌면 다른 디렉터리가 되어 옛 fold 결과가 섞이지 않는다
CONFIG = dict(max_len=MAX_LEN, epochs=EPOCHS, bs=BS, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, wd=WD, warmup=WARM, seed=SEED, rev=REV, ctrl=[CTRL_DONORS_RAND, CTRL_DONORS_SHUF],
              torch=torch.__version__, transformers=md.version("transformers"))
DATA_SHA = {f: sha256(SC / f) for f in ["pbmc_sub.h5ad", "geneformer_embeddings.npz"]}
KEY = hashlib.sha256((json.dumps(CONFIG, sort_keys=True) + DATA_SHA["pbmc_sub.h5ad"]).encode()).hexdigest()[:12]
CACHE = OUT / f"cache_{KEY}"
CACHE.mkdir(exist_ok=True)
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(), device=str(DEVICE),
            sha256={f: v[:16] + "…" for f, v in DATA_SHA.items()}, geneformer_repo_revision=REV, cache_key=KEY,
            config=dict(max_len=MAX_LEN, epochs=EPOCHS, bs=BS, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, wd=WD, warmup=WARM), primary=PRIMARY,
            packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "scanpy", "anndata", "torch", "transformers", "scipy"]})
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))
print("device:", DEVICE)

# ---------- 1. 데이터와 토큰화 (모듈 04/01의 rank value encoding을 옮김; V1 어휘) ----------
sub = ad.read_h5ad(SC / "pbmc_sub.h5ad")
obs = sub.obs
donors = obs.donor.values
dn = sorted(set(donors))
cls = sorted(set(obs.label))
y_all = np.array([cls.index(v) for v in obs.label])
counts = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
n_counts = obs.n_counts.values.astype(float)
var_ids = sub.var_names.values
assert len(dn) == 13 and all((donors == d).sum() == 800 for d in dn) and len(cls) == 13
print(f"세포 {sub.n_obs} | 기증자 {len(dn)} | 라벨 {len(cls)}종")


class SafeUnpickler(pickle.Unpickler):
    """허용 목록 unpickler(모듈 04/01과 같음): numpy dtype·scalar 외의 전역 객체는 차단한다. 고정한 저장소 리비전의 Geneformer 사전 파일에만 쓴다(근사 방어)."""
    ALLOWED = {("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global: {module}.{name}")


def load_pkl(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()


fetch = lambda rel: hf_hub_download(REPO, rel, revision=REV)
tok_d, med_d, map_d = load_pkl(fetch(V1["tok"])), load_pkl(fetch(V1["med"])), load_pkl(fetch(V1["map"]))
map_d = {k: v for k, v in map_d.items() if v in tok_d}
genes = [g for g in tok_d if g.startswith("ENS")]
gidx = {g: i for i, g in enumerate(genes)}
token_ids, median = np.array([tok_d[g] for g in genes]), np.array([med_d[g] for g in genes])
mapped = pd.Series(var_ids).str.upper().map(map_d)
keep_cols = np.where(mapped.notna().values)[0]
G = sp.csr_matrix((np.ones(len(keep_cols)), (np.arange(len(keep_cols)), [gidx[mapped.iloc[i]] for i in keep_cols])), shape=(len(keep_cols), len(genes)))
Xc = (counts[:, keep_cols] @ G).tocsr()
Xc.eliminate_zeros()
Xn = ((sp.diags(10_000 / n_counts) @ Xc) @ sp.diags(1.0 / median)).tocsr()
TOK_FULL, TOK = [], []
for i in range(Xn.shape[0]):
    s, e = Xn.indptr[i], Xn.indptr[i + 1]
    order = np.argsort(-Xn.data[s:e], kind="stable")
    t = token_ids[Xn.indices[s:e][order]].astype(np.int64)
    TOK_FULL.append(t[:V1["max_len"]])
    TOK.append(t[:MAX_LEN])
L = np.array([len(t) for t in TOK_FULL])
print(f"토큰화: 길이 중앙값 {np.median(L):.0f}, 2,048에서 잘린 세포 {np.mean(L >= 2048):.2f}, {MAX_LEN}토큰으로 자른 세포 {np.mean(L > MAX_LEN):.2f}")
# 점검: 캐시된 동결 임베딩과 같은 어휘·같은 순서로 토큰화했는지 확인하려고, 동결 V1 임베딩을 새로 계산해 캐시(2,048토큰)와 비교한다
cfg_dir = Path(fetch(V1["model"] + "/config.json")).parent
fetch(V1["model"] + "/model.safetensors")


def load_backbone(random_init=False):
    if random_init:
        return BertModel(BertConfig.from_pretrained(cfg_dir), add_pooling_layer=False)
    return BertModel.from_pretrained(cfg_dir, add_pooling_layer=False)


def pad_batch(tl):
    ml = max(len(t) for t in tl)
    ids = torch.zeros(len(tl), ml, dtype=torch.long)
    for j, t in enumerate(tl):
        ids[j, :len(t)] = torch.from_numpy(t)
    return ids


@torch.no_grad()
def embed_frozen(token_lists, layer=-2, batch=64):
    model = load_backbone().to(DEVICE).eval()
    order = np.argsort([len(t) for t in token_lists])
    out = np.zeros((len(token_lists), model.config.hidden_size), np.float32)
    for b in range(0, len(order), batch):
        bi = order[b:b + batch]
        ids = pad_batch([token_lists[i] for i in bi]).to(DEVICE)
        att = ids != 0
        hs = model(input_ids=ids, attention_mask=att.long(), output_hidden_states=True).hidden_states[layer]
        m = att.unsqueeze(-1).float()
        out[bi] = ((hs * m).sum(1) / m.sum(1)).cpu().numpy()
    del model
    torch.mps.empty_cache()
    return out


t = time.time()
E_cache = np.load(SC / "geneformer_embeddings.npz")["v1_mean_L-2"]
E512 = embed_frozen(TOK)
E2048 = embed_frozen(TOK_FULL[:200])
print(f"동결 임베딩(512토큰) {time.time() - t:.0f}s | 점검: 새로 계산한 2,048토큰 임베딩 200개 vs 노트북 04/01 캐시 최대 절대차 {np.abs(E2048 - E_cache[:200]).max():.2e}")

# ---------- 2. 동결 baseline: PCA(HVG,50), V1 동결 임베딩 + 로지스틱 (트랙 C와 같은 절차, 기증자 13명 LODO) ----------
a = sub.copy()
sc.pp.filter_genes(a, min_cells=3)
sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3")
hvg = a.var.highly_variable.values
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
b = a[:, hvg].copy()
sc.pp.scale(b, max_value=10)
sc.pp.pca(b, n_comps=50, random_state=SEED)
Z_PCA = b.obsm["X_pca"].copy()


def l2(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def split_static(Z, te):
    mu = Z[~te].mean(0)
    return l2(Z[~te] - mu), l2(Z[te] - mu)


def fold_pca_strict(te):
    """민감도: 유전자 필터(min_cells=3)·HVG(seurat_v3)·스케일·PCA를 모두 훈련 fold의 세포만으로 적합하고 test 기증자에 적용한다(위 주 PCA는 모든 기증자의 세포로 적합한 transductive 기준선)."""
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


def frozen_fold(Z, d):
    te = donors == d
    Ztr, Zte = Z(te) if callable(Z) else split_static(Z.astype(np.float64), te)
    scl = StandardScaler().fit(Ztr)
    pred = LogisticRegression(C=C_LR, max_iter=3000).fit(scl.transform(Ztr), y_all[~te]).predict(scl.transform(Zte))
    cm = np.zeros((len(cls), len(cls)))
    np.add.at(cm, (y_all[te], pred), 1)
    return cm, balanced_accuracy_score(y_all[te], pred)


def macro_f1(cm):
    tp = np.diag(cm)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)
    return float(np.nanmean(f1))


RUNS = {}
for name, Z in [("PCA(HVG,50)", Z_PCA), ("V1 동결(2,048토큰, 캐시)", E_cache), ("V1 동결(512토큰)", E512), ("PCA(HVG,50) 엄격 fold별", fold_pca_strict)]:
    out = Parallel(n_jobs=8)(delayed(frozen_fold)(Z, d) for d in dn)
    RUNS[name] = dict(cms=np.stack([o[0] for o in out]), bal=np.array([o[1] for o in out]))
    print(f"  {name:26s} 풀링 macro-F1 {macro_f1(RUNS[name]['cms'].sum(0)):.3f} | 기증자별 균형정확도 평균 {RUNS[name]['bal'].mean():.3f}", flush=True)


# ---------- 3. 미세조정 ----------
class Clf(nn.Module):
    """Geneformer V1 backbone + 평균 풀링(패딩 제외) + 드롭아웃 + 선형 head. (공식 BertForSequenceClassification은 첫 토큰 풀러를 쓰지만 V1에는 <cls>가 없어 평균 풀링으로 바꿨다)"""

    def __init__(self, backbone, ncls):
        super().__init__()
        self.backbone, self.drop, self.head = backbone, nn.Dropout(0.1), nn.Linear(backbone.config.hidden_size, ncls)

    def forward(self, ids):
        att = ids != 0
        h = self.backbone(input_ids=ids, attention_mask=att.long()).last_hidden_state
        m = att.unsqueeze(-1).float()
        return self.head(self.drop((h * m).sum(1) / m.sum(1)))


def finetune_fold(d, random_init=False, shuffle=False):
    te = donors == d
    tr_idx, te_idx = np.where(~te)[0], np.where(te)[0]
    seed = SEED + dn.index(d)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ytr = y_all[tr_idx].copy()
    if shuffle:
        ytr = rng.permutation(ytr)                                       # 훈련 라벨만 섞는다
    net = Clf(load_backbone(random_init), len(cls)).to(DEVICE)
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": LR_BACKBONE}, {"params": net.head.parameters(), "lr": LR_HEAD}], weight_decay=WD)
    spe = int(np.ceil(len(tr_idx) / BS))
    total, warm = spe * EPOCHS, max(1, int(WARM * spe * EPOCHS))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / (total - warm)))
    losses = []
    t0 = time.time()
    net.train()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(tr_idx))
        tot = 0.0
        for s in range(0, len(perm), BS):
            bi = perm[s:s + BS]
            ids = pad_batch([TOK[i] for i in tr_idx[bi]]).to(DEVICE)
            loss = nn.functional.cross_entropy(net(ids), torch.tensor(ytr[bi], device=DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(bi)
        losses.append(tot / len(perm))
    net.eval()
    pred = []
    with torch.no_grad():
        order = np.argsort([len(TOK[i]) for i in te_idx])
        out = np.zeros(len(te_idx), int)
        for s in range(0, len(order), 128):
            bi = order[s:s + 128]
            out[bi] = net(pad_batch([TOK[te_idx[i]] for i in bi]).to(DEVICE)).argmax(1).cpu().numpy()
    cm = np.zeros((len(cls), len(cls)))
    np.add.at(cm, (y_all[te_idx], out), 1)
    secs = time.time() - t0
    del net, opt
    torch.mps.empty_cache()
    return dict(donor=d, cm=cm.tolist(), bal=float(balanced_accuracy_score(y_all[te_idx], out)), losses=losses, seconds=secs)


LOG = CACHE / "finetune_results.jsonl"
DONE = {}
if LOG.exists():
    for line in LOG.read_text().splitlines():
        r = json.loads(line)
        DONE[(r["kind"], r["donor"])] = r
JOBS = [("ft", d) for d in dn] + [("rand", d) for d in dn[:CTRL_DONORS_RAND]] + [("shuf", d) for d in dn[:CTRL_DONORS_SHUF]]
for kind, d in JOBS:
    if (kind, d) in DONE:
        continue
    r = finetune_fold(d, random_init=(kind == "rand"), shuffle=(kind == "shuf"))
    r["kind"] = kind
    DONE[(kind, d)] = r
    with LOG.open("a") as f:
        f.write(json.dumps(r) + "\n")
    print(f"  [{kind:4s}] 기증자 {d}: 균형정확도 {r['bal']:.3f} | 훈련 손실 {', '.join(f'{x:.3f}' for x in r['losses'])} | {r['seconds']:.0f}s | 누적 {time.time() - T0:.0f}s", flush=True)

# ---------- 4. 보고 ----------
def pack(kind, donors_):
    return dict(cms=np.stack([np.array(DONE[(kind, d)]["cm"]) for d in donors_]), bal=np.array([DONE[(kind, d)]["bal"] for d in donors_]))


RUNS["V1 미세조정(512토큰)"] = pack("ft", dn)
BI = np.random.default_rng(SEED).integers(0, len(dn), size=(BOOT, len(dn)))


def compare(k1, k0, idx=None):
    r1, r0 = RUNS[k1], RUNS[k0]
    ii = np.arange(len(dn)) if idx is None else np.array(idx)
    d_bal = r1["bal"][ii] - r0["bal"][ii]
    bi = np.random.default_rng(SEED).integers(0, len(ii), size=(BOOT, len(ii)))
    f1d = np.array([macro_f1(r1["cms"][ii][i].sum(0)) - macro_f1(r0["cms"][ii][i].sum(0)) for i in bi])
    ci = np.percentile(f1d, [2.5, 97.5])
    pooled = macro_f1(r1["cms"][ii].sum(0)) - macro_f1(r0["cms"][ii].sum(0))
    return pooled, ci, d_bal


print("\n=== 결과 (기증자 13명 LODO, 거친 라벨 13종) ===")
for k, r in RUNS.items():
    print(f"  {k:26s} 풀링 macro-F1 {macro_f1(r['cms'].sum(0)):.3f} | 기증자별 균형정확도 평균 {r['bal'].mean():.3f} (최소 {r['bal'].min():.2f}, 최대 {r['bal'].max():.2f})")
print("\n=== 주 비교 ===")
for base in ["PCA(HVG,50)", "V1 동결(512토큰)", "V1 동결(2,048토큰, 캐시)", "PCA(HVG,50) 엄격 fold별"]:
    pooled, ci, d_bal = compare("V1 미세조정(512토큰)", base)
    dec, mean_d, wins = decide(d_bal, tuple(ci))
    tag = "PRIMARY" if base + "+logreg" in PRIMARY["baselines"] else ("민감도" if "엄격" in base else "참고")
    print(f"  [{tag}] 미세조정 − {base}: 풀링 macro-F1 Δ {pooled:+.3f} 95%[{ci[0]:+.3f},{ci[1]:+.3f}] | 기증자별 균형정확도 평균 Δ {mean_d:+.3f}, Δ>0 {wins}/{len(d_bal)} → {dec}")
print("\n=== 대조 ===")
RUNS["무작위 초기화 미세조정"] = pack("rand", dn[:CTRL_DONORS_RAND])
idx5 = list(range(CTRL_DONORS_RAND))
d_bal = RUNS["V1 미세조정(512토큰)"]["bal"][idx5] - RUNS["무작위 초기화 미세조정"]["bal"]
f1_ft5 = macro_f1(RUNS["V1 미세조정(512토큰)"]["cms"][idx5].sum(0))
f1_rand = macro_f1(RUNS["무작위 초기화 미세조정"]["cms"].sum(0))
print(f"양성 대조(사전학습 효과, 앞의 {CTRL_DONORS_RAND}개 기증자): 미세조정 풀링 macro-F1 {f1_ft5:.3f} vs 무작위 초기화 미세조정 {f1_rand:.3f} | 기증자별 균형정확도 Δ 평균 {d_bal.mean():+.3f}, Δ>0 {int((d_bal > 0).sum())}/{CTRL_DONORS_RAND} (사전 규칙: 5개 모두 양수여야 '사전학습이 미세조정에서도 의미 있음')")
sh = np.array([DONE[("shuf", d)]["bal"] for d in dn[:CTRL_DONORS_SHUF]])
print(f"음성 대조(훈련 라벨 섞기, {CTRL_DONORS_SHUF}개 fold): 기증자별 균형정확도 {', '.join(f'{x:.3f}' for x in sh)} (13종 균등 추측 {1 / len(cls):.3f})")
print("\n기증자별 균형정확도 (PCA | V1 동결 512 | V1 미세조정):")
print("  " + " ".join(f"{d}:{RUNS['PCA(HVG,50)']['bal'][i]:.2f}|{RUNS['V1 동결(512토큰)']['bal'][i]:.2f}|{RUNS['V1 미세조정(512토큰)']['bal'][i]:.2f}" for i, d in enumerate(dn)))
print(f"\n총 {time.time() - T0:.0f}초")
