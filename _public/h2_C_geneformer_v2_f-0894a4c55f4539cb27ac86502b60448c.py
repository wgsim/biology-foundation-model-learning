# 트랙 C 질문의 다른 절반 (C③-V2): "Geneformer V2-104M을 *미세조정*하면 동결 임베딩 위 head나 PCA head보다 기증자 단위로 나은가?"
# C③(h2_C_geneformer_finetune.py, V1-10M, Mac)과 같은 데이터·분할·하이퍼파라미터·결정 규칙이고 모델만 V2-104M으로 바꿨다. Mac에서는 약 17분/에폭이라 Colab GPU에서 학습한다.
# 단계(H2_STAGE): check  = (Mac) 토큰화 점검 — 4,096토큰 동결 <cls> 임베딩이 모듈 04/01 캐시와 같은지
#                bench  = (Mac·Colab) fold 하나의 앞 N 스텝으로 스텝당 시간·메모리·손실 측정 (fp32, CUDA면 fp16도)
#                train  = (Colab) 512토큰 동결 임베딩 저장 + 20개 fold 미세조정. fold마다 결과를 저장하고 H2_DRIVE_DIR로 복사한다(재개 가능)
#                report = (Mac) 저장된 fold 결과 + PCA·동결 baseline으로 보고서
# 환경변수: H2_DATA(pbmc_sub.h5ad가 있는 폴더, 기본 .cache/singlecell), H2_OUT(결과 폴더, 기본 .cache/capstone_runs/<RUN_NAME>), H2_DRIVE_DIR(선택), H2_AMP=1(fp16 autocast, CUDA 전용), H2_BENCH_STEPS
# 실행(Mac): 저장소 루트에서 `H2_STAGE=report HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_C_geneformer_v2_finetune.py`
import hashlib, importlib.metadata as md, json, os, pickle, shutil, sys, time, warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from sklearn.metrics import balanced_accuracy_score
from transformers import BertConfig, BertModel
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
T0 = time.time()
STAGE = os.environ.get("H2_STAGE", "report")
assert STAGE in ("check", "bench", "train", "report")
DATA = Path(os.environ.get("H2_DATA", ".cache/singlecell"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
AMP = os.environ.get("H2_AMP", "0") == "1"
assert not AMP or DEVICE.type == "cuda", "fp16 autocast는 CUDA에서만 쓴다"

# ---------- 0. 사전 등록 (작성자는 V1-10M 미세조정 결과를 이미 보았다: 미세조정 − PCA −0.077, 1/13; V2 동결 <cls> L-2 + logreg 0.838, PCA 0.856. V2 미세조정 결과는 보기 전에 정했다) ----------
RUN_NAME = "h2_C_geneformer_v2_finetune"
MAX_LEN = 512                            # <cls> + 상위 순위 유전자 510개 + <eos>
EPOCHS, BS, LR_BACKBONE, LR_HEAD, WD, WARM = 2, 32, 5e-5, 1e-3, 0.01, 0.1       # V1 실행과 같은 고정 설정; 튜닝하지 않았다(검증 기반 선택 없음)
POOL = "cls"                             # V2에는 <cls> 토큰이 있다(V1 실행은 평균 풀링). 주 설정 하나만 돌린다
PRIMARY = dict(model="V2-104M 미세조정(512토큰, <cls>)", baselines=["PCA(HVG,50)", "V2 동결 <cls> L-2 (512토큰)"], unit="donor(13)")
MIN_EFFECT, MIN_WINS, BOOT = 0.02, 9, 1000       # 트랙 C MVP와 같은 규칙
CTRL_DONORS_RAND, CTRL_DONORS_SHUF = 5, 2        # V1 실행과 같은 수
C_LR, SEED = 1.0, 0
REV = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"
REPO = "ctheodoris/Geneformer"
V2 = dict(model="Geneformer-V2-104M", tok="geneformer/token_dictionary_gc104M.pkl", med="geneformer/gene_median_dictionary_gc104M.pkl", map="geneformer/ensembl_mapping_dict_gc104M.pkl", max_len=4096)
OUT = Path(os.environ.get("H2_OUT", f".cache/capstone_runs/{RUN_NAME}"))
OUT.mkdir(parents=True, exist_ok=True)
DRIVE = os.environ.get("H2_DRIVE_DIR")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def decide(deltas, ci):
    deltas = np.asarray(deltas, float)
    assert len(deltas) == 13, "MIN_WINS=9는 기증자 13명 기준이다"
    mean_d, wins = float(np.mean(deltas)), int(np.sum(deltas > 0))
    ok = mean_d >= MIN_EFFECT and wins >= MIN_WINS and (ci is not None and ci[0] > 0)
    return ("지지" if ok else "지지되지 않음"), mean_d, wins


# ---------- 1. 데이터와 토큰화 (모듈 04/01의 rank value encoding을 옮김; V2 어휘, <cls>/<eos>) ----------
DATA_SHA = sha256(DATA / "pbmc_sub.h5ad")
sub = ad.read_h5ad(DATA / "pbmc_sub.h5ad")
obs = sub.obs
donors = obs.donor.values
dn = sorted(set(donors))
cls = sorted(set(obs.label))
y_all = np.array([cls.index(v) for v in obs.label])
counts = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
n_counts = obs.n_counts.values.astype(float)
var_ids = sub.var_names.values
assert len(dn) == 13 and all((donors == d).sum() == 800 for d in dn) and len(cls) == 13


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
tok_d, med_d, map_d = load_pkl(fetch(V2["tok"])), load_pkl(fetch(V2["med"])), load_pkl(fetch(V2["map"]))
assert tok_d["<pad>"] == 0, "패딩 id가 0이라는 가정(attention mask = ids != 0)이 깨졌다"
CLS_ID, EOS_ID = tok_d["<cls>"], tok_d["<eos>"]
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


def wrap(t, L):
    return np.concatenate([[CLS_ID], t[:L - 2], [EOS_ID]]).astype(np.int64)


RANKED = []
for i in range(Xn.shape[0]):
    s, e = Xn.indptr[i], Xn.indptr[i + 1]
    RANKED.append(token_ids[Xn.indices[s:e][np.argsort(-Xn.data[s:e], kind="stable")]])
TOK = [wrap(t, MAX_LEN) for t in RANKED]
NG = np.array([len(t) for t in RANKED])
TOK_SHA = hashlib.sha256(b"".join(t.tobytes() for t in TOK)).hexdigest()
print(f"device {DEVICE} | stage {STAGE} | AMP {AMP} | 세포 {sub.n_obs} | 기증자 {len(dn)} | 라벨 {len(cls)}종 | 어휘와 연결된 열 {len(keep_cols)} | 유전자 수 중앙값 {np.median(NG):.0f}, "
      f"{MAX_LEN}토큰으로 잘린 세포 {np.mean(NG > MAX_LEN - 2):.2f} | 토큰 sha {TOK_SHA[:12]}", flush=True)

CONFIG = dict(model=V2["model"], max_len=MAX_LEN, epochs=EPOCHS, bs=BS, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, wd=WD, warmup=WARM, pool=POOL, seed=SEED, rev=REV,
              ctrl=[CTRL_DONORS_RAND, CTRL_DONORS_SHUF], amp=AMP)
cfg_dir = Path(fetch(V2["model"] + "/config.json")).parent
fetch(V2["model"] + "/model.safetensors")


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


def empty_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def embed_frozen(token_lists, layer=-2, batch=64):
    """동결 임베딩: 층 -2의 <cls> 위치와 유전자 위치 평균(<cls>·<eos>·패딩 제외). 항상 fp32."""
    model = load_backbone().to(DEVICE).eval()
    order = np.argsort([len(t) for t in token_lists])
    H = model.config.hidden_size
    out_c, out_m = np.zeros((len(token_lists), H), np.float32), np.zeros((len(token_lists), H), np.float32)
    for b in range(0, len(order), batch):
        bi = order[b:b + batch]
        ids = pad_batch([token_lists[i] for i in bi]).to(DEVICE)
        att = ids != 0
        hs = model(input_ids=ids, attention_mask=att.long(), output_hidden_states=True).hidden_states[layer]
        gm = (att & (ids != CLS_ID) & (ids != EOS_ID)).unsqueeze(-1).float()
        out_c[bi] = hs[:, 0].float().cpu().numpy()
        out_m[bi] = ((hs * gm).sum(1) / gm.sum(1)).float().cpu().numpy()
    del model
    empty_cache()
    return out_c, out_m


# ---------- 2. 미세조정 ----------
class Clf(nn.Module):
    """Geneformer V2 backbone + <cls> 위치 출력 + 드롭아웃 + 선형 head."""

    def __init__(self, backbone, ncls):
        super().__init__()
        self.backbone, self.drop, self.head = backbone, nn.Dropout(0.1), nn.Linear(backbone.config.hidden_size, ncls)

    def forward(self, ids):
        h = self.backbone(input_ids=ids, attention_mask=(ids != 0).long()).last_hidden_state
        return self.head(self.drop(h[:, 0]))


def finetune_fold(d, random_init=False, shuffle=False, amp=AMP, max_steps=None):
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
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses, step_losses, step = [], [], 0
    t0 = time.time()
    net.train()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(tr_idx))
        tot = 0.0
        for s in range(0, len(perm), BS):
            bi = perm[s:s + BS]
            ids = pad_batch([TOK[i] for i in tr_idx[bi]]).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = nn.functional.cross_entropy(net(ids).float(), torch.tensor(ytr[bi], device=DEVICE))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(bi)
            step_losses.append(loss.item())
            step += 1
            if max_steps and step >= max_steps:
                break
        losses.append(tot / len(perm))
        if max_steps and step >= max_steps:
            break
    train_secs = time.time() - t0
    if max_steps:                                                        # 벤치마크: 평가 없이 측정값만
        peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else float("nan")
        del net, opt
        empty_cache()
        return dict(steps=step, sec_per_step=train_secs / step, steps_per_fold=total, peak_gb=peak, loss_first10=float(np.mean(step_losses[:10])), loss_last10=float(np.mean(step_losses[-10:])))
    net.eval()
    with torch.no_grad():
        order = np.argsort([len(TOK[i]) for i in te_idx])
        out = np.zeros(len(te_idx), int)
        for s in range(0, len(order), 128):
            bi = order[s:s + 128]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                out[bi] = net(pad_batch([TOK[te_idx[i]] for i in bi]).to(DEVICE)).float().argmax(1).cpu().numpy()
    cm = np.zeros((len(cls), len(cls)))
    np.add.at(cm, (y_all[te_idx], out), 1)
    secs = time.time() - t0
    del net, opt
    empty_cache()
    return dict(donor=d, cm=cm.tolist(), bal=float(balanced_accuracy_score(y_all[te_idx], out)), losses=losses, seconds=secs)


def versions():
    return {k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "anndata", "torch", "transformers", "scipy"]}


# ---------- 단계: check (Mac) ----------
if STAGE == "check":
    E_cache = np.load(DATA / "geneformer_embeddings.npz")["v2_cls_L-2"]
    ec, _ = embed_frozen([wrap(t, V2["max_len"]) for t in RANKED[:100]], batch=8)
    print(f"점검: 새로 계산한 4,096토큰 동결 <cls> L-2 임베딩 100개 vs 노트북 04/01 캐시 최대 절대차 {np.abs(ec - E_cache[:100]).max():.2e}")
    raise SystemExit(0)

# ---------- 단계: bench ----------
if STAGE == "bench":
    n = int(os.environ.get("H2_BENCH_STEPS", "60"))
    gpu = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else str(DEVICE)
    for amp in ([False, True] if DEVICE.type == "cuda" else [False]):
        r = finetune_fold(dn[0], amp=amp, max_steps=n)
        fold_min = r["sec_per_step"] * r["steps_per_fold"] / 60
        print(f"[bench] {gpu} | {'fp16 autocast' if amp else 'fp32'} | {r['steps']}스텝 {r['sec_per_step']:.3f}s/스텝 → fold당 학습 약 {fold_min:.1f}분(평가 제외), 20 fold 약 {fold_min * 20 / 60:.1f}시간 | "
              f"최대 GPU 메모리 {r['peak_gb']:.1f} GB | 손실 처음 10스텝 {r['loss_first10']:.3f} → 마지막 10스텝 {r['loss_last10']:.3f}", flush=True)
    print("versions:", versions())
    raise SystemExit(0)

# ---------- 재개용 캐시: 설정·데이터·토큰 지문이 붙은 디렉터리 ----------
KEY = hashlib.sha256((json.dumps(CONFIG, sort_keys=True) + DATA_SHA + TOK_SHA).encode()).hexdigest()[:12]


def to_drive(cache):
    """fold가 끝날 때마다 결과 폴더를 Drive로 복사(임시 이름 → 교체). Drive가 없으면 아무것도 하지 않는다."""
    if not DRIVE:
        return
    dst = Path(DRIVE) / cache.name
    dst.mkdir(parents=True, exist_ok=True)
    for p in cache.iterdir():
        if p.is_file() and (not (dst / p.name).exists() or (dst / p.name).stat().st_size != p.stat().st_size or p.suffix in (".jsonl", ".log", ".json")):
            tmp = dst / (p.name + ".tmp")
            shutil.copyfile(p, tmp)
            os.replace(tmp, dst / p.name)


# ---------- 단계: train (Colab) ----------
if STAGE == "train":
    CACHE = OUT / f"cache_{KEY}"
    if DRIVE and not CACHE.exists() and (Path(DRIVE) / CACHE.name).exists():
        shutil.copytree(Path(DRIVE) / CACHE.name, CACHE)                  # VM이 새로 떴을 때 Drive에서 재개
        print("Drive에서 캐시 복원:", sorted(p.name for p in CACHE.iterdir()))
    CACHE.mkdir(exist_ok=True)
    (CACHE / "meta.json").write_text(json.dumps(dict(run=RUN_NAME, key=KEY, config=CONFIG, primary=PRIMARY, data_sha256=DATA_SHA, token_sha256=TOK_SHA, device=str(DEVICE),
                                                     gpu=torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None, packages=versions()), ensure_ascii=False, indent=1))
    plog = (CACHE / "progress.log").open("a")

    def say(msg):
        print(msg, flush=True)
        plog.write(msg + "\n")
        plog.flush()

    if not (CACHE / "frozen512.npz").exists():
        t = time.time()
        ec, em = embed_frozen(TOK)
        np.savez(CACHE / "frozen512.npz", cls=ec, mean=em)
        say(f"동결 임베딩(512토큰, L-2) 저장 {time.time() - t:.0f}s")
        to_drive(CACHE)
    LOG = CACHE / "finetune_results.jsonl"
    DONE = {}
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            r = json.loads(line)
            DONE[(r["kind"], r["donor"])] = r
    JOBS = [("ft", d) for d in dn] + [("rand", d) for d in dn[:CTRL_DONORS_RAND]] + [("shuf", d) for d in dn[:CTRL_DONORS_SHUF]]
    say(f"시작: 캐시 {CACHE.name} | 끝난 fold {len(DONE)}/{len(JOBS)}")
    for kind, d in JOBS:
        if (kind, d) in DONE:
            continue
        r = finetune_fold(d, random_init=(kind == "rand"), shuffle=(kind == "shuf"))
        r["kind"] = kind
        DONE[(kind, d)] = r
        with LOG.open("a") as f:
            f.write(json.dumps(r) + "\n")
        say(f"  [{kind:4s}] 기증자 {d}: 균형정확도 {r['bal']:.3f} | 훈련 손실 {', '.join(f'{x:.3f}' for x in r['losses'])} | {r['seconds']:.0f}s | 누적 {time.time() - T0:.0f}s")
        to_drive(CACHE)
    (CACHE / "DONE").write_text(f"{len(DONE)} folds, {time.time() - T0:.0f}s\n")
    say(f"완료: {len(DONE)} fold, 총 {time.time() - T0:.0f}초")
    to_drive(CACHE)
    raise SystemExit(0)

# ---------- 단계: report (Mac) ----------
import scanpy as sc  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

caches = sorted(p for p in OUT.glob("cache_*") if (p / "finetune_results.jsonl").exists())
assert len(caches) == 1, f"결과 캐시가 정확히 하나여야 한다: {caches}"
CACHE = caches[0]
meta = json.loads((CACHE / "meta.json").read_text())
assert meta["data_sha256"] == DATA_SHA and meta["token_sha256"] == TOK_SHA, "학습에 쓴 데이터·토큰이 이 Mac의 것과 다르다"
cfg_run = dict(meta["config"])
amp_run = cfg_run.pop("amp")
assert cfg_run == {k: v for k, v in CONFIG.items() if k != "amp"}, "학습 설정이 이 스크립트의 사전 지정과 다르다"
print(f"학습 실행: 캐시 {CACHE.name} | GPU {meta['gpu']} | fp16 autocast {amp_run} | torch {meta['packages']['torch']}, transformers {meta['packages']['transformers']}")
FZ = np.load(CACHE / "frozen512.npz")
GF = np.load(DATA / "geneformer_embeddings.npz")
DONE = {}
for line in (CACHE / "finetune_results.jsonl").read_text().splitlines():
    r = json.loads(line)
    DONE[(r["kind"], r["donor"])] = r
JOBS = [("ft", d) for d in dn] + [("rand", d) for d in dn[:CTRL_DONORS_RAND]] + [("shuf", d) for d in dn[:CTRL_DONORS_SHUF]]
assert set(DONE) == set(JOBS), f"fold가 모자라거나 남는다: {len(DONE)}/{len(JOBS)}"

a = sub.copy()
sc.pp.filter_genes(a, min_cells=3)
sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3")
hvg = a.var.highly_variable.values
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
b = a[:, hvg].copy()
sc.pp.scale(b, max_value=10)
sc.pp.pca(b, n_comps=50, random_state=SEED)
Z_PCA = b.obsm["X_pca"].copy()                                           # 주 PCA: 모든 기증자의 세포로 적합한다(transductive). 엄격한 fold별 적합은 민감도로 따로 낸다


def l2(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def split_static(Z, te):
    mu = Z[~te].mean(0)
    return l2(Z[~te] - mu), l2(Z[te] - mu)


def fold_pca_strict(te):
    """유전자 필터(min_cells=3)·HVG(seurat_v3)·스케일·PCA를 모두 훈련 fold의 세포만으로 적합한다."""
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


FT = "V2 미세조정(512토큰, <cls>)"
RUNS = {}
for name, Z in [("PCA(HVG,50)", Z_PCA), ("V2 동결 <cls> L-2 (512토큰)", FZ["cls"]), ("V2 동결 평균 L-2 (512토큰)", FZ["mean"]), ("V2 동결 <cls> L-2 (4,096토큰, 캐시)", GF["v2_cls_L-2"]),
                ("PCA(HVG,50) 엄격 fold별", fold_pca_strict)]:
    out = Parallel(n_jobs=8)(delayed(frozen_fold)(Z, d) for d in dn)
    RUNS[name] = dict(cms=np.stack([o[0] for o in out]), bal=np.array([o[1] for o in out]))


def pack(kind, donors_):
    return dict(cms=np.stack([np.array(DONE[(kind, d)]["cm"]) for d in donors_]), bal=np.array([DONE[(kind, d)]["bal"] for d in donors_]))


RUNS[FT] = pack("ft", dn)
BI = np.random.default_rng(SEED).integers(0, len(dn), size=(BOOT, len(dn)))      # 모든 비교에 같은 기증자 재표본


def compare(k1, k0):
    r1, r0 = RUNS[k1], RUNS[k0]
    f1d = np.array([macro_f1(r1["cms"][i].sum(0)) - macro_f1(r0["cms"][i].sum(0)) for i in BI])
    return macro_f1(r1["cms"].sum(0)) - macro_f1(r0["cms"].sum(0)), np.percentile(f1d, [2.5, 97.5]), r1["bal"] - r0["bal"]


print("\n=== 결과 (기증자 13명 LODO, 거친 라벨 13종; baseline 행은 로지스틱 head, 미세조정 행은 backbone과 함께 학습한 선형 head) ===")
for k, r in RUNS.items():
    print(f"  {k:34s} 풀링 macro-F1 {macro_f1(r['cms'].sum(0)):.3f} | 기증자별 균형정확도 평균 {r['bal'].mean():.3f} (최소 {r['bal'].min():.2f}, 최대 {r['bal'].max():.2f})")
print("\n=== 주 비교 ===")
for base in ["PCA(HVG,50)", "V2 동결 <cls> L-2 (512토큰)", "V2 동결 <cls> L-2 (4,096토큰, 캐시)", "V2 동결 평균 L-2 (512토큰)", "PCA(HVG,50) 엄격 fold별"]:
    pooled, ci, d_bal = compare(FT, base)
    dec, mean_d, wins = decide(d_bal, tuple(ci))
    tag = "PRIMARY" if base in PRIMARY["baselines"] else ("민감도" if "엄격" in base else "참고")
    print(f"  [{tag}] 미세조정 − {base}: 풀링 macro-F1 Δ {pooled:+.3f} 95%[{ci[0]:+.3f},{ci[1]:+.3f}] | 기증자별 균형정확도 평균 Δ {mean_d:+.3f}, Δ>0 {wins}/{len(d_bal)} → {dec}")
print("\n=== 대조 ===")
rand = pack("rand", dn[:CTRL_DONORS_RAND])
idx5 = list(range(CTRL_DONORS_RAND))
d_bal = RUNS[FT]["bal"][idx5] - rand["bal"]
print(f"양성 대조(사전학습 효과, 앞의 {CTRL_DONORS_RAND}개 기증자): 미세조정 풀링 macro-F1 {macro_f1(RUNS[FT]['cms'][idx5].sum(0)):.3f} vs 무작위 초기화 미세조정 {macro_f1(rand['cms'].sum(0)):.3f} | "
      f"기증자별 균형정확도 Δ 평균 {d_bal.mean():+.3f}, Δ>0 {int((d_bal > 0).sum())}/{CTRL_DONORS_RAND} (사전 규칙: 5개 모두 양수)")
sh = [DONE[("shuf", d)]["bal"] for d in dn[:CTRL_DONORS_SHUF]]
ch = [1 / len(set(y_all[donors == d])) for d in dn[:CTRL_DONORS_SHUF]]
print(f"음성 대조(훈련 라벨 섞기, {CTRL_DONORS_SHUF}개 fold): 기증자별 균형정확도 {', '.join(f'{x:.3f}' for x in sh)} (그 기증자에 있는 클래스 수 기준 균등 추측 {', '.join(f'{x:.3f}' for x in ch)})")
print("\n훈련 손실(에폭 1, 2)·fold 시간:", " ".join(f"{d}:{DONE[('ft', d)]['losses'][0]:.2f}/{DONE[('ft', d)]['losses'][-1]:.2f}/{DONE[('ft', d)]['seconds']:.0f}s" for d in dn))
print("기증자별 균형정확도 (PCA | V2 동결 512 | V2 미세조정):")
print("  " + " ".join(f"{d}:{RUNS['PCA(HVG,50)']['bal'][i]:.2f}|{RUNS['V2 동결 <cls> L-2 (512토큰)']['bal'][i]:.2f}|{RUNS[FT]['bal'][i]:.2f}" for i, d in enumerate(dn)))
print(f"\n총 {time.time() - T0:.0f}초 (보고 단계)")
