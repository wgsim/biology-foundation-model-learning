# 트랙 C 질문의 다른 절반 (C④): "scGPT(whole-human) zero-shot 임베딩 위의 지도 head는 PCA head·Geneformer 임베딩보다 기증자 단위로 나은가?"
# 데이터: 모듈 04의 Wilk 2020 PBMC 하위표본(캐시) 10,400세포, 기증자 13명, 거친 라벨 13종 — 트랙 C MVP와 같은 설정
# scGPT: 소스 26개 .py(GitHub bowang-lab/scGPT 커밋 cebd6fae…, MIT)를 `.cache/scgpt_src/`에 두고 *설치 없이* sys.path로 불러 쓴다. 체크포인트 whole-human(Google Drive, 약 207 MB)은 weights_only=True로만 읽는다.
# 임베딩: 공식 `tasks/cell_emb.py`의 입력 구성(<cls> + 0이 아닌 유전자, 값은 세포 안에서 51구간 binning, 1200 초과 시 무작위 표본, <cls> 위치의 마지막 층 출력)을 옮긴 자체 루프다.
#   공식 `embed_data`는 CUDA·os.sched_getaffinity(리눅스 전용)를 가정해 그대로 부를 수 없다. 공식과 다른 점: 공식은 반환 직전에 행별 L2 정규화를 하지만 여기서는 모든 특징에 공통인 파이프라인
#   (훈련 fold 평균 센터링 → L2)을 쓴다 — 공식 순서(L2 → 센터링)는 민감도 비교로 따로 낸다. 입력은 공식 함수처럼 adata.X(원 카운트)를 그대로 넣는다.
# 체크포인트 args.json: no_cls=true, no_cce=true, MVC=true로 기록(이 플래그가 실제 사전학습에서 어떻게 쓰였는지는 확인하지 못했다). cell_emb_style은 기록되어 있지 않아 `<cls>` 위치가 사전학습에서 세포 임베딩으로 쓰였는지는
#   확인하지 못했다 → `<cls>`와 유전자 위치 평균 풀링을 둘 다 낸다.
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_C_scgpt_zero_shot.py` (캐시만 사용; H2_SMOKE=1이면 128세포 시간 측정만)
import hashlib, importlib.metadata as md, json, os, re, subprocess, sys, time, warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path.cwd()
SC = ROOT / ".cache" / "singlecell"
SRC = ROOT / ".cache" / "scgpt_src"
CKPT = ROOT / ".cache" / "scgpt_ckpt" / "whole-human"
OUT = ROOT / ".cache" / "capstone_runs" / "h2_C_scgpt_zero_shot"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.mha.set_fastpath_enabled(False)   # eval 모드의 nested-tensor 빠른 경로(aten::_nested_tensor_from_mask_left_aligned)가 MPS에 없다 — 같은 수식의 일반 경로를 쓴다
SMOKE = os.environ.get("H2_SMOKE") == "1"

# ---------- 0. 사전 등록 (작성자는 같은 데이터에서 Geneformer V2 < PCA, V1 < V2의 결과를 이미 보았다; scGPT 결과는 보기 전에 정했다) ----------
RUN_NAME = "h2_C_scgpt_zero_shot"
MAX_LEN, BATCH = 1200, 32            # 공식 embed_data 기본 max_length 1200 (배치는 Mac 메모리에 맞게 32)
PRIMARY = dict(model="scGPT whole-human cls", baseline="PCA(HVG,50)", head="logreg", labels="거친 13종", unit="donor(13)")
MIN_EFFECT, MIN_WINS, BOOT = 0.02, 9, 1000
C_LR, K_NN, SEED = 1.0, 15, 0
COMMIT = "cebd6fae655b9c585a4807daa3ac31bb764f06b4"
CKPT_SHA = {"best_model.pt": "6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922", "vocab.json": "acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f",
            "args.json": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"}


def decide(deltas, ci):
    deltas = np.asarray(deltas, float)
    mean_d, wins = float(np.mean(deltas)), int(np.sum(deltas > 0))
    assert len(deltas) == 13, "MIN_WINS=9는 기증자 13명 기준이다"
    ok = mean_d >= MIN_EFFECT and wins >= MIN_WINS and (ci is None or ci[0] > 0)
    return ("지지" if ok else "지지되지 않음"), mean_d, wins


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# ---------- 1. 출처 ----------
for f, d in CKPT_SHA.items():
    assert sha256(CKPT / f) == d, f"{f}의 sha256이 내려받을 때 기록한 값과 다르다"
src_files = sorted((SRC / "scgpt").rglob("*.py"))
assert len(src_files) == 26 and sum(p.stat().st_size for p in src_files) == 273078, "scGPT 소스가 승인받은 26개 파일(273,078 B)이 아니다"
src_digest = hashlib.sha256("".join(f"{p.relative_to(SRC)}:{sha256(p)}" for p in src_files).encode()).hexdigest()
DATA_SHA = {f: sha256(SC / f) for f in ["pbmc_sub.h5ad", "geneformer_embeddings.npz"]}
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(), device=str(DEVICE),
            scgpt=dict(source="https://github.com/bowang-lab/scGPT", commit=COMMIT, files=26, bytes=273078, source_digest=src_digest[:16] + "…", license="MIT (LICENSE 파일 확인)"),
            checkpoint=dict(name="scGPT whole-human", source="Google Drive 폴더 1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y (scGPT README)", sha256={k: v[:16] + "…" for k, v in CKPT_SHA.items()},
                            license="README에 별도 표기 없음 — 확인하지 못함, 재배포하지 않는다"),
            data={f: v[:16] + "…" for f, v in DATA_SHA.items()}, max_len=MAX_LEN, batch=BATCH, primary=PRIMARY,
            packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "scanpy", "anndata", "torch", "scipy"]})
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))
print("device:", DEVICE, "| smoke:", SMOKE)

# ---------- 2. 데이터 ----------
sub = ad.read_h5ad(SC / "pbmc_sub.h5ad")
obs = sub.obs
donors = obs.donor.values
dn = sorted(set(donors))
cls = sorted(set(obs.label))
y_all = np.array([cls.index(v) for v in obs.label])
assert len(dn) == 13 and all((donors == d).sum() == 800 for d in dn) and len(cls) == 13
counts = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
symbols = sub.var["gene_symbol"].astype(str).values
print(f"세포 {sub.n_obs} | 기증자 {len(dn)} | 라벨 {len(cls)}종 | 유전자 {sub.n_vars}")

# ---------- 3. scGPT 불러오기 (설치 없이 소스 경로로) ----------
sys.path.insert(0, str(SRC))
import scgpt  # noqa: E402
from scgpt.data_collator import DataCollator  # noqa: E402
from scgpt.model import TransformerModel  # noqa: E402
from scgpt.tokenizer import GeneVocab  # noqa: E402
from scgpt.utils import load_pretrained  # noqa: E402

assert Path(scgpt.__file__).resolve().is_relative_to(SRC.resolve()), "scgpt가 .cache/scgpt_src가 아닌 곳에서 불려왔다"
vocab = GeneVocab.from_file(CKPT / "vocab.json")
for s in ["<pad>", "<cls>", "<eoc>"]:
    if s not in vocab:
        vocab.append_token(s)
cfg = json.loads((CKPT / "args.json").read_text())
in_vocab = np.array([vocab[g] if g in vocab else -1 for g in symbols])
use_cols = np.where(in_vocab >= 0)[0]
print(f"어휘({len(vocab)})와 연결된 유전자 {len(use_cols)}/{len(symbols)} ({len(use_cols) / len(symbols):.2f}); 연결된 유전자가 가진 카운트 비율 {np.asarray(counts[:, use_cols].sum()).item() / np.asarray(counts.sum()).item():.3f}")
gene_ids = in_vocab[use_cols].astype(int)
assert len(set(gene_ids.tolist())) == len(gene_ids), "서로 다른 열이 같은 어휘 id에 연결되었다(중복 유전자 기호)"
Xv = counts[:, use_cols].tocsr()
Xv.eliminate_zeros()
vocab.set_default_index(vocab["<pad>"])
PAD_ID, CLS_ID = vocab[cfg["pad_token"]], vocab["<cls>"]


def build_model(pretrained):
    torch.manual_seed(SEED)
    m = TransformerModel(ntoken=len(vocab), d_model=cfg["embsize"], nhead=cfg["nheads"], d_hid=cfg["d_hid"], nlayers=cfg["nlayers"], nlayers_cls=cfg["n_layers_cls"], n_cls=1,
                         vocab=vocab, dropout=cfg["dropout"], pad_token=cfg["pad_token"], pad_value=cfg["pad_value"], do_mvc=True, do_dab=False, use_batch_labels=False,
                         domain_spec_batchnorm=False, explicit_zero_prob=False, use_fast_transformer=False, fast_transformer_backend="flash", pre_norm=False)
    if pretrained:
        state = torch.load(CKPT / "best_model.pt", map_location="cpu", weights_only=True)       # 텐서 state_dict만 허용 (pickle 코드 실행 차단)
        rules = {r"self_attn\._impl\.Wqkv\.": "self_attn.in_proj_", r"self_attn\.Wqkv\.": "self_attn.in_proj_", r"self_attn\._impl\.out_proj\.": "self_attn.out_proj."}      # load_pretrained의 use_fast_transformer=False 규칙(개수 점검용)
        ren = {}
        for k, v in state.items():
            kk = k
            for pat, rep in rules.items():
                kk = re.sub(pat, rep, kk)
            ren[kk] = v
        md_ = m.state_dict()
        matched = [k for k in md_ if k in ren and ren[k].shape == md_[k].shape]
        missing = [k for k in md_ if k not in matched]
        unused = [k for k in ren if k not in md_ or ren[k].shape != md_[k].shape]
        print(f"체크포인트 텐서 {len(state)}개 | 모델 파라미터 {len(md_)}개 중 이름·모양이 맞는 것 {len(matched)}개")
        print("  체크포인트에 없는(=무작위 초기값으로 남는) 모델 파라미터:", missing)
        print("  모델에 대응이 없는 체크포인트 텐서:", unused)
        # 임베딩 경로(encoder·value_encoder·transformer_encoder 등)는 전부 체크포인트에서 와야 한다. 허용하는 누락은 분류 head(cls_decoder; 임베딩에 쓰지 않음)뿐이다.
        assert len(matched) > 0 and all(k.startswith("cls_decoder.") for k in missing), f"임베딩 경로 파라미터가 체크포인트에서 오지 않았다: {missing}"
        assert set(unused) <= {"flag_encoder.weight"}, f"예상 밖의 미사용 체크포인트 텐서: {unused}"
        load_pretrained(m, state, strict=False, verbose=False)
        ref = m.state_dict()
        assert all(torch.equal(ref[k].cpu(), ren[k]) for k in matched), "불러온 값이 체크포인트와 다르다"
    return m.to(DEVICE).eval()


@torch.no_grad()
def embed_scgpt(model, n_cells=None, tag=""):
    """공식 embed_data와 같은 입력: <cls> + 발현이 0이 아닌 유전자, 값은 세포 안에서 51구간으로 binning, 유전자 수가 max_length를 넘으면 무작위 표본(시드 고정). 반환: cls 위치 출력, 유전자 위치 평균."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n = Xv.shape[0] if n_cells is None else n_cells
    coll = DataCollator(do_padding=True, pad_token_id=PAD_ID, pad_value=cfg["pad_value"], do_mlm=False, do_binning=True, max_length=MAX_LEN, sampling=True, keep_first_n_tokens=1)
    cls_out = np.zeros((n, cfg["embsize"]), np.float32)
    mean_out = np.zeros((n, cfg["embsize"]), np.float32)
    t0 = time.time()
    n_samp = 0
    for b in range(0, n, BATCH):
        ex = []
        for i in range(b, min(b + BATCH, n)):
            row = Xv[i]
            genes = np.insert(gene_ids[row.indices], 0, CLS_ID)
            values = np.insert(row.data.astype(np.float32), 0, cfg["pad_value"])
            n_samp += int(len(genes) > MAX_LEN)
            ex.append({"id": i, "genes": torch.from_numpy(genes).long(), "expressions": torch.from_numpy(values).float()})
        d = coll(ex)
        g, e = d["gene"].to(DEVICE), d["expr"].to(DEVICE)
        pad = g.eq(PAD_ID)
        h = model._encode(g, e, src_key_padding_mask=pad).float()
        cls_out[b:b + len(ex)] = h[:, 0].cpu().numpy()
        gm = (~pad).clone()
        gm[:, 0] = False
        mean_out[b:b + len(ex)] = ((h * gm.unsqueeze(-1)).sum(1) / gm.sum(1, keepdim=True).clamp(min=1)).cpu().numpy()
        if (b // BATCH) % 50 == 0:
            print(f"  [{tag}] {b + len(ex)}/{n} 세포 | {time.time() - t0:.0f}s", flush=True)
    print(f"  [{tag}] 완료 {time.time() - t0:.0f}s | {MAX_LEN} 초과로 무작위 표본을 뽑은 세포 {n_samp}/{n}")
    return cls_out, mean_out


if SMOKE:
    m = build_model(True)
    t = time.time()
    c, mn = embed_scgpt(m, n_cells=128, tag="smoke")
    print(f"smoke: 128세포 {time.time() - t:.0f}s → 10,400세포 환산 약 {(time.time() - t) / 128 * 10400 / 60:.0f}분 | NaN {bool(np.isnan(c).any())} | cls 노름 중앙값 {np.median(np.linalg.norm(c, axis=1)):.2f}")
    raise SystemExit(0)

EMB_FILE = OUT / "emb.npz"
KEY = hashlib.sha256((src_digest + "".join(CKPT_SHA.values()) + DATA_SHA["pbmc_sub.h5ad"] + f"|{MAX_LEN}|{BATCH}|{SEED}|{torch.__version__}|{DEVICE}").encode()).hexdigest()[:24]
emb = None
if EMB_FILE.exists():
    z = np.load(EMB_FILE)
    if "key" in z.files and str(z["key"]) == KEY:
        emb = {k: z[k] for k in z.files if k != "key"}
        print("캐시된 임베딩 사용:", sorted(emb))
if emb is None:
    emb = {}
    m = build_model(True)
    emb["scgpt_cls"], emb["scgpt_mean"] = embed_scgpt(m, tag="whole-human")
    del m
    torch.mps.empty_cache()
    mr = build_model(False)
    emb["scgpt_rand_cls"], emb["scgpt_rand_mean"] = embed_scgpt(mr, tag="random-init")
    del mr
    torch.mps.empty_cache()
    np.savez(EMB_FILE, key=np.array(KEY), **emb)
assert all(np.isfinite(v).all() for v in emb.values()), "임베딩에 NaN/inf가 있다"
GF = np.load(SC / "geneformer_embeddings.npz")

# ---------- 4. PCA baseline와 평가 (트랙 C와 같은 절차; 주 PCA는 모든 기증자의 세포로 비지도 적합 = transductive — 엄격한 fold별 적합은 민감도로 따로 낸다) ----------
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
    """유전자 필터(min_cells=3)·HVG(seurat_v3)·스케일·PCA를 모두 훈련 fold의 세포만으로 적합하고 test 기증자에 적용한다. 세포별 정규화(총합 1e4)와 log1p는 세포 단위라 누수가 없다."""
    tr = ~te
    a_tr = ad.AnnData(X=counts[tr].astype(np.float32), var=pd.DataFrame(index=np.arange(counts.shape[1]).astype(str)))
    sc.pp.filter_genes(a_tr, min_cells=3)
    kept = a_tr.var_names.astype(int).values
    sc.pp.highly_variable_genes(a_tr, n_top_genes=2000, flavor="seurat_v3")
    cols = kept[a_tr.var.highly_variable.values]
    tot = np.asarray(counts[:, kept].sum(1)).ravel()                       # 필터를 통과한 유전자의 합 — 주 PCA(filter → normalize_total)와 같은 정의
    L = np.log1p(counts[:, cols].toarray() / tot[:, None] * 1e4).astype(np.float32)
    mu, sd = L[tr].mean(0), L[tr].std(0) + 1e-8
    f = lambda M: np.clip((M - mu) / sd, -10, 10)
    p = PCA(n_components=50, random_state=SEED, svd_solver="randomized").fit(f(L[tr]))
    Ztr, Zte = p.transform(f(L[tr])), p.transform(f(L[te]))
    mu_ = Ztr.mean(0)
    return l2(Ztr - mu_), l2(Zte - mu_)


def static(M):
    M = np.asarray(M, np.float64)
    return lambda te: split_static(M, te)


FEATS = {"PCA(HVG,50)": static(Z_PCA), "scGPT cls": static(emb["scgpt_cls"]), "scGPT 평균": static(emb["scgpt_mean"]), "scGPT random-init cls": static(emb["scgpt_rand_cls"]),
         "Geneformer V2 cls L-2": static(GF["v2_cls_L-2"]), "Geneformer V1 평균 L-2": static(GF["v1_mean_L-2"]),
         "scGPT cls (L2 먼저)": static(l2(emb["scgpt_cls"].astype(np.float64))),          # 공식 embed_data의 순서: 행별 L2 → (훈련 fold 센터링 → L2)
         "PCA(HVG,50) 엄격 fold별": fold_pca_strict}


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
        ytr = np.random.default_rng(SEED).permutation(ytr)
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
    out = Parallel(n_jobs=8)(delayed(one_fold)(feat, head, d, shuffle) for d in dn)
    RUNS[(feat, head, shuffle)] = dict(cms=np.stack([o[0] for o in out]), bal=np.array([o[1] for o in out]))
    r = RUNS[(feat, head, shuffle)]
    print(f"  {feat:22s} {head:6s} shuffle={shuffle!s:5s} 풀링 macro-F1 {macro_f1(r['cms'].sum(0)):.3f} | 기증자별 균형정확도 평균 {r['bal'].mean():.3f}", flush=True)


print("\n[로지스틱 head]")
for f in ["PCA(HVG,50)", "scGPT cls", "scGPT 평균", "scGPT random-init cls", "Geneformer V2 cls L-2", "Geneformer V1 평균 L-2", "scGPT cls (L2 먼저)", "PCA(HVG,50) 엄격 fold별"]:
    run(f, "logreg")
run("scGPT cls", "logreg", shuffle=True)
print("\n[kNN(k=15) head — zero-shot 방식]")
for f in ["PCA(HVG,50)", "scGPT cls", "Geneformer V2 cls L-2"]:
    run(f, "knn")
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
print("PRIMARY scGPT cls vs PCA(HVG,50) [logreg]:", compare(P("scGPT cls"), P("PCA(HVG,50)")))
print("scGPT cls vs Geneformer V2 cls L-2 [logreg]:", compare(P("scGPT cls"), P("Geneformer V2 cls L-2")))
print("\n=== 대조 ===")
print("양성 대조(사전학습 효과) scGPT cls vs scGPT random-init cls:", compare(P("scGPT cls"), P("scGPT random-init cls")))
cmS = RUNS[P("scGPT cls", s=True)]
print(f"음성 대조(훈련 라벨 섞기): 풀링 macro-F1 {macro_f1(cmS['cms'].sum(0)):.3f}, 기증자별 균형정확도 평균 {cmS['bal'].mean():.3f} (13종 균등 추측 ≈ {1 / 13:.3f})")
print("\n=== 민감도 ===")
print("scGPT 평균 풀링 vs PCA:", compare(P("scGPT 평균"), P("PCA(HVG,50)")))
print("scGPT cls, 공식 순서(L2 → 센터링 → L2) vs PCA:", compare(P("scGPT cls (L2 먼저)"), P("PCA(HVG,50)")))
print("PCA를 엄격하게 fold별 적합(필터·HVG·스케일·PCA 모두 훈련 fold만): scGPT cls vs PCA 엄격:", compare(P("scGPT cls"), P("PCA(HVG,50) 엄격 fold별")))
print("  같은 PCA 엄격 기준: Geneformer V2 cls vs PCA 엄격:", compare(P("Geneformer V2 cls L-2"), P("PCA(HVG,50) 엄격 fold별")))
print("  PCA 엄격 vs PCA(transductive) — 엄격 적합이 PCA를 얼마나 낮추는가:", compare(P("PCA(HVG,50) 엄격 fold별"), P("PCA(HVG,50)")))
print("head = kNN(k=15) scGPT cls vs PCA:", compare(P("scGPT cls", "knn"), P("PCA(HVG,50)", "knn")))
print("head = kNN(k=15) scGPT cls vs Geneformer V2 cls:", compare(P("scGPT cls", "knn"), P("Geneformer V2 cls L-2", "knn")))
print("\n기증자별 균형정확도 (PCA | scGPT cls | Geneformer V2 cls):")
print("  " + " ".join(f"{d}:{RUNS[P('PCA(HVG,50)')]['bal'][i]:.2f}|{RUNS[P('scGPT cls')]['bal'][i]:.2f}|{RUNS[P('Geneformer V2 cls L-2')]['bal'][i]:.2f}" for i, d in enumerate(dn)))
print(f"\n총 {time.time() - T0:.0f}초")
