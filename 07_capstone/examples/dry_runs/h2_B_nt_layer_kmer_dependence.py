# 트랙 B 질문의 다른 절반: "NT probe의 이득이 층 선택·6-mer baseline 크기(k·학습 크기)에 얼마나 의존하는가?" — 템플릿 구조로
# 모듈 03/01의 프로모터 분류(promoter_tata, promoter_no_tata)에서 NT-500M 평균 임베딩(층 0~24, 2층 간격) + 로지스틱 probe vs k-mer 빈도 baseline.
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_B_nt_layer_kmer_dependence.py` (캐시만 사용, 다운로드 없음; 임베딩은 .cache/capstone_runs/ 아래에 저장)
import hashlib, importlib.metadata as md, json, subprocess, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoTokenizer, EsmModel
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
T0 = time.time()
ROOT = Path.cwd()

# ---------- 0. 사전 등록 (작성자는 모듈 03/01에서 층 6/12/18/24와 k=6 baseline의 결과를 이미 보았다) ----------
RUN_NAME = "h2_B_nt_layer_kmer_dependence"
NT = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
NT_TASKS = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
TASKS = ["promoter_tata", "promoter_no_tata"]
LAYERS = list(range(0, 25, 2))                 # hidden_states 인덱스 (0 = 임베딩 출력)
RAND_LAYERS = [6, 12, 18, 24]                   # 무작위 초기화 대조에 쓰는 층
KS = [3, 4, 5, 6, 7]                            # baseline k-mer 길이 (차원 4^k)
NS = [500, 1000, 2000, 4000, 6000]              # probe 학습 서열 수
N_TRAIN, SEED = 6000, 0                         # 노트북 03/01과 같은 하위표본
SWEEP_LAYERS = [6, 12, 18, 24]                  # 학습 크기 스윕에서 층 후보
CS = [1e-4, 1e-3, 1e-2, 1e-1]
N_PERM_CTRL = 10                                # 음성 대조의 라벨 순열 횟수
MIN_EFFECT, BOOT = 0.01, 1000                   # Δ AUROC ≥ 0.01 이고 test 재표본 95% 구간 하한 > 0 (두 과제 모두) → '지지'
PRIMARY = dict(nt="train-CV로 고른 층(짝수 층 0~24)", baseline="6-mer 빈도", n_train=N_TRAIN, metric="test AUROC")


def decide(delta, ci):
    ok = delta >= MIN_EFFECT and ci[0] > 0
    return "지지" if ok else "지지되지 않음"


# ---------- 1. 출처 ----------
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


root = Path(snapshot_download(NT_TASKS, repo_type="dataset", allow_patterns=["promoter_tata/*.parquet", "promoter_no_tata/*.parquet"]))
nt_dir = Path(snapshot_download(NT, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"]))
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            dataset_snapshot=root.name, model_snapshot=nt_dir.name, device=str(DEVICE),
            sha256={f"{t}/{sp}": sha256(root / t / f"{sp}.parquet")[:16] + "…" for t in TASKS for sp in ["train", "test"]},
            packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "torch", "transformers", "scipy"]},
            layers=LAYERS, ks=KS, ns=NS, primary=PRIMARY)
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))
print("device:", DEVICE, "| 데이터 스냅샷", root.name[:10], "| 모델 스냅샷", nt_dir.name[:10])

# ---------- 2. 데이터 점검 (노트북 03/01과 같은 하위표본) ----------
data = {}
for t in TASKS:
    d = {sp: pd.read_parquet(root / t / f"{sp}.parquet") for sp in ["train", "test"]}
    for sp in d:
        d[sp]["chrom"] = d[sp].name.str.split(":").str[0]
    tr = d["train"].sample(n=min(N_TRAIN, len(d["train"])), random_state=SEED).reset_index(drop=True)
    te = d["test"].reset_index(drop=True)
    assert not (set(tr.chrom) & set(te.chrom)), "train·test 염색체가 겹친다"
    assert te.chrom.nunique() == 2 and not (set(tr.name) & set(te.name))
    data[t] = dict(tr=tr, te=te)
    print(f"{t:17s} train {len(tr)} test {len(te)} | 양성 비율 train {tr.label.mean():.2f} test {te.label.mean():.2f} | train 염색체 {tr.chrom.nunique()}개, test {sorted(te.chrom.unique())}")

# ---------- 4. 특징 ----------
tok = AutoTokenizer.from_pretrained(nt_dir)
SPECIAL = torch.tensor(tok.all_special_ids)


@torch.no_grad()
def embed_all(model, seqs, layers=LAYERS, batch=64):
    out = {l: [] for l in layers}
    sp = SPECIAL.to(DEVICE)
    for s in range(0, len(seqs), batch):
        enc = tok(list(seqs[s:s + batch]), return_tensors="pt", padding=True).to(DEVICE)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = (enc["attention_mask"].bool() & ~torch.isin(enc["input_ids"], sp)).float()
        for l in layers:
            out[l].append(((hs[l] * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)).float().cpu())
    return {l: torch.cat(v).numpy() for l, v in out.items()}


def cache_key(kind):
    """캐시된 임베딩이 지금의 데이터·모델·하위표본과 같은 입력에서 나왔는지 확인하는 키 (파일 이름만 믿지 않는다)."""
    h = hashlib.sha256("|".join([root.name, nt_dir.name, kind, str(LAYERS), str(RAND_LAYERS), str(N_TRAIN), str(SEED)]).encode())
    for t in TASKS:
        for sp in ["tr", "te"]:
            h.update("\n".join(data[t][sp].sequence.tolist()).encode())
    return h.hexdigest()[:24]


def get_embeddings(kind):
    f = OUT / f"emb_{kind}.npz"
    if f.exists():
        z = np.load(f)
        if "key" in z.files and str(z["key"]) == cache_key(kind):
            return {t: {sp: {l: z[f"{t}|{sp}|{l}"] for l in (LAYERS if kind == "trained" else RAND_LAYERS)} for sp in ["tr", "te"]} for t in TASKS}
        print(f"[{kind}] 캐시의 키가 지금 입력과 다르다 → 다시 계산한다")
    t0 = time.time()
    if kind == "trained":
        model = EsmModel.from_pretrained(nt_dir, add_pooling_layer=False).to(DEVICE).eval()
        layers = LAYERS
    else:
        torch.manual_seed(SEED)
        model = EsmModel(AutoConfig.from_pretrained(nt_dir), add_pooling_layer=False).to(DEVICE).eval()
        layers = RAND_LAYERS
    s = data["promoter_tata"]["tr"].sequence.iloc[0]
    diff = float(np.abs(embed_all(model, [s], layers[-1:])[layers[-1]][0] - embed_all(model, [s, s[:250]], layers[-1:])[layers[-1]][0]).max())
    print(f"[{kind}] 패딩 유무 최대 차이 {diff:.2e}")
    res, flat = {}, {}
    for t in TASKS:
        res[t] = {}
        for sp in ["tr", "te"]:
            E = embed_all(model, data[t][sp].sequence.tolist(), layers)
            res[t][sp] = E
            for l in layers:
                flat[f"{t}|{sp}|{l}"] = E[l]
        print(f"[{kind}] {t} 임베딩 완료 ({time.time() - t0:.0f}s)", flush=True)
    np.savez(f, key=np.array(cache_key(kind)), **flat)
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return res


EMB = get_embeddings("trained")
EMB_R = get_embeddings("random")
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


def kmer_freq(seqs, k):
    rows, cols = [], []
    for i, s in enumerate(seqs):
        a = np.array([BASE.get(c, -1) for c in s])
        idx = sum(a[j:len(a) - k + 1 + j] * 4 ** (k - 1 - j) for j in range(k))
        ok = np.all([a[j:len(a) - k + 1 + j] >= 0 for j in range(k)], axis=0)
        cols.append(idx[ok])
        rows.append(np.full(ok.sum(), i))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(seqs), 4 ** k))
    return np.asarray((sparse.diags(1.0 / np.maximum(M.sum(1).A1, 1)) @ M).todense(), dtype=np.float32)


KM = {t: {k: (kmer_freq(data[t]["tr"].sequence, k), kmer_freq(data[t]["te"].sequence, k)) for k in KS} for t in TASKS}
print(f"특징 준비 {time.time() - T0:.0f}s")


# ---------- 5. probe (노트북 03/01의 fit_eval과 같은 절차: 표준화 + 로지스틱, C는 train 안의 염색체 단위 3-fold CV) ----------
def fit_eval(Xtr, ytr, gtr, Xte, yte):
    splits = list(GroupKFold(n_splits=3).split(Xtr, ytr, gtr))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    gs = GridSearchCV(pipe, {"logisticregression__C": CS}, cv=splits, scoring="roc_auc", n_jobs=8)
    gs.fit(Xtr, ytr)
    p = gs.predict_proba(Xte)[:, 1]
    return dict(auroc=roc_auc_score(yte, p), cv=gs.best_score_, C=gs.best_params_["logisticregression__C"], p=p)


def paired_delta(yte, pa, pb, n_boot=BOOT, seed=0):
    rs, ds = np.random.RandomState(seed), []
    for _ in range(n_boot):
        i = rs.randint(0, len(yte), len(yte))
        if yte[i].min() != yte[i].max():
            ds.append(roc_auc_score(yte[i], pa[i]) - roc_auc_score(yte[i], pb[i]))
    return roc_auc_score(yte, pa) - roc_auc_score(yte, pb), *np.percentile(ds, [2.5, 97.5])


NT_FIT, KM_FIT = {}, {}      # (task, layer) / (task, k, n) -> fit
t = time.time()
for task in TASKS:
    tr, te = data[task]["tr"], data[task]["te"]
    ytr, yte, gtr = tr.label.values, te.label.values, tr.chrom.values
    for l in LAYERS:
        NT_FIT[(task, l, N_TRAIN)] = fit_eval(EMB[task]["tr"][l], ytr, gtr, EMB[task]["te"][l], yte)
    for k in KS:
        KM_FIT[(task, k, N_TRAIN)] = fit_eval(KM[task][k][0], ytr, gtr, KM[task][k][1], yte)
    print(f"  {task} 층·k 스윕 완료 ({time.time() - t:.0f}s)", flush=True)

print("\n[층별 probe test AUROC (train CV AUROC)] 와 k-mer baseline")
tab = pd.DataFrame({task: {f"NT 층 {l}": f"{NT_FIT[(task, l, N_TRAIN)]['auroc']:.3f} ({NT_FIT[(task, l, N_TRAIN)]['cv']:.3f})" for l in LAYERS}
                    | {f"{k}-mer (차원 {4 ** k})": f"{KM_FIT[(task, k, N_TRAIN)]['auroc']:.3f} ({KM_FIT[(task, k, N_TRAIN)]['cv']:.3f})" for k in KS} for task in TASKS})
print(tab.to_string())

# ---------- 5a. 주 비교: train CV로 고른 층 vs 6-mer ----------
print("\n=== 주 비교: train-CV로 고른 층 − 6-mer (ΔAUROC, test 재표본 95% 구간) ===")
PRIM = {}
for task in TASKS:
    yte = data[task]["te"].label.values
    best = max(LAYERS, key=lambda l: NT_FIT[(task, l, N_TRAIN)]["cv"])
    d, lo, hi = paired_delta(yte, NT_FIT[(task, best, N_TRAIN)]["p"], KM_FIT[(task, 6, N_TRAIN)]["p"])
    PRIM[task] = (best, d, lo, hi, decide(d, (lo, hi)))
    print(f"  {task:17s} 층 {best}: Δ {d:+.3f} [{lo:+.3f},{hi:+.3f}] → {PRIM[task][4]}  (NT {NT_FIT[(task, best, N_TRAIN)]['auroc']:.3f} vs 6-mer {KM_FIT[(task, 6, N_TRAIN)]['auroc']:.3f}, test n={len(yte)})")
overall = "지지" if all(v[4] == "지지" for v in PRIM.values()) else "지지되지 않음"
print(f"  전체(두 과제 모두 '지지'여야 함) → {overall}")

# ---------- 6. 민감도 ----------
print("\n=== 민감도 ① 층: 모든 짝수 층을 고정했을 때 Δ vs 6-mer (구간이 0을 제외하면 *) ===")
SW = {}
for task in TASKS:
    yte = data[task]["te"].label.values
    for l in LAYERS:
        d, lo, hi = paired_delta(yte, NT_FIT[(task, l, N_TRAIN)]["p"], KM_FIT[(task, 6, N_TRAIN)]["p"])
        SW[(task, l)] = (d, lo, hi)
for l in LAYERS:
    print(f"  층 {l:2d} | " + " | ".join(f"{task} {SW[(task, l)][0]:+.3f} [{SW[(task, l)][1]:+.3f},{SW[(task, l)][2]:+.3f}]" + ("*" if SW[(task, l)][1] > 0 or SW[(task, l)][2] < 0 else " ") for task in TASKS))
for task in TASKS:
    ds = np.array([SW[(task, l)][0] for l in LAYERS])
    oracle = LAYERS[int(np.argmax([NT_FIT[(task, l, N_TRAIN)]["auroc"] for l in LAYERS]))]
    best = PRIM[task][0]
    print(f"  {task}: 층별 Δ 범위 {ds.min():+.3f} ~ {ds.max():+.3f} | CV 선택 층 {best}(Δ {SW[(task, best)][0]:+.3f}) · 마지막 층 24(Δ {SW[(task, 24)][0]:+.3f}) · test 오라클 층 {oracle}(Δ {SW[(task, oracle)][0]:+.3f}; 진단용, 선택에 쓰면 안 됨)")

print("\n=== 민감도 ② baseline k: CV로 고른 층 − k-mer ===")
for task in TASKS:
    yte = data[task]["te"].label.values
    best = PRIM[task][0]
    for k in KS:
        d, lo, hi = paired_delta(yte, NT_FIT[(task, best, N_TRAIN)]["p"], KM_FIT[(task, k, N_TRAIN)]["p"])
        print(f"  {task:17s} k={k} (차원 {4 ** k:5d}) AUROC {KM_FIT[(task, k, N_TRAIN)]['auroc']:.3f} | Δ {d:+.3f} [{lo:+.3f},{hi:+.3f}] → {decide(d, (lo, hi))}")

print("\n=== 민감도 ③ 학습 크기 n: 층은 {6,12,18,24} 중 train CV로 선택, baseline = 6-mer (같은 n) ===")
for task in TASKS:
    tr, te = data[task]["tr"], data[task]["te"]
    perm = np.random.RandomState(SEED).permutation(len(tr))
    yte = te.label.values
    for n in NS:
        ix = np.sort(perm[:n])
        ytr, gtr = tr.label.values[ix], tr.chrom.values[ix]
        if n == N_TRAIN:
            fits = {l: NT_FIT[(task, l, N_TRAIN)] for l in SWEEP_LAYERS}
            fb = KM_FIT[(task, 6, N_TRAIN)]
        else:
            fits = {l: fit_eval(EMB[task]["tr"][l][ix], ytr, gtr, EMB[task]["te"][l], yte) for l in SWEEP_LAYERS}
            fb = fit_eval(KM[task][6][0][ix], ytr, gtr, KM[task][6][1], yte)
        best = max(SWEEP_LAYERS, key=lambda l: fits[l]["cv"])
        d, lo, hi = paired_delta(yte, fits[best]["p"], fb["p"])
        print(f"  {task:17s} n={min(n, len(tr)):5d} 층 {best}: NT {fits[best]['auroc']:.3f} vs 6-mer {fb['auroc']:.3f} | Δ {d:+.3f} [{lo:+.3f},{hi:+.3f}] → {decide(d, (lo, hi))}", flush=True)

# ---------- 7. 대조 ----------
print("\n=== 대조 ===")
for task in TASKS:
    tr, te = data[task]["tr"], data[task]["te"]
    ytr, yte, gtr = tr.label.values, te.label.values, tr.chrom.values
    fr = {l: fit_eval(EMB_R[task]["tr"][l], ytr, gtr, EMB_R[task]["te"][l], yte) for l in RAND_LAYERS}
    best = max(RAND_LAYERS, key=lambda l: fr[l]["cv"])
    bt = max(RAND_LAYERS, key=lambda l: NT_FIT[(task, l, N_TRAIN)]["cv"])
    d, lo, hi = paired_delta(yte, NT_FIT[(task, bt, N_TRAIN)]["p"], fr[best]["p"])
    print(f"  {task:17s} 사전학습 효과(층 {bt} vs 무작위 초기화 층 {best}): NT {NT_FIT[(task, bt, N_TRAIN)]['auroc']:.3f} vs 무작위 {fr[best]['auroc']:.3f} | Δ {d:+.3f} [{lo:+.3f},{hi:+.3f}] → {decide(d, (lo, hi))} (양성 대조: 결정 규칙이 '지지'를 낼 수 있어야 함) | 무작위 초기화 vs 6-mer: {fr[best]['auroc'] - KM_FIT[(task, 6, N_TRAIN)]['auroc']:+.3f}")
    # 음성 대조: 훈련 라벨을 섞어 학습한 probe의 test AUROC. 섞은 라벨의 probe는 임의 방향이라 한 번의 AUROC는 0.5에서 꽤 멀어질 수 있으므로 순열 10회의 분포로 본다
    nt_a, km_a = [], []
    for ps in range(N_PERM_CTRL):
        ys = np.random.RandomState(ps).permutation(ytr)
        nt_a.append(fit_eval(EMB[task]["tr"][18], ys, gtr, EMB[task]["te"][18], yte)["auroc"])
        km_a.append(fit_eval(KM[task][6][0], ys, gtr, KM[task][6][1], yte)["auroc"])
    print(f"  {task:17s} 음성 대조(훈련 라벨 섞기 {N_PERM_CTRL}회): NT 층 18 AUROC 평균 {np.mean(nt_a):.3f} (sd {np.std(nt_a):.3f}, 범위 {np.min(nt_a):.3f}~{np.max(nt_a):.3f}), 6-mer 평균 {np.mean(km_a):.3f} (sd {np.std(km_a):.3f}, 범위 {np.min(km_a):.3f}~{np.max(km_a):.3f}) — 평균이 0.5 근처여야 하고, 한 번의 값은 이 sd만큼 흔들린다")
print(f"\n총 {time.time() - T0:.0f}초")
