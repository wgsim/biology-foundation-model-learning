# 트랙 D 시험 실행: Lipophilicity에서 Butina 군집 분할의 유사도 컷오프를 연속으로 바꿀 때 ChemBERTa-MLM − ECFP4가 바뀌는가 (모듈 05 과제 1)
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/track_D_lipo_cutoff_sweep.py` (캐시만 사용, 다운로드 없음)
import hashlib, importlib.metadata as md, json, subprocess, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
hf_logging.set_verbosity_error()
T0 = time.time()
ROOT = Path.cwd()
MOL = ROOT / ".cache" / "molecules"

# ---------- 0. 사전 등록 ----------
RUN_NAME = "dry_D_lipo_cutoff_sweep"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6]                  # Butina 유사도 컷오프 (결과를 보기 전에 고정)
REPEATS, K = 5, 5                                  # 반복(시드) 5회 × 5-fold
MODEL = ("DeepChem/ChemBERTa-77M-MLM", "ed8a5374f2024ec8da53760af91a33fb8f6a15ff")
PRIMARY = dict(model="ChemBERTa-MLM", baseline="ECFP4", head="linear(RidgeCV)", metric="RMSE")
MIN_EFFECT, MIN_WINS, BOOT, MIN_TEST_GROUPS = 0.02, 4, 300, 8     # Δ = RMSE_ECFP4 − RMSE_model (양수 = 모델이 낫다)
ALPHAS = np.logspace(-2, 5, 15)
SEED = 0


def decide(deltas, ci):
    deltas = np.asarray(deltas, float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < MIN_WINS:
        return "판정 불가(유효 시드 부족)", float("nan"), 0
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


import rdkit

p_model = Path(snapshot_download(MODEL[0], revision=MODEL[1], allow_patterns=["pytorch_model.bin", "*.json", "merges.txt"]))
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            sha256={"Lipophilicity.csv": sha256(MOL / "Lipophilicity.csv")[:16] + "…", "pytorch_model.bin": sha256(p_model / "pytorch_model.bin")[:16] + "…"},
            model=dict(repo=MODEL[0], revision=MODEL[1]), packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "torch", "transformers"]} | {"rdkit": rdkit.__version__},
            thresholds=THRESHOLDS, primary=PRIMARY)
assert sha256(MOL / "Lipophilicity.csv") == "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME                  # git 제외 폴더: 실행 기록은 여기에만 쓴다
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))
print("provenance:", json.dumps(PROV, ensure_ascii=False)[:300], "...")

# ---------- 2. 데이터 점검 (노트북 05/01 load_task와 같은 규칙) ----------
raw = pd.read_csv(MOL / "Lipophilicity.csv")
raw["can"] = [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(s)) is not None else None for s in raw["smiles"]]
d = raw.dropna(subset=["can"])[["can", "exp"]].rename(columns={"exp": "y"})
g = d.groupby("can", sort=False)["y"]
df = g.mean().reset_index()
mols = [Chem.MolFromSmiles(s) for s in df["can"]]
y = df["y"].values.astype(float)
print(f"원본 {len(raw)} | 무효 SMILES {int(raw['can'].isna().sum())} | 중복 그룹 {int((g.size() > 1).sum())} | 최종 {len(df)} | y {y.mean():.2f}±{y.std():.2f}")
assert df["can"].is_unique

# ---------- 3. 분할과 누출 ----------
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
FPS = [FPGEN.GetFingerprint(m) for m in mols]
t = time.time()
dists = []
for i in range(1, len(FPS)):
    dists.extend((1.0 - np.asarray(DataStructs.BulkTanimotoSimilarity(FPS[i], FPS[:i]))).tolist())
print(f"거리 행렬 {time.time() - t:.0f}s")


def butina_ids(thr):
    clusters = Butina.ClusterData(dists, len(FPS), 1.0 - thr, isDistData=True)
    ids = np.zeros(len(FPS), int)
    for c, members in enumerate(clusters):
        ids[list(members)] = c
    return ids


def balanced_group_folds(groups, k, seed):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    order = np.lexsort((rng.random(len(u)), -sizes))
    load, assign = np.zeros(k), np.zeros(len(u), int)
    for gi in order:
        f = int(np.argmin(load))
        assign[gi] = f
        load[f] += sizes[gi]
    fo = assign[inv]
    return [(np.where(fo != f)[0], np.where(fo == f)[0]) for f in range(k)]


GROUPS = {thr: butina_ids(thr) for thr in THRESHOLDS}
GROUPS["random"] = np.arange(len(FPS))
FOLDS = {}
for key, grp in GROUPS.items():
    FOLDS[key] = ([list(KFold(K, shuffle=True, random_state=r).split(np.arange(len(grp)))) for r in range(REPEATS)] if key == "random"
                  else [balanced_group_folds(grp, K, r) for r in range(REPEATS)])


def nearest_train_sim(folds):
    sims = []
    for tr, te in folds:
        trf = [FPS[j] for j in tr]
        sims += [max(DataStructs.BulkTanimotoSimilarity(FPS[i], trf)) for i in te]
    return np.array(sims)


rows = []
for key in ["random"] + THRESHOLDS:
    nn = nearest_train_sim(FOLDS[key][0])
    n_cl = len(np.unique(GROUPS[key]))
    rows.append(dict(split=f"cluster@{key}" if key != "random" else "random", 군집수=n_cl, 최소_test_군집수=min(len(np.unique(GROUPS[key][te])) for _, te in FOLDS[key][0]),
                     nn_sim_중앙값=np.median(nn), nn_sim_p90=np.percentile(nn, 90), 컷오프_이상_비율=(np.mean(nn >= key) if key != "random" else np.nan), test_폴드크기="/".join(str(len(te)) for _, te in FOLDS[key][0])))
LEAK = pd.DataFrame(rows)
print(LEAK.round(3).to_string(index=False))

# ---------- 4. 특징 ----------
X_ECFP = np.log1p(np.stack([FPGEN.GetCountFingerprintAsNumPy(m) for m in mols]).astype(np.float32))


@torch.no_grad()
def chemberta_mean(smiles, random_init=False, batch=128):
    tok = AutoTokenizer.from_pretrained(p_model)
    if random_init:
        torch.manual_seed(SEED)
        model = AutoModel.from_config(AutoConfig.from_pretrained(p_model), add_pooling_layer=False)
    else:
        model = AutoModel.from_pretrained(p_model, add_pooling_layer=False)
    model.eval()
    order = np.argsort([len(s) for s in smiles])
    out = None
    for s in range(0, len(smiles), batch):
        ix = order[s:s + batch]
        b = tok([smiles[i] for i in ix], return_tensors="pt", padding=True, truncation=True, max_length=512)
        h = model(**b).last_hidden_state
        am = b["attention_mask"].clone()
        last = am.sum(1) - 1
        am[:, 0] = 0
        am[torch.arange(len(ix)), last] = 0                  # <s>, </s> 제외한 평균 풀링 (노트북 05/01과 같은 규칙)
        w = am[:, :, None].float()
        m_ = (h * w).sum(1) / w.sum(1).clamp(min=1)
        if out is None:
            out = np.zeros((len(smiles), h.shape[-1]), np.float32)
        out[ix] = m_.numpy()
    return out


t = time.time()
smi = df["can"].tolist()
REPS = {"ECFP4": X_ECFP, "ChemBERTa-MLM": chemberta_mean(smi), "MLM-random-init": chemberta_mean(smi, random_init=True)}
print(f"표현 {time.time() - t:.0f}s | " + ", ".join(f"{k}:{v.shape[1]}" for k, v in REPS.items()))


# ---------- 5. 평가 ----------
def make_model(rep):
    core = RidgeCV(alphas=ALPHAS)
    return core if rep == "ECFP4" else make_pipeline(StandardScaler(), core)


def run_cv(X, yy, folds_r, rep):
    out = np.zeros((len(folds_r), len(yy)))
    for r, folds in enumerate(folds_r):
        for tr, te in folds:
            out[r, te] = make_model(rep).fit(X[tr], yy[tr]).predict(X[te])
    return out


t = time.time()
OOF = {}
for key in ["random"] + THRESHOLDS:
    for rep, X in REPS.items():
        OOF[(key, rep)] = run_cv(X, y, FOLDS[key], rep)
    OOF[(key, "ECFP4-shuffled")] = run_cv(X_ECFP, np.random.default_rng(SEED).permutation(y), FOLDS[key], "ECFP4")      # 음성 대조(전체 라벨 섞기; 진단용)
    print(f"  split={key} 완료 (누적 {time.time() - t:.0f}s)", flush=True)

rmse = lambda o, yy=y: np.sqrt(np.mean((o - yy[None]) ** 2, axis=1))          # 반복별 RMSE
print("\n--- 반복별 RMSE 평균 (±sd) ---")
tab = pd.DataFrame({rep: {str(key): f"{rmse(OOF[(key, rep)]).mean():.3f}±{rmse(OOF[(key, rep)]).std():.3f}" for key in ["random"] + THRESHOLDS} for rep in list(REPS) + ["ECFP4-shuffled"]})
print(tab.to_string())
print(f"(참고: y의 표준편차 {y.std():.3f})")


def boot_weights(group, B=BOOT, seed=123):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(group, return_inverse=True)
    draws = rng.integers(0, len(u), (B, len(u)))
    return np.stack([np.bincount(dr, minlength=len(u))[inv] for dr in draws]).astype(float)


def paired(key, model, base):
    d_rep = rmse(OOF[(key, base)]) - rmse(OOF[(key, model)])                  # 반복(시드)별 Δ, 양수 = 모델이 낫다
    W = boot_weights(GROUPS[key] if key != "random" else GROUPS[0.4])         # 재표본 단위: 해당 분할의 군집 (random 분할은 0.4 군집으로 재표본)
    se_m, se_b = (OOF[(key, model)] - y[None]) ** 2, (OOF[(key, base)] - y[None]) ** 2
    bm = np.sqrt((W @ se_m.T) / W.sum(1, keepdims=True)).mean(1)
    bb = np.sqrt((W @ se_b.T) / W.sum(1, keepdims=True)).mean(1)
    ci = np.percentile(bb - bm, [2.5, 97.5])
    ok_groups = LEAK.set_index("split").loc[f"cluster@{key}" if key != "random" else "random", "최소_test_군집수"] >= MIN_TEST_GROUPS
    dec, mean_d, wins = decide(d_rep, tuple(ci) if ok_groups else None)
    return f"Δ(RMSE_{base} − RMSE_{model}) {mean_d:+.3f} 95%[{ci[0]:+.3f},{ci[1]:+.3f}] | 시드 Δ>0 {wins}/{len(d_rep)} → {dec}"


print("\n=== 주 비교: ChemBERTa-MLM vs ECFP4 (양수 = MLM이 낫다) ===")
for key in ["random"] + THRESHOLDS:
    print(f"split={str(key):6s}", paired(key, "ChemBERTa-MLM", "ECFP4"))
print("\n=== 대조: 사전학습 효과 ChemBERTa-MLM vs 무작위 초기화 ===")
for key in ["random"] + THRESHOLDS:
    print(f"split={str(key):6s}", paired(key, "ChemBERTa-MLM", "MLM-random-init"))
print(f"\n총 {time.time() - T0:.0f}초")
