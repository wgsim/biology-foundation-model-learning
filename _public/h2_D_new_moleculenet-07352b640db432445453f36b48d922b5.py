# 트랙 D 질문의 다른 절반 (모듈 05 과제 3): "노트북 05/01의 비교(동결 ChemBERTa-MLM vs ECFP4, 분할 3종)가 다른 MoleculeNet 과제에서도 같은 방향인가?" — 템플릿 구조로
# 새 과제 3개(승인 후 내려받음): FreeSolv(SAMPL.csv, 회귀), BACE(bace.csv, 이진 분류 Class), ClinTox(clintox.csv.gz, 이진 분류 CT_TOX). 파이프라인은 노트북 05/01과 같다.
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_D_new_moleculenet_tasks.py` (캐시만 사용. CSV 3개는 .cache/molecules/에 이미 내려받아 두었다)
import hashlib, importlib.metadata as md, json, subprocess, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.metrics import roc_auc_score
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
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------- 0. 사전 등록 (작성자는 모듈 05 노트북 01에서 ESOL·Lipophilicity·BBBP의 결과를 이미 보았다) ----------
RUN_NAME = "h2_D_new_moleculenet_tasks"
TASKS = {
    "freesolv": dict(file="SAMPL.csv", smi="smiles", y="expt", kind="reg", title="FreeSolv ΔG_hydr(실험) — RMSE↓", sha="ab5895d914ee87cb563bd7b9611e869527bba45bec6b014d34dc495a0f9dcb72"),
    "bace": dict(file="bace.csv", smi="mol", y="Class", kind="clf", title="BACE Class — ROC-AUC↑(폴드 평균)", sha="f3fb9ce90bada3e2bd6148b0df13f8f8145a357bf87df0dd5b391ede974fc737"),
    "clintox": dict(file="clintox.csv.gz", smi="smiles", y="CT_TOX", kind="clf", title="ClinTox CT_TOX — ROC-AUC↑(폴드 평균)", sha="7324a84a180ad9068d3847677c85d58e8432e82b3ae6ac95cf0e98b3a58d9697"),
}
MODELS = {"MLM": ("DeepChem/ChemBERTa-77M-MLM", "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"), "MTR": ("DeepChem/ChemBERTa-77M-MTR", "66b895cab8adebea0cb59a8effa66b2020f204ca")}
SPLITS = ["random", "scaffold", "cluster"]
REP_NAMES = ["ECFP4", "RDKit-217", "RDKit-8", "ChemBERTa-MLM", "ChemBERTa-MTR", "MLM-random-init"]
N_PERM = 10                             # 음성 대조의 라벨 순열 횟수
K, R, B = 5, 3, 300                     # 5-fold × 3회 반복, 군집 부트스트랩 300회 (노트북 05/01과 같음)
SIM_THR = 0.4                           # Butina 유사도 컷오프 (거리 0.6)
ALPHAS = np.logspace(-2, 5, 15)
SEED = 0
PRIMARY = dict(head="linear", comparisons=["(a) MLM − ECFP4", "(b) MLM − 무작위 초기화"])
# 개선량(양수 = MLM이 낫다): 회귀는 RMSE(비교 대상) − RMSE(MLM), 분류는 AUROC(MLM) − AUROC(비교 대상). 규칙: 반복 3회 모두 개선 + 평균 개선 ≥ 최소 효과 + 군집 부트스트랩 구간 하한 > 0 → '지지'
MIN_EFFECT = dict(reg=0.05, clf=0.02)   # 회귀는 y 표준편차의 0.05배, 분류는 AUROC 0.02
MIN_WINS = R


def decide(deltas, ci, min_effect):
    deltas = np.asarray(deltas, float)
    mean_d, wins = float(np.mean(deltas)), int(np.sum(deltas > 0))
    ok = mean_d >= min_effect and wins >= MIN_WINS and (ci is None or ci[0] > 0)
    return ("지지" if ok else "지지되지 않음"), mean_d, wins


# ---------- 1. 출처 ----------
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


import rdkit

LOCAL = {k: Path(snapshot_download(repo, revision=rev, allow_patterns=["pytorch_model.bin", "*.json", "merges.txt"])) for k, (repo, rev) in MODELS.items()}
for k, t in TASKS.items():
    assert sha256(MOL / t["file"]) == t["sha"], f"{t['file']}: sha256이 내려받을 때 기록한 값과 다르다"
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(), device=str(DEVICE),
            data={t["file"]: t["sha"][:16] + "…" for t in TASKS.values()}, source="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/ (MoleculeNet; 파일별 크기: SAMPL 32,060 B · clintox 19,870 B · bace 3,897,767 B)",
            license="미확인 — 원 출처별 라이선스를 확인하지 못했으므로 재배포하지 않고 로컬 캐시로만 쓴다 (노트북 05/01과 같은 취급)",
            models={k: dict(repo=v[0], revision=v[1]) for k, v in MODELS.items()}, k=K, repeats=R, boot=B, sim_thr=SIM_THR, primary=PRIMARY,
            packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "torch", "transformers"]} | {"rdkit": rdkit.__version__})
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))
print("device:", DEVICE)


# ---------- 2. 데이터 점검 (노트북 05/01 load_task와 같은 규칙) ----------
def load_task(key):
    cfg = TASKS[key]
    raw = pd.read_csv(MOL / cfg["file"])
    raw["can"] = [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(s)) is not None else None for s in raw[cfg["smi"]]]
    n_invalid = int(raw["can"].isna().sum())
    d = raw.dropna(subset=["can"])[["can", cfg["y"]]].rename(columns={cfg["y"]: "y"})
    g = d.groupby("can", sort=False)["y"]
    spread = g.agg(lambda s: s.max() - s.min())
    n_dup_groups = int((g.size() > 1).sum())
    n_conflict = int((spread > 0).sum())
    out = g.mean().reset_index() if cfg["kind"] == "reg" else g.first()[spread == 0].reset_index()
    mols = [Chem.MolFromSmiles(s) for s in out["can"]]
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    group = np.array([s if s != "" else f"acyclic_{i}" for i, s in enumerate(scaf)])
    stats = dict(raw=len(raw), invalid=n_invalid, dup_groups=n_dup_groups, conflicting=n_conflict, final=len(out), acyclic=int((np.array(scaf) == "").sum()))
    return dict(df=out, mols=mols, group=group, y=out["y"].values.astype(float), kind=cfg["kind"], stats=stats, raw=raw)


DATA = {k: load_task(k) for k in TASKS}
rows = []
for k, d in DATA.items():
    vc = pd.Series(d["group"]).value_counts()
    rows.append(dict(task=k, **d["stats"], scaffold_groups=len(vc), singleton_groups=int((vc == 1).sum()), largest_group=int(vc.iloc[0]),
                     y=(f"{d['y'].mean():.2f}±{d['y'].std():.2f}" if d["kind"] == "reg" else f"양성 {d['y'].mean():.3f}")))
print(pd.DataFrame(rows).set_index("task").to_string())
bace_cols = [c for c in DATA["bace"]["raw"].columns[:8]]
print("BACE 열(앞 8개):", bace_cols, "| 시간·날짜 열:", [c for c in DATA["bace"]["raw"].columns if any(w in c.lower() for w in ("date", "time", "year"))] or "없음",
      "| Model 열 값:", DATA["bace"]["raw"]["Model"].value_counts().to_dict(), "(이 열의 의미는 확인하지 못했다; 날짜·시간 열은 없다)")
fs = DATA["freesolv"]["raw"]
print(f"FreeSolv 참고: 'calc'(FreeSolv가 제공한 계산값; 비학습 기준) vs expt RMSE {np.sqrt(np.mean((fs.calc - fs.expt) ** 2)):.3f}, y 표준편차 {fs.expt.std():.3f}")

# ---------- 3. 분할과 누출 ----------
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def balanced_group_folds(groups, k, seed):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    order = np.lexsort((rng.random(len(u)), -sizes))
    load, assign = np.zeros(k), np.zeros(len(u), int)
    for g in order:
        f = int(np.argmin(load))
        assign[g] = f
        load[f] += sizes[g]
    fo = assign[inv]
    return [(np.where(fo != f)[0], np.where(fo == f)[0]) for f in range(k)]


def butina_ids(fps, sim_thr=SIM_THR):
    n = len(fps)
    dists = []
    for i in range(1, n):
        dists.extend((1.0 - np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))).tolist())
    clusters = Butina.ClusterData(dists, n, 1.0 - sim_thr, isDistData=True)
    ids = np.zeros(n, int)
    for c, members in enumerate(clusters):
        ids[list(members)] = c
    return ids


FPS, GROUPS, FOLDS, NN = {}, {}, {}, {}
for k, d in DATA.items():
    FPS[k] = [FPGEN.GetFingerprint(m) for m in d["mols"]]
    GROUPS[k] = {"random": np.arange(len(d["y"])), "scaffold": d["group"], "cluster": butina_ids(FPS[k])}
    FOLDS[k] = {}
    for sp, grp in GROUPS[k].items():
        FOLDS[k][sp] = ([list(KFold(K, shuffle=True, random_state=r).split(np.arange(len(grp))))
                         for r in range(R)] if sp == "random" else [balanced_group_folds(grp, K, r) for r in range(R)])
leak = []
for k in DATA:
    for sp in SPLITS:
        tr, te = FOLDS[k][sp][0][0]
        trf = [FPS[k][j] for j in tr]
        nn_ = np.array([max(DataStructs.BulkTanimotoSimilarity(FPS[k][i], trf)) for i in te])
        leak.append(dict(task=k, split=sp, 군집_또는_그룹수=len(np.unique(GROUPS[k][sp])), nn_sim_중앙값=np.median(nn_), nn_sim_p90=np.percentile(nn_, 90)))
print("\n--- 누출 점검 (첫 반복·첫 폴드의 test→train 최근접 Tanimoto) ---")
print(pd.DataFrame(leak).round(3).to_string(index=False))


# ---------- 4. 표현 ----------
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


RD8 = ["MolLogP", "MolWt", "TPSA", "NumRotatableBonds", "NumHDonors", "NumHAcceptors", "RingCount", "FractionCSP3"]


@torch.no_grad()
def chemberta_mean(model_key, smiles, random_init=False, batch=128):
    p = LOCAL[model_key]
    tok = AutoTokenizer.from_pretrained(p)
    if random_init:
        torch.manual_seed(SEED)
        model = AutoModel.from_config(AutoConfig.from_pretrained(p), add_pooling_layer=False)
    else:
        model = AutoModel.from_pretrained(p, add_pooling_layer=False)
    model = model.to(DEVICE).eval()
    order = np.argsort([len(s) for s in smiles])
    out, n_trunc = None, 0
    for s in range(0, len(smiles), batch):
        ix = order[s:s + batch]
        b = tok([smiles[i] for i in ix], return_tensors="pt", padding=True, truncation=True, max_length=512)
        n_trunc += int((b["attention_mask"].sum(1) >= 512).sum())
        h = model(**{kk: v.to(DEVICE) for kk, v in b.items()}).last_hidden_state.float().cpu()
        am = b["attention_mask"].clone()
        last = am.sum(1) - 1
        am[:, 0] = 0
        am[torch.arange(len(ix)), last] = 0                  # <s>, </s> 제외한 평균 풀링
        w = am[:, :, None].float()
        m_ = (h * w).sum(1) / w.sum(1).clamp(min=1)
        if out is None:
            out = np.zeros((len(smiles), h.shape[-1]), np.float32)
        out[ix] = m_.numpy()
    return out, n_trunc


REPS = {}
t0 = time.time()
for k, d in DATA.items():
    smi = d["df"]["can"].tolist()
    X = {"ECFP4": np.log1p(np.stack([FPGEN.GetCountFingerprintAsNumPy(m) for m in d["mols"]]).astype(np.float32))}
    desc = pd.DataFrame([Descriptors.CalcMolDescriptors(m) for m in d["mols"]])
    X["RDKit-8"] = np.nan_to_num(desc[RD8].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    full = desc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    full = full.loc[:, full.std() > 0]                       # 노트북 05/01과 같이 *전체* 데이터에서 상수인 열만 제거(라벨은 쓰지 않는 transductive 전처리이며, 폴드 안에서 하지 않았다)
    X["RDKit-217"] = signed_log1p(np.clip(full.values.astype(np.float64), -1e12, 1e12))
    X["ChemBERTa-MLM"], nt = chemberta_mean("MLM", smi)
    X["ChemBERTa-MTR"], _ = chemberta_mean("MTR", smi)
    X["MLM-random-init"], _ = chemberta_mean("MLM", smi, random_init=True)
    REPS[k] = X
    print(f"{k:9s} n={len(smi)} | " + ", ".join(f"{n}:{v.shape[1]}" for n, v in X.items()) + f" | 512토큰 초과 {nt}개")
print(f"표현 {time.time() - t0:.0f}s")


# ---------- 5. 평가 (노트북 05/01과 같은 head·절차) ----------
def make_model(rep, head, kind):
    # 선형 head의 alpha는 노트북 05/01과 같은 RidgeCV/RidgeClassifierCV(내부 leave-one-out이며 분할 그룹을 고려하지 않는다. 표준화는 훈련 폴드 전체로 한 번 적합한다)
    if head == "linear":
        core = RidgeCV(alphas=ALPHAS) if kind == "reg" else RidgeClassifierCV(alphas=ALPHAS)
        return core if rep == "ECFP4" else make_pipeline(StandardScaler(), core)
    cls = RandomForestRegressor if kind == "reg" else RandomForestClassifier
    return cls(n_estimators=100, max_features="sqrt", n_jobs=-1, random_state=SEED)


def predict(model, X, kind):
    if kind == "reg":
        return model.predict(X)
    return model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]


def run_cv(Xr, y, folds_r, rep, head, kind):
    out = np.zeros((len(folds_r), len(y)))
    for r, folds in enumerate(folds_r):
        for tr, te in folds:
            out[r, te] = predict(make_model(rep, head, kind).fit(Xr[tr], y[tr]), Xr[te], kind)
    return out


OOF = {}
t0 = time.time()
for k, d in DATA.items():
    for sp in SPLITS:
        for rep in REP_NAMES:
            for head in ["linear", "rf"]:
                OOF[(k, sp, rep, head)] = run_cv(REPS[k][rep], d["y"], FOLDS[k][sp], rep, head, d["kind"])
    # 음성 대조: 각 폴드에서 *훈련* 라벨만 섞은 ECFP4 선형 head를 원래 test 라벨로 평가 (scaffold 분할, 순열 N_PERM회. 순열 r은 반복 r % R의 폴드를 쓴다).
    # (전체 라벨을 한 번 섞어 모든 폴드에 쓰고 원래 라벨로 평가하면 AUROC가 0.5보다 낮게 치우쳐 나왔다. 첫 실행은 BACE 0.458, ClinTox 0.437이었다. 그래서 폴드 안 순열로 바꿨다)
    for sp in ["scaffold"]:
        sh = np.zeros((N_PERM, len(d["y"])))
        for r in range(N_PERM):
            rng_s = np.random.default_rng(100 + r)
            for tr, te in FOLDS[k][sp][r % R]:
                sh[r, te] = predict(make_model("ECFP4", "linear", d["kind"]).fit(REPS[k]["ECFP4"][tr], rng_s.permutation(d["y"][tr])), REPS[k]["ECFP4"][te], d["kind"])
        OOF[(k, sp, "ECFP4-shuffled", "linear")] = sh
    print(f"{k} 완료 (누적 {time.time() - t0:.0f}s)", flush=True)

FOLD_ID = {}
for k in DATA:
    for sp in SPLITS:
        arr = np.zeros((R, len(DATA[k]["y"])), int)
        for r_, folds_ in enumerate(FOLDS[k][sp]):
            for f_, (_, te_) in enumerate(folds_):
                arr[r_, te_] = f_
        FOLD_ID[(k, sp)] = arr


def fold_metric(kind, y, p, fold_id, w=None):
    if kind == "reg":
        return float(np.sqrt(np.average((p - y) ** 2, weights=w)))
    aucs = []
    for f in np.unique(fold_id):
        m = fold_id == f
        try:
            aucs.append(roc_auc_score(y[m], p[m], sample_weight=None if w is None else w[m]))
        except ValueError:
            pass
    return float(np.mean(aucs))


def per_repeat(k, sp, rep, head):
    d = DATA[k]
    return np.array([fold_metric(d["kind"], d["y"], o, FOLD_ID[(k, sp)][r_]) for r_, o in enumerate(OOF[(k, sp, rep, head)])])


def boot_weights(group, B, seed=123):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(group, return_inverse=True)
    draws = rng.integers(0, len(u), (B, len(u)))
    return np.stack([np.bincount(dr, minlength=len(u))[inv] for dr in draws]).astype(float)


W_BOOT = {k: boot_weights(GROUPS[k]["cluster"], B) for k in DATA}
_BOOT = {}


def boot_metric(k, sp, rep, head):
    key = (k, sp, rep, head)
    if key not in _BOOT:
        d, W, oof = DATA[k], W_BOOT[k], OOF[key]
        if d["kind"] == "reg":
            se = (oof - d["y"][None]) ** 2
            _BOOT[key] = np.sqrt((W @ se.T) / W.sum(1, keepdims=True)).mean(1)
        else:
            _BOOT[key] = np.array([np.mean([fold_metric("clf", d["y"], o, FOLD_ID[(k, sp)][r_], w) for r_, o in enumerate(oof)]) for w in W])
    return _BOOT[key]


def improvement(k, sp, head, rep_a, rep_b):
    """개선량(양수 = rep_a가 낫다): 반복별 값, 점추정, 구간."""
    kind = DATA[k]["kind"]
    sgn = -1.0 if kind == "reg" else 1.0                  # RMSE는 작을수록 좋다
    pa, pb = per_repeat(k, sp, rep_a, head), per_repeat(k, sp, rep_b, head)
    d_rep = sgn * (pa - pb)
    bd = sgn * (boot_metric(k, sp, rep_a, head) - boot_metric(k, sp, rep_b, head))
    return d_rep, np.percentile(bd, [2.5, 97.5])


for k in DATA:
    print(f"\n===== {TASKS[k]['title']} · 선형 head · 3회 반복 평균 [군집 부트스트랩 95%] =====")
    tab = {}
    for sp in SPLITS:
        col = {}
        for rep in REP_NAMES:
            bm = boot_metric(k, sp, rep, "linear")
            lo, hi = np.percentile(bm, [2.5, 97.5])
            col[rep] = f"{per_repeat(k, sp, rep, 'linear').mean():.3f} [{lo:.3f},{hi:.3f}]"
        tab[sp] = col
    print(pd.DataFrame(tab).to_string())
    shv = np.array([fold_metric(DATA[k]["kind"], DATA[k]["y"], o, FOLD_ID[(k, 'scaffold')][r_ % R]) for r_, o in enumerate(OOF[(k, 'scaffold', 'ECFP4-shuffled', 'linear')])])
    print(f"음성 대조(ECFP4, 라벨 섞기 {N_PERM}회, scaffold): 평균 {shv.mean():.3f} (sd {shv.std():.3f}, 범위 {shv.min():.3f}~{shv.max():.3f}) ({'y 표준편차 ' + format(DATA[k]['y'].std(), '.3f') + '에 가까워야 함' if DATA[k]['kind'] == 'reg' else '평균이 0.5 근처여야 함; 한 번의 값은 이 sd만큼 흔들린다'})")

print("\n=== 주 비교 (선형 head, 개선량: 양수 = 앞의 것이 낫다) ===")
SUMMARY = []
for k in DATA:
    kind = DATA[k]["kind"]
    me = MIN_EFFECT[kind] * (DATA[k]["y"].std() if kind == "reg" else 1.0)
    for sp in SPLITS:
        out = [f"{k:9s} {sp:8s}"]
        for lab, (ra, rb) in [("(a) MLM−ECFP4", ("ChemBERTa-MLM", "ECFP4")), ("(b) MLM−무작위초기화", ("ChemBERTa-MLM", "MLM-random-init"))]:
            d_rep, ci = improvement(k, sp, "linear", ra, rb)
            dec, mean_d, wins = decide(d_rep, tuple(ci), me)
            out.append(f"{lab} {mean_d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {wins}/{R} → {dec}")
            SUMMARY.append((k, sp, lab, mean_d, ci, dec))
        print(" | ".join(out) + f"   (최소 효과 {me:.3f})")

print("\n=== 민감도: RF head로 같은 (a) 비교 ===")
for k in DATA:
    kind = DATA[k]["kind"]
    me = MIN_EFFECT[kind] * (DATA[k]["y"].std() if kind == "reg" else 1.0)
    for sp in SPLITS:
        d_rep, ci = improvement(k, sp, "rf", "ChemBERTa-MLM", "ECFP4")
        dec, mean_d, wins = decide(d_rep, tuple(ci), me)
        print(f"  {k:9s} {sp:8s} RF (a) MLM−ECFP4 {mean_d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {wins}/{R} → {dec}")

print("\n=== 참고: 가장 단순한 baseline과 MTR ===")
for k in DATA:
    for sp in SPLITS:
        d1, c1 = improvement(k, sp, "linear", "ChemBERTa-MLM", "RDKit-8")
        d2, c2 = improvement(k, sp, "linear", "ChemBERTa-MTR", "ChemBERTa-MLM")
        print(f"  {k:9s} {sp:8s} MLM−RDKit-8 {d1.mean():+.3f} [{c1[0]:+.3f},{c1[1]:+.3f}] | MTR−MLM {d2.mean():+.3f} [{c2[0]:+.3f},{c2[1]:+.3f}]")
print(f"\n총 {time.time() - T0:.0f}초")
