# 트랙 D 질문의 다른 절반 (모듈 05 과제 2): "표기 불변성은 데이터 증강으로 해결되는가?" — 템플릿 구조로
# Lipophilicity, scaffold 분할, 동결 ChemBERTa-77M-MLM 임베딩(평균 풀링, 마지막 층) + 선형(ridge) head.
# 비교: (a) 정규화 SMILES로만 학습, (b) 학습 분자의 무작위 표기 K개를 "평균" 또는 "행으로 증강", (c) 추론 때 입력을 항상 정규화.
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_D_notation_augmentation.py` (캐시만 사용, 다운로드 없음)
import hashlib, importlib.metadata as md, json, os, subprocess, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
hf_logging.set_verbosity_error()
T0 = time.time()
ROOT = Path.cwd()
MOL = ROOT / ".cache" / "molecules"

# ---------- 0. 사전 등록 (작성자는 모듈 05 노트북 §8에서 (a)의 붕괴 0.863 → 1.448을 이미 보았다) ----------
RUN_NAME = "h2_D_notation_augmentation"
MODEL = ("DeepChem/ChemBERTa-77M-MLM", "ed8a5374f2024ec8da53760af91a33fb8f6a15ff")
KS = [1, 5, 20]                                 # 증강 개수 K
N_TTA = 5                                       # 추론 때 무작위 표기를 평균하는 개수(보조 조건)
REPEATS, K_FOLD, K_INNER = 3, 5, 3              # 바깥 scaffold 분할 반복 3 × 5-fold, 안쪽 alpha 선택은 분자(scaffold) 단위 3-fold
ALPHAS = np.logspace(-2, 6, 17)                 # 증강하면 행 수가 K배라 최적 alpha도 커지므로 노트북(1e5까지)보다 넓게 잡는다
PRIMARY = dict(variant="rows5", baseline="canon", test="rand1")   # 증강(행 K=5)을 학습한 head vs 정규화만 학습한 head, test는 분자당 무작위 표기 1개
MIN_EFFECT, MIN_WINS, BOOT = 0.10, 3, 300       # 반복 3회 중 3회 모두 개선 + 평균 RMSE 개선 ≥ 0.10 + 군집 부트스트랩 구간 하한 > 0
SOLVED_TOL = 0.05                               # '해결됨': 증강 head의 무작위 표기 RMSE − 정규화 head의 정규화 RMSE ≤ 0.05
SEED = 0
DEVICE = os.environ.get("H2_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")


def decide(deltas, ci):
    deltas = np.asarray(deltas, float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < MIN_WINS:
        return "판정 불가(유효 반복 부족)", float("nan"), 0
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
assert sha256(MOL / "Lipophilicity.csv") == "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            sha256={"Lipophilicity.csv": sha256(MOL / "Lipophilicity.csv")[:16] + "…", "pytorch_model.bin": sha256(p_model / "pytorch_model.bin")[:16] + "…"},
            model=dict(repo=MODEL[0], revision=MODEL[1]), device=DEVICE,
            packages={k: md.version(k) for k in ["numpy", "pandas", "torch", "transformers"]} | {"rdkit": rdkit.__version__},
            ks=KS, n_tta=N_TTA, primary=PRIMARY)
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))
print("device:", DEVICE)

# ---------- 2. 데이터 점검 (노트북 05/01 load_task와 같은 규칙) ----------
raw = pd.read_csv(MOL / "Lipophilicity.csv")
raw["can"] = [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(s)) is not None else None for s in raw["smiles"]]
d = raw.dropna(subset=["can"])[["can", "exp"]].rename(columns={"exp": "y"})
df = d.groupby("can", sort=False)["y"].mean().reset_index()
mols = [Chem.MolFromSmiles(s) for s in df["can"]]
y = df["y"].values.astype(float)
N = len(y)
assert df["can"].is_unique and N == 4200
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
GROUP = np.array([s if s != "" else f"acyclic_{i}" for i, s in enumerate(scaf)])
print(f"분자 {N} | scaffold 그룹 {len(np.unique(GROUP))} | y {y.mean():.2f}±{y.std():.2f}")


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


FOLDS = [balanced_group_folds(GROUP, K_FOLD, r) for r in range(REPEATS)]       # 노트북 05/01의 scaffold 분할과 같은 함수·시드


# ---------- 4. 특징: 정규화 / 무작위 표기 임베딩 ----------
def random_smiles(set_id):
    out = []
    for i, m in enumerate(mols):
        s = Chem.MolToRandomSmilesVect(m, 1, randomSeed=100003 * set_id + i)[0]
        assert Chem.CanonSmiles(s) == df["can"].iloc[i], "무작위 SMILES가 같은 분자가 아니다"
        out.append(s)
    return out


@torch.no_grad()
def chemberta_mean(smiles, device=DEVICE, batch=128):
    tok = AutoTokenizer.from_pretrained(p_model)
    model = AutoModel.from_pretrained(p_model, add_pooling_layer=False).to(device).eval()
    order = np.argsort([len(s) for s in smiles])
    out = None
    for s in range(0, len(smiles), batch):
        ix = order[s:s + batch]
        b = tok([smiles[i] for i in ix], return_tensors="pt", padding=True, truncation=True, max_length=512)
        h = model(**{k: v.to(device) for k, v in b.items()}).last_hidden_state.float().cpu()
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
E_CAN = chemberta_mean(df["can"].tolist())
print(f"정규화 임베딩 {time.time() - t:.0f}s")
if DEVICE != "cpu":
    ref = chemberta_mean(df["can"].tolist()[:64], device="cpu")
    print(f"장치 점검: {DEVICE} vs cpu 최대 절대차 {np.abs(ref - E_CAN[:64]).max():.2e} (임베딩 크기 중앙값 {np.median(np.abs(E_CAN[:64])):.2f})")
TRAIN_SETS = list(range(1, max(KS) + 1))
TEST_SETS = list(range(101, 101 + N_TTA))
SMI = {sid: random_smiles(sid) for sid in TRAIN_SETS + TEST_SETS}
same = np.mean([a == b for sid in TRAIN_SETS for a, b in zip(SMI[sid], df["can"])])
print(f"무작위 표기가 정규화 문자열과 같은 비율 {same:.3f}")
E_TR = np.stack([chemberta_mean(SMI[s]) for s in TRAIN_SETS])           # [20, N, 768]
print(f"학습용 무작위 표기 임베딩 {time.time() - t:.0f}s (누적)")
E_TE = np.stack([chemberta_mean(SMI[s]) for s in TEST_SETS])            # [5, N, 768]
print(f"임베딩 전체 {time.time() - t:.0f}s")
cos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
print(f"같은 분자 정규화 vs 무작위 표기 코사인 중앙값 {np.median(cos(E_CAN, E_TE[0])):.3f} | 다른 분자 {np.median(cos(E_CAN, E_CAN[np.random.default_rng(0).permutation(N)])):.3f}")

X_ECFP = np.log1p(np.stack([rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetCountFingerprintAsNumPy(m) for m in mols]).astype(np.float32))


# ---------- 5. ridge (alpha 경로를 고유분해로 한 번에) + 분자 단위 안쪽 CV ----------
def path_fit(X, yy, scale=True):
    X = X.astype(np.float64)
    mu = X.mean(0)
    sd = (X.std(0) + 1e-8) if scale else np.ones(X.shape[1])
    Z = (X - mu) / sd
    ym = yy.mean()
    ev, U = np.linalg.eigh(Z.T @ Z)
    return dict(mu=mu, sd=sd, ym=ym, ev=np.clip(ev, 0, None), U=U, Ub=U.T @ (Z.T @ (yy - ym)))


def path_predict(m, Xn, alpha):
    Zn = (Xn.astype(np.float64) - m["mu"]) / m["sd"]
    return (Zn @ m["U"]) @ (m["Ub"] / (m["ev"] + alpha)) + m["ym"]


def path_predict_all(m, Xn):
    T = ((Xn.astype(np.float64) - m["mu"]) / m["sd"]) @ m["U"]
    return np.stack([T @ (m["Ub"] / (m["ev"] + a)) + m["ym"] for a in ALPHAS])        # [n_alpha, n]


# 학습 특징 만들기: 분자 인덱스 → (X, y 행). 검증도 같은 분포(그 변형이 학습하는 입력)로 만든다
def build_canon(idx, yy):
    return E_CAN[idx], yy[idx]


def build_avg(k):
    return lambda idx, yy: (E_TR[:k, idx].mean(0), yy[idx])


def build_rows(k):
    return lambda idx, yy: (E_TR[:k, idx].reshape(-1, E_TR.shape[-1]), np.tile(yy[idx], k))


def build_ecfp(idx, yy):
    return X_ECFP[idx], yy[idx]


VARIANTS = {"canon": (build_canon, True), "ecfp4": (build_ecfp, False)}
for k in KS:
    VARIANTS[f"avg{k}"] = (build_avg(k), True)
    VARIANTS[f"rows{k}"] = (build_rows(k), True)

TEST_COND = {
    "canon": lambda te: E_CAN[te],
    "rand1": lambda te: E_TE[0, te],
    f"tta{N_TTA}": lambda te: E_TE[:N_TTA, te].mean(0),
}


def fit_select(build, scale, tr, yy, seed):
    """바깥 train 분자 안에서 scaffold 단위 안쪽 3-fold로 alpha를 고른 뒤 전체 train으로 다시 적합한다."""
    inner = balanced_group_folds(GROUP[tr], K_INNER, seed)
    err = np.zeros(len(ALPHAS))
    for itr, iva in inner:
        Xa, ya = build(tr[itr], yy)
        Xv, yv = build(tr[iva], yy)
        P = path_predict_all(path_fit(Xa, ya, scale), Xv)
        err += np.sqrt(((P - yv[None]) ** 2).mean(1))
    a_best = ALPHAS[int(np.argmin(err))]
    Xa, ya = build(tr, yy)
    return path_fit(Xa, ya, scale), a_best


OOF, ALPHA_CHOSEN = {}, {}
t = time.time()
for name, (build, scale) in VARIANTS.items():
    conds = {"canon": None} if name == "ecfp4" else TEST_COND
    for cn in conds:
        OOF[(name, cn)] = np.zeros((REPEATS, N))
    for r, folds in enumerate(FOLDS):
        for fi, (tr, te) in enumerate(folds):
            m, a = fit_select(build, scale, tr, y, seed=100 * r + fi)
            ALPHA_CHOSEN.setdefault(name, []).append(a)
            for cn in conds:
                Xt = X_ECFP[te] if name == "ecfp4" else TEST_COND[cn](te)
                OOF[(name, cn)][r, te] = path_predict(m, Xt, a)
    print(f"  {name:7s} 완료 (누적 {time.time() - t:.0f}s) | 선택된 alpha 중앙값 {np.median(ALPHA_CHOSEN[name]):.3g}", flush=True)
# 음성 대조: 훈련 라벨만 섞어 (정규화로) 학습한 뒤 정규화 test에서 평가한다
OOF[("canon-shuffled", "canon")] = np.zeros((REPEATS, N))
for r, folds in enumerate(FOLDS):
    for fi, (tr, te) in enumerate(folds):
        y_shuf = y.copy()
        y_shuf[tr] = np.random.default_rng(SEED + 100 * r + fi).permutation(y[tr])      # 각 폴드에서 훈련 라벨만 섞고 test는 원래 라벨로 평가
        m, a = fit_select(build_canon, True, tr, y_shuf, seed=100 * r + fi)
        OOF[("canon-shuffled", "canon")][r, te] = path_predict(m, E_CAN[te], a)

rmse = lambda o: np.sqrt(((o - y[None]) ** 2).mean(1))
print("\n--- RMSE (3회 반복 평균 ± sd), 행 = 학습 방식, 열 = test 입력 ---")
rows = []
for name in VARIANTS:
    r_ = {"학습": name}
    for cn in (["canon"] if name == "ecfp4" else TEST_COND):
        v = rmse(OOF[(name, cn)])
        r_[cn] = f"{v.mean():.3f}±{v.std():.3f}"
    rows.append(r_)
print(pd.DataFrame(rows).set_index("학습").fillna("(표기 무관)").to_string())
print(f"(음성 대조: 라벨 섞기 canon RMSE {rmse(OOF[('canon-shuffled', 'canon')]).mean():.3f}; y의 표준편차 {y.std():.3f})")
print("임베딩 횟수(분자당): 학습 canon 1, avgK/rowsK K회 | 추론 canon 1, rand1 1, tta{0} {0}회 (ECFP4는 임베딩 없음)".format(N_TTA))

# ---------- 6. 쌍 비교: 분자 군집(scaffold 그룹) 부트스트랩 ----------
def boot_weights(group, B=BOOT, seed=123):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(group, return_inverse=True)
    draws = rng.integers(0, len(u), (B, len(u)))
    return np.stack([np.bincount(dr, minlength=len(u))[inv] for dr in draws]).astype(float)


W = boot_weights(GROUP)


def paired(a, b):
    """Δ = RMSE(a) − RMSE(b) (양수 = b가 낫다). a, b = (변형, test 입력). 반복별 Δ, 구간(군집 부트스트랩), Δ>0인 반복 수."""
    d_rep = rmse(OOF[a]) - rmse(OOF[b])
    se_a, se_b = (OOF[a] - y[None]) ** 2, (OOF[b] - y[None]) ** 2
    ba = np.sqrt((W @ se_a.T) / W.sum(1, keepdims=True)).mean(1)
    bb = np.sqrt((W @ se_b.T) / W.sum(1, keepdims=True)).mean(1)
    return d_rep, np.percentile(ba - bb, [2.5, 97.5])


print("\n=== 주 비교: 증강(행 K=5)이 무작위 표기 test의 RMSE를 정규화-only 학습보다 줄이는가 ===")
d_rep, ci = paired(("canon", "rand1"), ("rows5", "rand1"))
dec, mean_d, wins = decide(d_rep, tuple(ci))
gap = rmse(OOF[("rows5", "rand1")]).mean() - rmse(OOF[("canon", "canon")]).mean()
solved = dec == "지지" and gap <= SOLVED_TOL
print(f"Δ(RMSE_canon-only − RMSE_rows5), test=rand1: {mean_d:+.3f} 95%[{ci[0]:+.3f},{ci[1]:+.3f}] | 반복 Δ>0 {wins}/{len(d_rep)} → {dec}")
print(f"'해결됨' 판정: 증강 head의 무작위 표기 RMSE − 정규화 head의 정규화 RMSE = {gap:+.3f} (허용 {SOLVED_TOL}) → {'해결됨' if solved else '해결되지 않음'}")

print("\n=== K에 따른 변화 (test = 무작위 표기 1개) ===")
for k in KS:
    for kind in ["avg", "rows"]:
        d_rep, ci = paired(("canon", "rand1"), (f"{kind}{k}", "rand1"))
        dec, mean_d, wins = decide(d_rep, tuple(ci))
        g = rmse(OOF[(f"{kind}{k}", "rand1")]).mean() - rmse(OOF[("canon", "canon")]).mean()
        print(f"  {kind}{k:<2d}: Δ vs canon-only {mean_d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {wins}/{len(d_rep)} → {dec} | 무작위 표기 RMSE − 정규화 RMSE(canon-only) {g:+.3f}")

print("\n=== (c) 추론 때 입력을 항상 정규화: canon-only 학습 + test 정규화 vs 증강 학습 + test 무작위 표기 ===")
d_rep, ci = paired(("rows5", "rand1"), ("canon", "canon"))
print(f"  Δ(RMSE_rows5@rand1 − RMSE_canon@canon) {d_rep.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] (양수 = (c)가 낫다)")
print("  정규화 입력을 우리가 통제할 수 있다면 (c)가 비용 1회로 해결한다. 통제할 수 없는 입력(외부 SMILES)에서만 증강이 필요하다.")

print("\n=== 대조 ===")
g_pos = rmse(OOF[("canon", "rand1")]).mean() - rmse(OOF[("canon", "canon")]).mean()
r0 = (rmse(OOF[("canon", "canon")])[0], rmse(OOF[("canon", "rand1")])[0])
print(f"양성 대조(문제가 실제로 있는가): canon-only 학습의 test 무작위 표기 − 정규화 RMSE 차이 {g_pos:+.3f} (노트북 05/01 §8은 반복 0에서 0.863 → 1.448; 여기 반복 0: {r0[0]:.3f} → {r0[1]:.3f})")
print(f"음성 대조: 라벨 섞기 RMSE {rmse(OOF[('canon-shuffled', 'canon')]).mean():.3f} ≈ y 표준편차 {y.std():.3f}")
d_ref = rmse(OOF[("canon", "canon")]).mean() - rmse(OOF[("ecfp4", "canon")]).mean()
print(f"참고: ECFP4(표기 무관) RMSE {rmse(OOF[('ecfp4', 'canon')]).mean():.3f} vs canon-only MLM RMSE {rmse(OOF[('canon', 'canon')]).mean():.3f} (차 {d_ref:+.3f})")
print(f"\n총 {time.time() - T0:.0f}초")
