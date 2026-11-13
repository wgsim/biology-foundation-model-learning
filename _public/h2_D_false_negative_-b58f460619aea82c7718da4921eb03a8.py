# 트랙 D 질문의 다른 절반 (모듈 05 과제 5): "거짓 음성을 줄이면(같은 그룹 쌍 마스킹 / 배치당 그룹당 한 쌍) 서열–텍스트 대조 정렬이 나아지는가?" — 템플릿 구조로
# 캐시된 동결 임베딩(ESM-2 35M 480차원, MiniLM 384차원)과 UniProt 표만 쓴다. Pfam 묶음 5-fold, seq→text 검색.
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_D_false_negative_masking.py` (캐시만 사용, 다운로드 없음)
import hashlib, importlib.metadata as md, json, re, subprocess, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path.cwd()
UP = ROOT / ".cache" / "uniprot"
torch.set_num_threads(8)

# ---------- 0. 사전 등록 (작성자는 노트북 02의 기본 설정 결과와 민감도 표를 이미 보았다) ----------
RUN_NAME = "h2_D_false_negative_masking"
SEEDS = [0, 1, 2]                                  # 학습 시드 3개 (노트북은 시드 0 하나)
VARIANTS = ["base", "mask", "sampler"]             # base: 기본 InfoNCE / mask: 같은 Pfam 그룹·같은 텍스트의 비대각 쌍을 로짓에서 제외 / sampler: 배치마다 그룹당 한 쌍만
PRIMARY = dict(split="pfam-group", direction="s2t", metric="R@1", variants=["mask", "sampler"], baseline="base")
MIN_EFFECT, MIN_WINS, BOOT = 0.01, 3, 500          # 시드 평균 R@1 개선 ≥ 0.01(절대), 시드 3개 모두 개선, Pfam 그룹 부트스트랩 구간 하한 > 0
EPOCHS, BS, LR, WD, TAU0, K = 40, 256, 1e-3, 1e-2, 0.07, 5     # 노트북 02의 사전 고정 설정 그대로


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


TSV = UP / "human_reviewed_len80_400.tsv.gz"
assert sha256(TSV)[:12] == "e3acc2dbebb9", "노트북 05/02와 다른 UniProt 파일"
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            sha256={"uniprot_tsv": sha256(TSV)[:16] + "…", "emb05_esm35_minilm.npz": sha256(UP / "emb05_esm35_minilm.npz")[:16] + "…"},
            packages={k: md.version(k) for k in ["numpy", "pandas", "torch", "scipy", "scikit-learn"]}, seeds=SEEDS, variants=VARIANTS, primary=PRIMARY)
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))

# ---------- 2. 데이터 (노트북 05/02 §1 · §3과 같은 규칙) ----------
df = pd.read_csv(TSV, sep="\t")
df.columns = ["acc", "name", "gene", "length", "seq", "func", "kw", "pfam", "interpro"]
df = df[~df.seq.duplicated(keep="first")].reset_index(drop=True)


def clean_function(t):
    t = re.sub(r"^FUNCTION:\s*", "", str(t))
    t = re.sub(r"\{ECO:[^}]*\}", "", t)
    t = re.sub(r"\(PubMed:[^)]*\)", "", t)
    t = re.sub(r"\(By similarity\)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"\s+([.,;])", r"\1", t)


df["text"] = df["func"].map(clean_function)
N = len(df)
z = np.load(UP / "emb05_esm35_minilm.npz")
E, T = z["E"], z["T"]
assert E.shape[0] == N and T.shape[0] == N and np.isfinite(E).all() and np.isfinite(T).all()
pf = df["pfam"].fillna("").str.strip(";").str.split(";")
pf_ids = {p: i for i, p in enumerate(sorted({p for x in pf for p in x if p}))}
rows_, cols_ = zip(*[(i, pf_ids[p]) for i, x in enumerate(pf) for p in set(x) if p])
M = sparse.csr_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(N, len(pf_ids)))
n_groups, GROUP = connected_components((M @ M.T) > 0, directed=False)
TEXT_ID = pd.factorize(df.text)[0]
gs = np.sort(np.bincount(GROUP))[::-1]
print(f"단백질 {N} | Pfam 그룹 {n_groups} (가장 큰 그룹 {gs[:3].tolist()}) | 서로 다른 텍스트 {TEXT_ID.max() + 1}개 (가장 흔한 텍스트 {np.sort(np.bincount(TEXT_ID))[::-1][:3].tolist()}회)")


def group_folds(seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(GROUP, return_inverse=True)
    sizes = np.bincount(inv)
    order = np.lexsort((rng.random(len(u)), -sizes))
    load, assign = np.zeros(K), np.zeros(len(u), int)
    for g in order:
        f = int(np.argmin(load))
        assign[g] = f
        load[f] += sizes[g]
    fo = assign[inv]
    return [(np.where(fo != f)[0], np.where(fo == f)[0]) for f in range(K)]


FOLDS = group_folds()
assert sorted(np.concatenate([te for _, te in FOLDS]).tolist()) == list(range(N))
assert all(len(set(GROUP[tr]) & set(GROUP[te])) == 0 for tr, te in FOLDS), "Pfam 그룹이 train·test에 걸친다"


def false_negative_stats(idx, bs=BS, n_batches=200, seed=0):
    """학습 배치 안의 비대각 쌍 중 같은 그룹 / 같은 텍스트인 쌍의 비율이다 (무작위 배치 기준). 거짓 음성이 얼마나 되는지 잰다."""
    rng = np.random.default_rng(seed)
    g_frac, t_frac = [], []
    for _ in range(n_batches):
        ix = rng.choice(idx, bs, replace=False)
        sg = GROUP[ix][:, None] == GROUP[ix][None]
        st = TEXT_ID[ix][:, None] == TEXT_ID[ix][None]
        off = ~np.eye(bs, dtype=bool)
        g_frac.append(sg[off].mean())
        t_frac.append(st[off].mean())
    return float(np.mean(g_frac)), float(np.mean(t_frac))


gf, tf = false_negative_stats(FOLDS[0][0])
print(f"무작위 배치({BS})의 비대각 쌍 중 같은 Pfam 그룹 {gf:.4f}, 같은 텍스트 {tf:.4f}  (배치당 음성 {BS - 1}개 중 약 {gf * (BS - 1):.1f} / {tf * (BS - 1):.1f}개가 거짓 음성)")


# ---------- 4. 모델·학습 (노트북 05/02 §4와 같은 head·InfoNCE, 마스크·샘플러만 추가) ----------
class Head(nn.Module):
    def __init__(self, din, d=256, hid=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, d))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def info_nce(a, b, logit_scale, ignore=None):
    logits = logit_scale.exp().clamp(max=100) * a @ b.T
    if ignore is not None:
        logits = logits.masked_fill(ignore, -1e4)              # 대각은 ignore가 아니므로 정답 쌍은 항상 남는다
    y = torch.arange(len(a))
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2


def zscore(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def train_clip(Xs, Xt, tr, seed, variant, epochs=EPOCHS, tau0=TAU0, learn_tau=True):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hs, ht = Head(Xs.shape[1]), Head(Xt.shape[1])
    ls = nn.Parameter(torch.tensor(float(np.log(1 / tau0))), requires_grad=learn_tau)
    params = list(hs.parameters()) + list(ht.parameters()) + ([ls] if learn_tau else [])
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    Xs_t, Xt_t = torch.tensor(Xs), torch.tensor(Xt)
    n = len(Xs)
    g_tr, t_tr = GROUP[tr], TEXT_ID[tr]
    if variant == "sampler":
        ug, inv = np.unique(g_tr, return_inverse=True)
        members = [np.where(inv == k)[0] for k in range(len(ug))]
        p = np.array([len(m) for m in members], float)
        p /= p.sum()
        assert len(ug) >= BS, "그룹 수가 배치 크기보다 작다"
    n_steps = n // BS
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(n_steps):
            if variant == "sampler":
                gsel = rng.choice(len(ug), BS, replace=False, p=p)
                ix = torch.tensor([members[k][rng.integers(len(members[k]))] for k in gsel])
                ign = None
                # 같은 텍스트가 다른 그룹에서 뽑히는 경우는 남는다 (그룹 기준 샘플러이므로)
            else:
                ix = perm[s * BS:(s + 1) * BS]
                ign = None
                if variant == "mask":
                    gg, tt = g_tr[ix.numpy()], t_tr[ix.numpy()]
                    same = (gg[:, None] == gg[None]) | (tt[:, None] == tt[None])
                    ign = torch.tensor(same & ~np.eye(BS, dtype=bool))
            loss = info_nce(hs(Xs_t[ix]), ht(Xt_t[ix]), ls, ign)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        losses.append(tot / n_steps)
    return hs, ht, ls, losses


def ranks_of(S):
    """정답보다 점수가 높거나 같은 다른 후보의 수 (노트북 05/02의 동점 규칙: 반올림 + 동점은 정답에 불리)."""
    S = np.round(S, 6)
    return (S >= np.diag(S)[:, None]).sum(1) - 1


@torch.no_grad()
def scores(hs, ht, Xs_te, Xt_te):
    return hs(torch.tensor(Xs_te)).numpy() @ ht(torch.tensor(Xt_te)).numpy().T


def run_variant(variant, seed, shuffle_pairs=False, **kw):
    rk = np.zeros(N, int)
    gal = np.zeros(N, int)
    final_tau, final_loss = [], []
    for tr, te in FOLDS:
        Xtr, Xte = zscore(E[tr], E[te])
        Ttr = T[tr]
        if shuffle_pairs:
            Ttr = Ttr[np.random.default_rng(0).permutation(len(tr))]
        hs, ht, ls, losses = train_clip(Xtr, Ttr, tr, seed, variant, **kw)
        rk[te] = ranks_of(scores(hs, ht, Xte, T[te]))
        gal[te] = len(te)
        final_tau.append(float(1 / ls.exp().clamp(max=100)))
        final_loss.append(losses[-1])
    return rk, gal, float(np.mean(final_tau)), float(np.mean(final_loss)), losses


# ---------- 5. 평가 ----------
RK, GAL, TAU, LOSS, CURVE = {}, None, {}, {}, {}
t = time.time()
for v in VARIANTS:
    for sd in SEEDS:
        rk, gal, tau, loss, curve = run_variant(v, sd)
        RK[(v, sd)], GAL, TAU[(v, sd)], LOSS[(v, sd)] = rk, gal, tau, loss
        if sd == SEEDS[0]:
            CURVE[v] = curve                       # 마지막 fold의 손실 곡선
    print(f"  {v:8s} 완료 (누적 {time.time() - t:.0f}s)", flush=True)
for sd in SEEDS:                                    # 음성 대조: 짝을 섞어 학습 (우연 수준이어야 한다)
    RK[("shuffled", sd)] = run_variant("base", sd, shuffle_pairs=True)[0]

rng_b = np.random.default_rng(1)
u_g, inv_g = np.unique(GROUP, return_inverse=True)
BW = np.stack([np.bincount(rng_b.integers(0, len(u_g), len(u_g)), minlength=len(u_g))[inv_g] for _ in range(BOOT)]).astype(float)


def hit(v, sd, k):
    return (RK[(v, sd)] < k).astype(float)


print(f"\n--- seq→text Recall (Pfam 묶음 5-fold, 시드 {SEEDS}; 갤러리 ≈ {GAL.mean():.0f}; 우연 R@1 ≈ {1 / GAL.mean():.4f}) ---")
tab = []
for v in VARIANTS + ["shuffled"]:
    r1 = np.array([hit(v, sd, 1).mean() for sd in SEEDS])
    r10 = np.array([hit(v, sd, 10).mean() for sd in SEEDS])
    med = np.array([np.median(RK[(v, sd)] / (GAL - 1)) for sd in SEEDS])
    bs = (BW @ np.mean([hit(v, sd, 1) for sd in SEEDS], axis=0)) / BW.sum(1)        # 시드 평균 hit의 그룹 부트스트랩
    lo, hi = np.percentile(bs, [2.5, 97.5])
    row = {"변형": v, "R@1 (시드 평균±sd) [그룹 부트스트랩]": f"{r1.mean():.3f}±{r1.std():.3f} [{lo:.3f},{hi:.3f}]", "R@10": f"{r10.mean():.3f}±{r10.std():.3f}", "중앙 순위(비율)": f"{med.mean():.3f}"}
    if v in VARIANTS:
        row["최종 τ"] = f"{np.mean([TAU[(v, sd)] for sd in SEEDS]):.3f}"
        row["최종 학습 손실"] = f"{np.mean([LOSS[(v, sd)] for sd in SEEDS]):.3f}"
    tab.append(row)
print(pd.DataFrame(tab).set_index("변형").fillna("").to_string())
print("손실 곡선 (마지막 fold, 시드 0, 에폭 1/10/20/30/40):")
for v in VARIANTS:
    c = CURVE[v]
    print(f"  {v:8s} " + " ".join(f"{c[i]:.3f}" for i in (0, 9, 19, 29, 39)) + "   (손실의 절대값은 마스크로 음성 수가 줄어 변형 사이에 비교할 수 없다)")


def paired(v1, v0, k=1):
    d_seed = np.array([hit(v1, sd, k).mean() - hit(v0, sd, k).mean() for sd in SEEDS])
    bs = (BW @ (np.mean([hit(v1, sd, k) for sd in SEEDS], axis=0) - np.mean([hit(v0, sd, k) for sd in SEEDS], axis=0))) / BW.sum(1)
    return d_seed, np.percentile(bs, [2.5, 97.5])


print("\n=== 주 비교: Pfam 묶음 seq→text R@1, 변형 − 기본 InfoNCE ===")
for v in ["mask", "sampler"]:
    d_seed, ci = paired(v, "base")
    dec, mean_d, wins = decide(d_seed, tuple(ci))
    r10 = paired(v, "base", 10)[0].mean()
    print(f"  {v:8s} Δ R@1 {mean_d:+.4f} 95%[{ci[0]:+.4f},{ci[1]:+.4f}] | 시드별 Δ {np.round(d_seed, 4).tolist()} | Δ>0 {wins}/{len(d_seed)} → {dec} | Δ R@10 {r10:+.4f}")

print("\n=== 대조 ===")
d_seed, ci = paired("base", "shuffled")
dec, mean_d, wins = decide(d_seed, tuple(ci))
print(f"양성 대조(정렬 학습 vs 짝 섞기): Δ R@1 {mean_d:+.4f} 95%[{ci[0]:+.4f},{ci[1]:+.4f}], Δ>0 {wins}/{len(d_seed)} → {dec}")
print(f"음성 대조: 짝 섞어 학습한 R@1 {np.mean([hit('shuffled', sd, 1).mean() for sd in SEEDS]):.4f} (우연 {1 / GAL.mean():.4f})")

# ---------- 6. 민감도: 기본 InfoNCE의 사전 고정 설정이 결과를 흔드는 폭 (노트북 §7의 6설정, 시드 0) ----------
print("\n=== 민감도: 기본 InfoNCE의 설정 변경 (시드 0) — 변형의 Δ가 이 변동폭 안인지 보기 위한 것 ===")
configs = {"기본(τ0=0.07, 40ep)": {}, "τ0=0.03": dict(tau0=0.03), "τ0=0.2": dict(tau0=0.2), "10 에폭": dict(epochs=10), "100 에폭": dict(epochs=100), "온도 고정": dict(learn_tau=False)}
srows = []
for name, kw in configs.items():
    rk = run_variant("base", 0, **kw)[0]
    srows.append({"설정": name, "R@1": (rk < 1).mean(), "R@10": (rk < 10).mean()})
S_ = pd.DataFrame(srows)
print(S_.round(3).to_string(index=False))
print(f"민감도 R@1 범위: {S_['R@1'].min():.3f} ~ {S_['R@1'].max():.3f} (폭 {S_['R@1'].max() - S_['R@1'].min():.3f}); 시드 사이 sd(기본): {np.std([hit('base', sd, 1).mean() for sd in SEEDS]):.4f}")
print(f"\n총 {time.time() - T0:.0f}초")
