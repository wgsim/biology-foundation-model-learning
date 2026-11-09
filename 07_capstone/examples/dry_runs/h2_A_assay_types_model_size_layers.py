# 트랙 A 질문의 다른 절반: (②) 다른 assay 유형(활성·발현·결합·적합도·안정성)에서도 probe가 위치 baseline을 이기는가?
# (③) 모델 크기(ESM-2 8M/35M/150M/650M)와 층 선택은 그 결과를 얼마나 바꾸는가?  — 템플릿 노트북의 절차를 여러 assay로 옮긴 것
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_A_assay_types_model_size_layers.py` (캐시만 사용, 다운로드 없음)
import hashlib, importlib.metadata as md, json, subprocess, time, warnings, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import substitution_matrices
from huggingface_hub import snapshot_download
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoTokenizer, EsmModel
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
T0 = time.time()
ROOT = Path.cwd()
PG = ROOT / ".cache" / "proteingym"

# ---------- 0. 사전 등록 (작성자는 템플릿 노트북에서 DN7A 한 assay의 결과를 이미 보았다) ----------
RUN_NAME = "h2_A_assay_types_model_size_layers"
MODELS = {   # (저장소, 리비전, 층 수)
    "8M": ("facebook/esm2_t6_8M_UR50D", "c731040fcd8d73dceaa04b0a8e6329b345b0f5df", 6),
    "35M": ("facebook/esm2_t12_35M_UR50D", "6fbf070e65b0b7291e7bbcd451118c216cff79d8", 12),
    "150M": ("facebook/esm2_t30_150M_UR50D", "a695f6045e2e32885fa60af20c13cb35398ce30c", 30),
    "650M": ("facebook/esm2_t33_650M_UR50D", "08e4846e537177426273712802403f7ba8261b6c", 33),
}
SEEDS = [0, 1, 2, 3, 4]
SELECT = dict(len_min=100, len_max=250, n_min=500, n_max=4000, per_type=3, order="DMS_id 알파벳순 앞에서부터")   # 결과를 보기 전에 고정한 assay 선택 규칙
TYPES = ["Activity", "Binding", "Expression", "OrganismalFitness", "Stability"]
SPLIT = "position-block"                         # 주 분할 (새 위치 일반화). random은 보조
N_BLOCKS = 20                                    # 위치 블록 수 ≈ 20 → 블록 크기 = ceil(길이 / 20)
PRIMARY_A = dict(model="8M", method="probe", baseline="B2", split=SPLIT)
MIN_EFFECT = 0.02                                # Δ Spearman
MIN_WIN_FRAC = 2 / 3                             # 유형별·전체에서 Δ > 0인 assay 비율
BOOT = 1000                                      # assay 단위 부트스트랩
SIZE_STEP = 0.02                                 # '크기가 커지면 Δ가 오른다': 650M의 Δ − 8M의 Δ ≥ 0.02인 assay ≥ 2/3


def decide(deltas, ci, min_frac=MIN_WIN_FRAC):
    deltas = np.asarray(deltas, float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < 2:
        return "판정 불가(assay 부족)", float("nan"), 0
    mean_d, wins = float(np.mean(deltas)), int(np.sum(deltas > 0))
    ok = mean_d >= MIN_EFFECT and wins >= np.ceil(min_frac * len(deltas)) and (ci is None or ci[0] > 0)
    return ("지지" if ok else "지지되지 않음"), mean_d, wins


# ---------- 1. 출처 · assay 선택 ----------
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


ZIP, REF = PG / "DMS_ProteinGym_substitutions.zip", PG / "DMS_substitutions.csv"
ref = pd.read_csv(REF)
cand = ref[(ref.seq_len >= SELECT["len_min"]) & (ref.seq_len <= SELECT["len_max"]) & (ref.DMS_number_multiple_mutants == 0)
           & (ref.DMS_total_number_mutants >= SELECT["n_min"]) & (ref.DMS_total_number_mutants <= SELECT["n_max"])]
ASSAYS = []
for ty in TYPES:
    sub = cand[cand.coarse_selection_type == ty]
    ASSAYS += sub.sort_values("DMS_id").head(SELECT["per_type"]).DMS_id.tolist()
sel = ref.set_index("DMS_id").loc[ASSAYS]
print("선택된 assay:\n" + sel[["coarse_selection_type", "seq_len", "DMS_total_number_mutants", "selection_assay"]].to_string())
LIC = "ProteinGym 저장소 코드 MIT; 개별 DMS 데이터의 라이선스는 assay별 원 출처 — 미확인"


def load_assay(dms_id):
    row = ref[ref.DMS_id == dms_id].iloc[0]
    with zipfile.ZipFile(ZIP) as z:
        df = pd.read_csv(z.open(f"DMS_ProteinGym_substitutions/{row.DMS_filename}"))
    assert not df.mutant.str.contains(":").any()
    df["wt"], df["pos"], df["mt"] = df.mutant.str[0], df.mutant.str[1:-1].astype(int), df.mutant.str[-1]
    assert all(row.target_seq[p - 1] == w for p, w in zip(df.pos, df.wt)), "wt 잔기가 target_seq와 다름"
    assert df.mutant.is_unique and df.mutated_sequence.str.len().nunique() == 1 and df.DMS_score.notna().all()
    return row.target_seq, df.reset_index(drop=True), row


OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
mdirs = {k: Path(snapshot_download(v[0], revision=v[1], allow_patterns=["model.safetensors", "*.json", "vocab.txt"])) for k, v in MODELS.items()}
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            sha256={"zip": sha256(ZIP)[:16] + "…", "ref": sha256(REF)[:16] + "…"}, device=str(DEVICE), assays=ASSAYS, select=SELECT,
            models={k: (v[0], mdirs[k].name) for k, v in MODELS.items()}, licenses=LIC,
            packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers"]})
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))

AA = "ACDEFGHIKLMNPQRSTVWY"
BLOSUM = substitution_matrices.load("BLOSUM62")


def onehot(chars):
    M = np.zeros((len(chars), len(AA)))
    M[np.arange(len(chars)), [AA.index(c) for c in chars]] = 1
    return M


DATA = {}
for a in ASSAYS:
    WT, DF, ROW = load_assay(a)
    blk = int(np.ceil(len(WT) / N_BLOCKS))
    blos = np.array([BLOSUM[w][m] for w, m in zip(DF.wt, DF.mt)], dtype=float)[:, None]
    B1 = np.hstack([onehot(DF.wt), onehot(DF.mt), blos])
    P = np.zeros((len(DF), len(WT)))
    P[np.arange(len(DF)), DF.pos.values - 1] = 1
    DATA[a] = dict(WT=WT, DF=DF, Y=DF.DMS_score.values.astype(float), B1=B1, B2=np.hstack([B1, P]), block=blk, type=ROW.coarse_selection_type)
    print(f"  {a[:42]:42s} {ROW.coarse_selection_type:17s} 길이 {len(WT)} 변이 {len(DF)} 위치 {DF.pos.nunique()} 블록크기 {blk} | 점수 sd {DATA[a]['Y'].std():.2f}")


# ---------- 2. 분할 (템플릿 make_split과 같은 규칙) ----------
def make_split(a, kind, seed):
    d = DATA[a]
    n = len(d["DF"])
    if kind == "random":
        tr, te = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(np.arange(n)))
        groups = np.arange(n)
    else:
        groups = (d["DF"].pos.values - 1) // d["block"]
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(np.arange(n), groups=groups))
    return tr, te, groups


def leak(a, tr, te):
    pos = DATA[a]["DF"].pos.values
    return float(np.isin(pos[te], pos[tr]).mean())


# ---------- 3. 특징 ----------
@torch.no_grad()
def site_features(a, model, layers, L, tok, batch=64):
    """변이 위치의 hidden state를 요청한 층에서 모아 {층: [n, d]}로 돌려준다. 층 L은 last_hidden_state(마지막 층 정규화 뒤)."""
    d = DATA[a]
    ids_all = tok(d["DF"].mutated_sequence.tolist(), return_tensors="pt")["input_ids"]
    pos_all = torch.tensor(d["DF"].pos.values)
    model = model.eval().to(DEVICE)
    out = {l: [] for l in layers}
    for s in range(0, len(ids_all), batch):
        ids = ids_all[s:s + batch].to(DEVICE)
        ix = torch.arange(len(ids))
        o = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        for l in layers:
            h = o.last_hidden_state if l == L else o.hidden_states[l]
            out[l].append(h[ix, pos_all[s:s + batch].to(DEVICE)].float().cpu())
    return {l: torch.cat(v).numpy() for l, v in out.items()}


TOK = {k: AutoTokenizer.from_pretrained(v) for k, v in mdirs.items()}


def rho(y, p):
    return float(spearmanr(y, p).statistic)


def rho_cv(y, p):
    r = spearmanr(y, p).statistic
    return -1.0 if np.isnan(r) else float(r)


def fit_predict(X, y, groups, tr, te):
    pipe = make_pipeline(StandardScaler(), Ridge())
    n_cv = min(3, len(np.unique(groups[tr])))
    gs = GridSearchCV(pipe, {"ridge__alpha": np.logspace(-1, 5, 13)}, scoring=make_scorer(rho_cv), cv=GroupKFold(n_splits=n_cv).split(X[tr], y[tr], groups[tr]))
    gs.fit(X[tr], y[tr])
    return gs.predict(X[te]), float(gs.best_score_)


def nearest_position_mean(a, tr, te):
    d = DATA[a]
    pos, Y = d["DF"].pos.values, d["Y"]
    pos_tr = np.unique(pos[tr])
    mean_by_pos = {p: Y[tr][pos[tr] == p].mean() for p in pos_tr}
    out = []
    for p in pos[te]:
        dist = np.abs(pos_tr - p)
        out.append(np.mean([mean_by_pos[q] for q in pos_tr[dist == dist.min()]]))
    return np.array(out)


# 임베딩 계산 (assay × 모델). ② 는 8M 마지막 층만, ③ 은 5개 assay(유형별 첫 번째) × 4개 모델 × 4개 층
SUB5 = [next(a for a in ASSAYS if DATA[a]["type"] == ty) for ty in TYPES if any(DATA[a]["type"] == ty for a in ASSAYS)]
EMB = {}        # (assay, model, layer) -> [n, d]
EMB_R = {}      # (assay, seed) -> [n, d]   (8M 무작위 초기화, 시드마다 새로 만듦)
t = time.time()
for model_key, (repo, rev, L) in MODELS.items():
    assays = ASSAYS if model_key == "8M" else SUB5
    layers = sorted({max(1, round(f * L)) for f in (0.25, 0.5, 0.75)} | {L})
    m = EsmModel.from_pretrained(mdirs[model_key], add_pooling_layer=False)
    for a in assays:
        feats = site_features(a, m, layers if (model_key != "8M" or a in SUB5) else [L], L, TOK[model_key], batch=64 if L <= 12 else 16)
        for l, X in feats.items():
            EMB[(a, model_key, l)] = X
    del m
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    print(f"  {model_key}: 층 {layers} 임베딩 {len(assays)}개 assay ({time.time() - t:.0f}s)", flush=True)
for a in ASSAYS:
    for s in SEEDS:
        torch.manual_seed(s)
        EMB_R[(a, s)] = site_features(a, EsmModel(AutoConfig.from_pretrained(mdirs["8M"]), add_pooling_layer=False), [6], 6, TOK["8M"])[6]
print(f"무작위 초기화 임베딩 완료 ({time.time() - t:.0f}s)")

# ---------- 5. 평가 ① : 모든 assay, 8M 마지막 층 ----------
RES = []      # dict(assay, split, seed, method, rho)
PRED = {}
t = time.time()
for a in ASSAYS:
    d = DATA[a]
    Y = d["Y"]
    for kind in ["position-block", "random"]:
        for seed in SEEDS:
            tr, te, groups = make_split(a, kind, seed)
            preds = {"B0": nearest_position_mean(a, tr, te), "B1": fit_predict(d["B1"], Y, groups, tr, te)[0], "B2": fit_predict(d["B2"], Y, groups, tr, te)[0],
                     "probe": fit_predict(EMB[(a, "8M", 6)], Y, groups, tr, te)[0], "probe-randinit": fit_predict(EMB_R[(a, seed)], Y, groups, tr, te)[0]}
            ysh = Y.copy()
            ysh[tr] = np.random.default_rng(seed).permutation(Y[tr])
            preds["probe-shuffled"] = fit_predict(EMB[(a, "8M", 6)], ysh, groups, tr, te)[0]
            for mth, p in preds.items():
                RES.append(dict(assay=a, split=kind, seed=seed, method=mth, rho=rho(Y[te], p), leak=leak(a, tr, te)))
    print(f"  {a[:40]} 완료 ({time.time() - t:.0f}s)", flush=True)
R = pd.DataFrame(RES)
print("\n--- 누출 점검: test 위치가 train에 있는 비율 (분할별 평균) ---")
print(R.groupby(["split"]).leak.mean().round(3).to_string())
print("\n--- assay별 test Spearman (5시드 평균) : position-block ---")
PB = R[R.split == SPLIT].groupby(["assay", "method"]).rho.mean().unstack()[["B0", "B1", "B2", "probe", "probe-randinit", "probe-shuffled"]]
PB.insert(0, "유형", [DATA[a]["type"] for a in PB.index])
print(PB.round(3).to_string())


def assay_delta(split, m1, m2):
    p = R[R.split == split].pivot_table(index=["assay", "seed"], columns="method", values="rho")
    d = (p[m1] - p[m2]).groupby(level=0).mean()
    return d


def boot_assays(d, B=BOOT, seed=0):
    rs = np.random.RandomState(seed)
    v = d.values
    return np.percentile([np.mean(v[rs.randint(0, len(v), len(v))]) for _ in range(B)], [2.5, 97.5])


print("\n=== ② 주 비교: probe(8M) − B2, position-block, assay 단위 (Δ = 5시드 평균) ===")
d = assay_delta(SPLIT, "probe", "B2")
ci = boot_assays(d)
dec, mean_d, wins = decide(d.values, tuple(ci))
print(f"전체 {len(d)}개 assay: Δ 평균 {mean_d:+.3f} 95%[{ci[0]:+.3f},{ci[1]:+.3f}] (assay 부트스트랩), Δ>0 {wins}/{len(d)} → {dec}")
for ty in TYPES:
    ids = [a for a in d.index if DATA[a]["type"] == ty]
    if not ids:
        continue
    dec_t, mean_t, wins_t = decide(d[ids].values, None)
    print(f"  {ty:17s} assay {len(ids)}개: Δ 평균 {mean_t:+.3f}, Δ>0 {wins_t}/{len(ids)} (assay별 {', '.join(f'{x:+.2f}' for x in d[ids].values)}) → {dec_t}   [구간 계산 안 함: assay 수 부족]")
print("\n보조 분할 random (같은 위치가 train에 있음):")
d_r = assay_delta("random", "probe", "B2")
print(f"  전체 Δ(probe − B2) 평균 {d_r.mean():+.3f} 95%[{boot_assays(d_r)[0]:+.3f},{boot_assays(d_r)[1]:+.3f}], Δ>0 {(d_r > 0).sum()}/{len(d_r)}")
print("\n비교 (position-block, assay 평균 Δ): " + " | ".join(f"{m1}−{m2} {assay_delta(SPLIT, m1, m2).mean():+.3f}" for m1, m2 in [("probe", "B0"), ("probe", "B1"), ("probe", "probe-randinit"), ("B2", "B0")]))

print("\n=== 대조 ===")
dpre = assay_delta(SPLIT, "probe", "probe-randinit")
ci_p = boot_assays(dpre)
dec_p, mean_p, wins_p = decide(dpre.values, tuple(ci_p))
print(f"양성 대조 (사전학습 효과 probe − randinit): Δ 평균 {mean_p:+.3f} 95%[{ci_p[0]:+.3f},{ci_p[1]:+.3f}] Δ>0 {wins_p}/{len(dpre)} → {dec_p}")
print(f"음성 대조 (훈련 라벨 섞기) test Spearman 평균 {R[(R.split == SPLIT) & (R.method == 'probe-shuffled')].rho.mean():+.3f} (0 근처여야 함)")

# ---------- 6. 평가 ③ : 모델 크기 × 층 (5개 assay, position-block) ----------
print("\n=== ③ 모델 크기 × 층: position-block test Spearman (5시드 평균), assay: " + ", ".join(a.split('_')[0] for a in SUB5) + " ===")
RES3 = []
t = time.time()
for a in SUB5:
    d_ = DATA[a]
    Y = d_["Y"]
    for seed in SEEDS:
        tr, te, groups = make_split(a, SPLIT, seed)
        pb2 = fit_predict(d_["B2"], Y, groups, tr, te)[0]
        RES3.append(dict(assay=a, seed=seed, model="-", layer=0, sel="B2", rho=rho(Y[te], pb2)))
        for mk, (_, _, L) in MODELS.items():
            lays = sorted(l for (aa, m_, l) in EMB if aa == a and m_ == mk)
            fits = {l: fit_predict(EMB[(a, mk, l)], Y, groups, tr, te) for l in lays}
            for l in lays:
                RES3.append(dict(assay=a, seed=seed, model=mk, layer=l, sel="fixed", rho=rho(Y[te], fits[l][0])))
            bl = max(lays, key=lambda l: fits[l][1])                               # train 안의 그룹 CV로 층 선택
            RES3.append(dict(assay=a, seed=seed, model=mk, layer=bl, sel="cv-selected", rho=rho(Y[te], fits[bl][0])))
            orc = max(lays, key=lambda l: rho(Y[te], fits[l][0]))                  # 진단용 오라클: test를 보고 고른 층(선택에 쓰면 안 됨)
            RES3.append(dict(assay=a, seed=seed, model=mk, layer=orc, sel="oracle", rho=rho(Y[te], fits[orc][0])))
    print(f"  {a[:40]} ({time.time() - t:.0f}s)", flush=True)
R3 = pd.DataFrame(RES3)
b2 = R3[R3.sel == "B2"].groupby("assay").rho.mean()
print("B2 baseline (assay 평균): " + ", ".join(f"{a.split('_')[0]} {b2[a]:.3f}" for a in SUB5))
for sel in ["fixed", "cv-selected", "oracle"]:
    sub = R3[R3.sel == sel]
    if sel == "fixed":
        tab = sub.groupby(["model", "layer"]).rho.mean().unstack(0)
        print("\n층을 고정했을 때 probe Spearman (assay·시드 평균) — 행 = 층(모델마다 다름):")
        for mk, (_, _, L) in MODELS.items():
            row = sub[sub.model == mk].groupby("layer").rho.mean()
            print(f"  {mk:5s} " + "  ".join(f"층{l}: {v:.3f}" for l, v in row.items()))
    else:
        print(f"\n{sel} 층: probe Spearman " + " | ".join(f"{mk} {sub[sub.model == mk].rho.mean():.3f}" for mk in MODELS))

print("\nΔ(probe − B2) by 모델 크기, 층 선택 방식 (assay 평균; assay별 Δ>0 수):")
D3 = {}
for sel, labl in [("fixed-last", "마지막 층 고정"), ("cv-selected", "train CV로 선택한 층"), ("oracle", "test 오라클(진단용)")]:
    row = []
    for mk, (_, _, L) in MODELS.items():
        if sel == "fixed-last":
            sub = R3[(R3.sel == "fixed") & (R3.model == mk) & (R3.layer == L)]
        else:
            sub = R3[(R3.sel == sel) & (R3.model == mk)]
        per_assay = sub.groupby("assay").rho.mean() - b2
        D3[(sel, mk)] = per_assay
        row.append(f"{mk} {per_assay.mean():+.3f} ({int((per_assay > 0).sum())}/{len(per_assay)})")
    print(f"  {labl:20s}: " + " | ".join(row))

print("\n=== ③ 주 질문: 모델 크기가 커지면 Δ(probe − B2)가 오르는가? (마지막 층 고정, 650M − 8M, assay 단위) ===")
step = D3[("fixed-last", "650M")] - D3[("fixed-last", "8M")]
mean_s = float(step.mean())
wins_s = int((step >= SIZE_STEP).sum())
verdict_s = "지지" if (mean_s >= SIZE_STEP and wins_s >= np.ceil(MIN_WIN_FRAC * len(step))) else "지지되지 않음"
print(f"  assay별 Δ650M − Δ8M: {', '.join(f'{a.split(chr(95))[0]} {x:+.3f}' for a, x in step.items())} | 평균 {mean_s:+.3f}, ≥{SIZE_STEP}인 assay {wins_s}/{len(step)} → {verdict_s}")
sel_gap = (D3[("oracle", "8M")] - D3[("cv-selected", "8M")]).mean(), (D3[("oracle", "650M")] - D3[("cv-selected", "650M")]).mean()
print(f"  층 선택의 낙관성(오라클 − CV 선택) 평균: 8M {sel_gap[0]:+.3f}, 650M {sel_gap[1]:+.3f}")
print(f"\n총 {time.time() - T0:.0f}초")
