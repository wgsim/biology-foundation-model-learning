# 트랙 A 질문의 다른 절반 (모듈 06 과제 2): "검증 기반 선택(조기 종료·학습률)을 하면 probe vs 파인튜닝의 순위가 바뀌는가?" — 템플릿 구조로
# 모듈 06/01의 설정(ESM-2 8M, assay 3종, position-block 분할, 시드 0~2) 위에서 (고정 설정) vs (중첩 검증으로 학습률·에폭 선택)을 같은 시드에서 짝지어 비교한다.
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/h2_A_validation_selected_finetuning.py` (캐시만 사용, 다운로드 없음, 약 1시간)
import gc, hashlib, importlib.metadata as md, json, subprocess, time, warnings, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio.Align import substitution_matrices
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, EsmModel
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
T0 = time.time()
ROOT = Path.cwd()
PG = ROOT / ".cache" / "proteingym"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ---------- 0. 사전 등록 (작성자는 노트북 06/01의 고정 설정 결과와 민감도 표를 이미 보았다) ----------
RUN_NAME = "h2_A_validation_selected_finetuning"
MODEL = ("facebook/esm2_t6_8M_UR50D", "c731040fcd8d73dceaa04b0a8e6329b345b0f5df")
ASSAYS = ["DN7A_SACS2_Tsuboyama_2023_1JIC", "IF1_ECOLI_Kelsic_2016", "A0A247D711_LISMN_Stadelmann_2021"]
SEEDS = [0, 1, 2]
SPLIT = "position-block"                      # 주 분할(새 위치 일반화). random은 이 실행에서 하지 않았다
FIXED = dict(epochs=5, lr_head=1e-3, lr_lora=1e-3, lr_full=5e-5)             # 노트북 06/01의 사전 고정 설정
GRID = {"lora": [3e-4, 1e-3, 3e-3], "full": [1e-5, 5e-5, 2e-4]}               # 선택할 학습률 (노트북 §6 민감도와 같은 값)
E_MAX = 8                                    # 에폭 상한: 이 안에서 검증 Spearman이 최대인 에폭에서 멈춘다(조기 종료)
VAL_FRAC = 0.2                               # 바깥 train 안에서 검증으로 뗄 비율 (바깥 분할과 같은 방식: 위치 블록 단위)
TIE_BAND = 0.02                              # 두 방법의 평균 Spearman 차이가 이 안이면 '비김'
PRIMARY = dict(split=SPLIT, comparisons=["LoRA − probe", "전체 − probe"], rule="assay 평균 Δ를 승(≥+0.02)/비김/패(≤−0.02)로 분류하고, 고정과 선택 사이에 분류가 달라지는 비교가 하나라도 있으면 '순위가 바뀜'")


def rho(y, p):
    r = spearmanr(y, p).statistic
    return -1.0 if np.isnan(r) else float(r)


# ---------- 1. 출처 ----------
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


ZIP, REF = PG / "DMS_ProteinGym_substitutions.zip", PG / "DMS_substitutions.csv"
mdir = Path(snapshot_download(MODEL[0], revision=MODEL[1], allow_patterns=["model.safetensors", "*.json", "vocab.txt"]))
OUT = ROOT / ".cache" / "capstone_runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
PROV = dict(run=RUN_NAME, repo_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(), device=str(DEVICE),
            sha256={"zip": sha256(ZIP)[:16] + "…", "ref": sha256(REF)[:16] + "…", "model.safetensors": sha256(mdir / "model.safetensors")[:16] + "…"},
            model=dict(repo=MODEL[0], revision=MODEL[1]), assays=ASSAYS, seeds=SEEDS, fixed=FIXED, grid=GRID, e_max=E_MAX, val_frac=VAL_FRAC, primary=PRIMARY,
            packages={k: md.version(k) for k in ["numpy", "pandas", "scikit-learn", "torch", "transformers", "peft"]})
(OUT / "provenance.json").write_text(json.dumps(PROV, ensure_ascii=False, indent=1))
print("device:", DEVICE)

# ---------- 2. 데이터 (노트북 06/01 §1과 같은 규칙) ----------
ref = pd.read_csv(REF)
TOK = AutoTokenizer.from_pretrained(mdir)
AA = "ACDEFGHIKLMNPQRSTVWY"
BLOSUM = substitution_matrices.load("BLOSUM62")
DATA = {}
for a in ASSAYS:
    row = ref[ref.DMS_id == a].iloc[0]
    with zipfile.ZipFile(ZIP) as z:
        df = pd.read_csv(z.open(f"DMS_ProteinGym_substitutions/{row.DMS_filename}"))
    df = df[~df.mutant.str.contains(":")].copy()
    df["wt"], df["pos"], df["mt"] = df.mutant.str[0], df.mutant.str[1:-1].astype(int), df.mutant.str[-1]
    assert all(row.target_seq[p - 1] == w for p, w in zip(df.pos, df.wt)) and df.mutated_sequence.str.len().nunique() == 1
    df = df.reset_index(drop=True)
    ids = TOK(df.mutated_sequence.tolist(), return_tensors="pt", padding=False)["input_ids"]
    DATA[a] = dict(df=df, ids=ids, pos=torch.tensor(df.pos.values), y=df.DMS_score.values.astype(np.float32), L=len(row.target_seq))
    print(f"  {a:36s} L={len(row.target_seq)} 변이 {len(df)} 위치 {df.pos.nunique()}")


def onehot(chars):
    M = np.zeros((len(chars), len(AA)))
    M[np.arange(len(chars)), [AA.index(c) for c in chars]] = 1
    return M


def make_split(df, seed):
    groups = (df.pos.values - 1) // 10
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(df, groups=groups))
    return tr, te, groups


def ridge_eval(X, y, groups, tr, te):
    pipe = make_pipeline(StandardScaler(), Ridge())
    gs = GridSearchCV(pipe, {"ridge__alpha": np.logspace(-1, 5, 13)}, scoring=make_scorer(rho), cv=GroupKFold(n_splits=3).split(X[tr], y[tr], groups[tr]))
    gs.fit(X[tr], y[tr])
    return rho(y[te], gs.predict(X[te]))


@torch.no_grad()
def site_features(key, batch=128):
    d = DATA[key]
    model = EsmModel.from_pretrained(mdir, add_pooling_layer=False).eval().to(DEVICE)
    out = []
    for s in range(0, len(d["ids"]), batch):
        ids = d["ids"][s:s + batch].to(DEVICE)
        h = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        out.append(h[torch.arange(len(ids)), d["pos"][s:s + batch].to(DEVICE)].float().cpu())
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return torch.cat(out).numpy()


FEATS = {a: site_features(a) for a in ASSAYS}


# ---------- 4. 모델·학습 (노트북 06/01 §3과 같은 구조; 에폭마다 평가 가능한 변형 추가) ----------
class SiteRegressor(nn.Module):
    def __init__(self, backbone, d):
        super().__init__()
        self.backbone, self.drop, self.head = backbone, nn.Dropout(0.1), nn.Linear(d, 1)

    def forward(self, ids, pos):
        h = self.backbone(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        return self.head(self.drop(h[torch.arange(len(ids), device=ids.device), pos])).squeeze(-1)


def build(mode):
    backbone = EsmModel.from_pretrained(mdir, add_pooling_layer=False)
    if mode == "lora":
        backbone = get_peft_model(backbone, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query", "value"], bias="none"))
        return SiteRegressor(backbone, backbone.base_model.model.config.hidden_size)
    return SiteRegressor(backbone, backbone.config.hidden_size)


def train_eval(mode, key, tr, evals, seed, epochs, lr, stop_at=None, bs=32, lr_head=1e-3):
    """tr로 학습하고 에폭마다 evals(dict 이름→인덱스)의 Spearman을 기록한다. 스케줄(10% 워밍업 + 선형 감소)은 epochs 기준. stop_at 에폭에서 멈춘다.
    반환: {이름: [에폭별 rho]}, 학습 초."""
    torch.manual_seed(seed)
    d = DATA[key]
    net = build(mode).to(DEVICE)
    mu, sd = d["y"][tr].mean(), d["y"][tr].std() + 1e-8
    yz = torch.tensor((d["y"] - mu) / sd)
    if mode == "lora":
        groups = [{"params": [p for n, p in net.backbone.named_parameters() if p.requires_grad], "lr": lr}, {"params": net.head.parameters(), "lr": lr_head}]
    else:
        groups = [{"params": net.backbone.parameters(), "lr": lr}, {"params": net.head.parameters(), "lr": lr_head}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    spe = int(np.ceil(len(tr) / bs))
    total, warm = spe * epochs, max(1, int(0.1 * spe * epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / (total - warm)))
    g = torch.Generator().manual_seed(seed)
    hist = {k: [] for k in evals}
    t0 = time.time()
    for ep in range(stop_at or epochs):
        net.train()
        perm = torch.randperm(len(tr), generator=g).numpy()
        for s in range(0, len(tr), bs):
            ix = tr[perm[s:s + bs]]
            loss = nn.functional.mse_loss(net(d["ids"][ix].to(DEVICE), d["pos"][ix].to(DEVICE)), yz[ix].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            for k, idx in evals.items():
                pred = torch.cat([net(d["ids"][idx[s:s + 256]].to(DEVICE), d["pos"][idx[s:s + 256]].to(DEVICE)).float().cpu() for s in range(0, len(idx), 256)]).numpy()
                hist[k].append(rho(d["y"][idx], pred))
    secs = time.time() - t0
    del net, opt
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return hist, secs


def lr_of(mode, default_cfg=FIXED):
    return default_cfg["lr_lora"] if mode == "lora" else default_cfg["lr_full"]


# ---------- 5. 실행 ----------
LOG = OUT / "results.jsonl"
done = {}
if LOG.exists():
    for line in LOG.read_text().splitlines():
        r = json.loads(line)
        done[(r["assay"], r["seed"], r["method"])] = r
rows = []
t_all = time.time()
for a in ASSAYS:
    d = DATA[a]
    df = d["df"]
    blos = np.array([BLOSUM[w][m] for w, m in zip(df.wt, df.mt)], dtype=float)[:, None]
    B1 = np.hstack([onehot(df.wt), onehot(df.mt), blos])
    P = np.zeros((len(df), d["L"]))
    P[np.arange(len(df)), df.pos.values - 1] = 1
    B2 = np.hstack([B1, P])
    for seed in SEEDS:
        tr, te, groups = make_split(df, seed)
        base = dict(assay=a.split("_")[0], seed=seed)
        for name, X in [("B2", B2), ("probe", FEATS[a])]:
            rows.append(dict(base, method=name, rho=ridge_eval(X, d["y"], groups, tr, te), seconds=0.0))
        # 안쪽 검증 분할: 바깥 train의 위치 블록 단위 (바깥 test와 블록이 겹치지 않음)
        inner = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=1000 + seed)
        itr_rel, iva_rel = next(inner.split(tr, groups=groups[tr]))
        itr, iva = tr[itr_rel], tr[iva_rel]
        assert not (set(groups[iva]) & set(groups[te])) and not (set(groups[itr]) & set(groups[iva])) and not (set(groups[tr]) & set(groups[te]))
        for mode in ["lora", "full"]:
            key = (base["assay"], seed, mode)
            if key in done:
                rows.extend(done[key]["rows"])
                continue
            out_rows = []
            # (a) 고정 설정: 노트북 06/01과 같은 5에폭·기본 LR, 마지막 에폭의 test
            hist, secs = train_eval(mode, a, tr, {"test": te}, seed, FIXED["epochs"], lr_of(mode))
            out_rows.append(dict(base, method=f"{mode}-fixed", rho=hist["test"][-1], seconds=secs))
            # (b) 선택 설정. 주의: 8에폭 스케줄에서 e*에서 멈추므로 LR이 0까지 감쇠하지 않아 고정(5에폭 스케줄)과 스케줄이 다르다. 선택 효과와 스케줄 효과가 섞여 있다. 학습률 3개 × 최대 E_MAX 에폭을 안쪽 train에서 돌려 검증 Spearman이 최대인 (lr, 에폭)을 고른 뒤 바깥 train 전체로 그 에폭까지 학습한다
            t_sel = time.time()
            best = (-9.0, None, None)
            curves = {}
            for lr in GRID[mode]:
                h, _ = train_eval(mode, a, itr, {"val": iva}, seed, E_MAX, lr)
                curves[lr] = h["val"]
                e = int(np.argmax(h["val"])) + 1
                if h["val"][e - 1] > best[0]:
                    best = (h["val"][e - 1], lr, e)
            hist, secs_final = train_eval(mode, a, tr, {"test": te}, seed, E_MAX, best[1], stop_at=best[2])
            out_rows.append(dict(base, method=f"{mode}-selected", rho=hist["test"][-1], seconds=time.time() - t_sel, sel_lr=best[1], sel_epoch=best[2], val_rho=best[0]))
            rows.extend(out_rows)
            with LOG.open("a") as f:
                f.write(json.dumps(dict(assay=base["assay"], seed=seed, method=mode, rows=out_rows)) + "\n")
            print(f"  {base['assay']:8s} seed {seed} {mode:4s}: 고정 {out_rows[0]['rho']:.3f} ({out_rows[0]['seconds']:.0f}s) | 선택 {out_rows[1]['rho']:.3f} (lr {best[1]:g}, 에폭 {best[2]}, 검증 {best[0]:.3f}; {out_rows[1]['seconds']:.0f}s) | 누적 {time.time() - t_all:.0f}s", flush=True)
R = pd.DataFrame(rows)

# ---------- 6. 보고 ----------
M = R.pivot_table(index=["assay", "seed"], columns="method", values="rho")
print("\n--- position-block test Spearman (assay별 3시드 평균) ---")
A = M.groupby("assay").mean()[["B2", "probe", "lora-fixed", "lora-selected", "full-fixed", "full-selected"]]
print(A.round(3).to_string())
print("시드별:")
for a_ in A.index:
    print(f"  {a_:8s} " + " | ".join(f"{m} " + "/".join(f"{x:.2f}" for x in M.loc[a_][m].values) for m in ["probe", "lora-fixed", "lora-selected", "full-fixed", "full-selected"]))
print("\n선택된 (lr, 에폭):")
print(R[R.method.str.endswith("selected")][["assay", "seed", "method", "sel_lr", "sel_epoch", "val_rho", "rho"]].round(3).to_string(index=False))


def klass(delta):
    return "승" if delta >= TIE_BAND else ("패" if delta <= -TIE_BAND else "비김")


print("\n=== 주 질문: 고정 → 선택으로 바꾸면 probe와의 순위가 바뀌는가? (assay 평균 Δ, 승/비김/패 기준 ±0.02) ===")
changed = []
for lab, ft in [("LoRA", "lora"), ("전체", "full")]:
    for setting in ["fixed", "selected"]:
        d_assay = A[f"{ft}-{setting}"] - A["probe"]
        print(f"  {lab:3s} − probe [{setting:8s}] assay별 {', '.join(f'{a_} {x:+.3f}' for a_, x in d_assay.items())} | 평균 {d_assay.mean():+.3f} → {klass(d_assay.mean())} (assay 중 Δ>0 {(d_assay > 0).sum()}/{len(d_assay)})")
    kf, ks = klass((A[f"{ft}-fixed"] - A["probe"]).mean()), klass((A[f"{ft}-selected"] - A["probe"]).mean())
    if kf != ks:
        changed.append((lab, kf, ks))
print(f"  분류가 바뀐 비교: {changed if changed else '없음'} → {'순위가 바뀜(지지)' if changed else '순위가 바뀌지 않음(지지되지 않음)'}")
gain = {ft: (A[f"{ft}-selected"] - A[f"{ft}-fixed"]) for ft in ["lora", "full"]}
print("\n선택으로 인한 자체 변화(선택 − 고정, assay 평균): " + " | ".join(f"{ft} {gain[ft].mean():+.3f} (assay별 {', '.join(f'{x:+.3f}' for x in gain[ft].values)}; 시드 단위 Δ>0 {int(((M[f'{ft}-selected'] - M[f'{ft}-fixed']) > 0).sum())}/{len(M)})" for ft in gain))
cost = R[R.method.str.contains("fixed|selected")].groupby("method").seconds.mean()
print("평균 학습 시간(초/실행): " + ", ".join(f"{k} {v:.0f}" for k, v in cost.items()) + f" | 선택은 고정의 약 {cost['lora-selected'] / cost['lora-fixed']:.1f}배(LoRA), {cost['full-selected'] / cost['full-fixed']:.1f}배(전체)")

print("\n=== 대조 ===")
for ft in ["lora", "full"]:
    dd = M[f"{ft}-fixed"] - M["B2"]
    print(f"  위치 baseline 대비 {ft}-fixed − B2: 평균 {dd.mean():+.3f} (시드 단위 Δ>0 {int((dd > 0).sum())}/{len(dd)}) — 노트북 06/01의 고정 설정 결과와 방향이 같아야 한다")
print(f"\n총 {time.time() - T0:.0f}초")
