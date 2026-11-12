# 트랙 E 시험 실행 3단계: 논문의 표·그림에서 읽은 값을, 캐시된 공개 지표 CSV·데이터와 대조한다 (모델을 다시 돌리지 않는다)
# 논문: Benegas, Eraslan, Song, bioRxiv 10.1101/2025.02.11.637758 v2 (CC BY-NC-ND). 논문의 표(Table 1, S4, S8, S9)와 그림(Figure 5, 보충 그림 S7·S8)에서 읽은 값과 막대 순서·길이는 이 저장소에 두지 않는다.
#   논문을 직접 읽고 `.cache/traitgym_paper_values.json`(git 제외)에 채우면 논문 대조까지 돌고, 파일이 없으면 공개 CSV 재계산과 사후 분석만 돈다.
#   값 파일의 키: t1_mendelian{positives,total,traits}, s4_tss, s8_cols, s8{모델: 열별 값 목록}, s9{모델: 고른 열}, fig5_order, figs7_order, figs8_order{패널: 모델 순서},
#   figs8_endx{패널: {모델: 막대 끝 x 픽셀}}, px_tick[눈금 0.25의 x 픽셀, 1.0에 해당하는 픽셀 길이]. 이미지 판독이라 틀릴 수 있다.
# 규칙(읽은 뒤, 대조 결과를 보기 전에 정함): 개수는 정확히 같아야 하고, AUPRC는 논문의 소수 셋째 자리 반올림과의 차이 ≤ 0.0005(+1e-6 부동소수점 여유, 반올림 경계는 대칭으로 처리), 순위는 어긋나는 쌍이 0개(동률도 어긋남으로 센다)여야 '일치'.
# 점검은 네 종류로 따로 센다: 논문↔캐시(paper), 공개 CSV 재계산(public), 논문 내부 일관성(internal, 옮긴 값만 쓰므로 옮김 오류를 독립적으로 걸러 주지 못한다), 사후 분석(post).
# 대조할 수 있는 것은 캐시에 점수가 있는 칸뿐이다. 나머지 칸과 다른 표·그림은 대조하지 못했다.
# 실행: 저장소 루트에서 `HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I 07_capstone/examples/dry_runs/track_E_traitgym_tables.py`
import glob, itertools, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
ROOT = Path.cwd()
PUB = ROOT / ".cache" / "traitgym_published" / "all"
TOL = 0.0005 + 1e-6

# ---------- 논문에서 옮긴 값: 저장소 밖의 로컬 파일에서 읽는다 ----------
PAPER_FILE = ROOT / ".cache" / "traitgym_paper_values.json"
PAPER = json.loads(PAPER_FILE.read_text(encoding="utf-8")) if PAPER_FILE.exists() else None
if PAPER is None:
    print(f"[알림] {PAPER_FILE.relative_to(ROOT)} 가 없다. 논문 값과 대조하는 점검은 건너뛰고 공개 CSV 재계산과 사후 분석만 돈다.")
else:
    T1_MENDELIAN, S4_TSS, S8_COLS, S8, S9 = PAPER["t1_mendelian"], PAPER["s4_tss"], PAPER["s8_cols"], PAPER["s8"], PAPER["s9"]
    FIG5_ORDER, FIGS7_ORDER, FIGS8_ORDER = PAPER["fig5_order"], PAPER["figs7_order"], PAPER["figs8_order"]
    FIGS8_ENDX, PX_TICK = PAPER["figs8_endx"], PAPER["px_tick"]

# 논문 모델명 → 캐시된 공개 CSV (파일 이름의 GPN_final은 논문 표의 GPN-Promoter LLR 칸과 값이 일치해 그 모델로 해석했다. 값 일치에 근거한 추정이다)
CSV = {"GPN-MSA": ("LLR", "GPN-MSA_LLR.minus.score"), "GPN-MSA/abs": ("abs(LLR)", "GPN-MSA_absLLR.plus.score"), "GPN-Promoter": ("LLR", "GPN_final_LLR.minus.score"),
       "NT": ("LLR", "NucleotideTransformer_LLR.minus.score"), "NT/abs": ("abs(LLR)", "NucleotideTransformer_absLLR.plus.score"), "HyenaDNA": ("LLR", "HyenaDNA_LLR.minus.score"),
       "Caduceus": ("LLR", "Caduceus_LLR.minus.score"), "Evo2 1B": ("LLR", "evo2_1b_base_LLR.minus.score"), "Evo2 7B": ("LLR", "evo2_7b_LLR.minus.score"),
       "Evo2 40B": ("LLR", "evo2_40b_LLR.minus.score")}
FIG5_CSV = {"Borzoi": "Borzoi_L2_L2.plus.all", "CADD": "CADD.plus.RawScore", "Enformer": "Enformer_L2_L2.plus.all", "Evo2 40B": "evo2_40b_LLR.minus.score",
            "Evo2 7B": "evo2_7b_LLR.minus.score", "GPN-MSA": "GPN-MSA_LLR.minus.score", "GPN-Promoter": "GPN_final_LLR.minus.score", "Sei": "Sei.plus.seqclass_max_absdiff",
            "phastCons-43p": "phastCons-43p.plus.score", "phyloP-100v": "phyloP-100v.plus.score", "phyloP-241m": "phyloP-241m.plus.score"}   # 모델 이름 순(그림의 순서가 아니다)


def pub_auprc(stem):
    r = pd.read_csv(PUB / f"{stem}.csv")
    assert list(r.columns) == ["model", "metric", "score", "se"] and len(r) == 1 and r.metric.iloc[0] == "AUPRC", stem
    return float(r.score.iloc[0])


snap = sorted(glob.glob(str(ROOT / ".cache/hf/hub/datasets--songlab--TraitGym/snapshots/*/mendelian_traits_matched_9/test.parquet")))
assert len(snap) == 1
t = pd.read_parquet(snap[0])
assert (t.groupby('match_group').size() == 10).all() and (t.groupby('match_group').label.sum() == 1).all(), "match_group마다 변이 10개·양성 1개라는 가정이 깨졌다"
RES = {"paper": [], "public": [], "internal": [], "post": []}     # 논문↔캐시 / 공개 CSV 재계산 / 논문 내부 일관성 / 사후 분석(그림의 지표를 식별한 뒤 추가)
TAG = {"paper": "", "public": "(공개 CSV)", "internal": "(논문 내부)", "post": "(사후)"}


def check(name, ok, detail, kind="paper"):
    RES[kind].append(ok)
    print(f"  [{'일치' if ok else '불일치'}]{TAG[kind]} {name}: {detail}")


if PAPER:
    print("=== Table 1 (Mendelian) ===")
    check("양성 수 / 전체 / trait 수", (int(t.label.sum()), len(t), int(t[t.label].OMIM.nunique())) == tuple(T1_MENDELIAN.values()),
          f"캐시 {int(t.label.sum())} / {len(t)} / {t[t.label].OMIM.nunique()} (OMIM id) vs 논문 {tuple(T1_MENDELIAN.values())}")

    print("\n=== Table S4: (-) TSS 거리, 전체 풀링 AUPRC ===")
    ap = average_precision_score(t.label, -t.tss_dist)
    check("(-) TSS distance", abs(ap - S4_TSS) <= TOL, f"캐시 {ap:.4f} vs 논문 {S4_TSS}")

    print("\n=== Table S8 (Mendelian) 중 캐시에 점수가 있는 칸 ===")
    for key, (col, stem) in CSV.items():
        model = key.split("/")[0]
        pub = S8[model][S8_COLS.index(col)]
        v = pub_auprc(stem)
        check(f"{model} {col}", abs(v - pub) <= TOL, f"캐시 {v:.4f} vs 논문 {pub:.3f}")
    n_cells = sum(len(v) for v in S8.values())
    print(f"  (대조한 칸 {len(CSV)}개 / 옮긴 Mendelian 칸 {n_cells}개 — 나머지는 캐시에 점수가 없어 대조하지 못했다)")

    print("\n=== Table S9 ↔ S8 일관성 (논문 내부 점검; 옮긴 값 점검 겸용) ===")
    for model, sel in S9.items():
        row = dict(zip(S8_COLS, S8[model]))
        best = max(row.values())
        check(f"{model}: 고른 점수 {sel}", row[sel] >= best - 1e-12, f"S8에서 {sel} = {row[sel]:.3f}, 행의 최댓값 {best:.3f} ({[c for c, x in row.items() if x == best]})", kind="internal")

    print("\n=== Figure 5 (Mendelian zero-shot) 순위 ↔ 캐시 값 순위 ===")
    vals = {k: pub_auprc(f) for k, f in FIG5_CSV.items()}
    disc = [(a, b) for a, b in itertools.combinations(FIG5_ORDER, 2) if vals[a] <= vals[b]]
    check(f"모델 {len(FIG5_ORDER)}개의 순위", len(disc) == 0, f"어긋나는 쌍 {len(disc)}/{len(FIG5_ORDER) * (len(FIG5_ORDER) - 1) // 2} {disc}")
    print("  캐시 값 순서:", " > ".join(f"{k} {vals[k]:.3f}" for k in sorted(vals, key=lambda k: -vals[k])))

SUBS_PUB = {"5UTR": "5_prime_UTR_variant", "3UTR": "3_prime_UTR_variant", "ncRNA": "non_coding_transcript_exon_variant", "distal": "nonexonic_AND_distal", "proximal": "nonexonic_AND_proximal"}
NON_EXONIC = ["intergenic_variant", "intron_variant", "upstream_gene_variant", "downstream_gene_variant"]
CRE = ["PLS", "pELS", "dELS", "DNase-H3K4me3", "CTCF-only"]
NONEXONIC_FULL = NON_EXONIC + CRE + [f"{c}_flank" for c in CRE]
D = Path(snap[0]).parent
y = t.label.values.astype(int)
chrom = t.chrom.astype(str).values
grp = t.match_group.values


def load_score(stem):
    for sign in ("plus", "minus"):
        if f".{sign}." in stem:
            feats, col = stem.split(f".{sign}.", 1)
            x = pd.read_parquet(D / "features" / f"{feats}.parquet", columns=[col])[col].values.astype(float)
            assert len(x) == len(t) and not np.isnan(x).any()
            return -x if sign == "minus" else x
    raise ValueError(stem)


def wavg(yy, ss, cc):
    """공개 코드의 가중 평균 AUPRC: 염색체별 AP를 변이 수로 가중 (양성 없는 염색체는 AP 0.0)."""
    tot, acc = 0, 0.0
    for c in pd.unique(cc):
        m = cc == c
        acc += m.sum() * (average_precision_score(yy[m], ss[m]) if yy[m].sum() > 0 else 0.0)
        tot += m.sum()
    return acc / tot


def subset_mask(name):
    if name in ("ncRNA", "5UTR", "3UTR"):
        return (t.consequence == SUBS_PUB[name]).values
    nonex = t.consequence.isin(NONEXONIC_FULL).values
    target = len(t[t.match_group == t.match_group.iloc[0]])
    keep = (~t.label.values.astype(bool)) | ((t.tss_dist.values <= 1000) if name == "proximal" else (t.tss_dist.values > 1000))
    return nonex & keep & (pd.Series(keep).groupby(grp).transform("sum").values == target)


print("\n=== 부분집합 규칙 검증: 공개 부분집합 CSV(GPN_final LLR)를 우리 마스크로 다시 계산 ===")
gpn = load_score("GPN_final_LLR.minus.score")
MASK = {n: subset_mask(n) for n in SUBS_PUB}
for n, folder in SUBS_PUB.items():
    r = pd.read_csv(ROOT / ".cache" / "traitgym_published" / folder / "GPN_final_LLR.minus.score.csv")
    m = MASK[n]
    v = wavg(y[m], gpn[m], chrom[m])
    check(f"{n} (양성 {int(y[m].sum())}개)", abs(v - float(r.score.iloc[0])) <= 1e-6, f"재계산 {v:.4f} vs 공개 CSV {float(r.score.iloc[0]):.4f}", kind="public")

EV = {"CADD": "CADD.plus.RawScore", "Evo2 1B": "evo2_1b_base_LLR.minus.score", "Evo2 7B": "evo2_7b_LLR.minus.score", "Evo2 40B": "evo2_40b_LLR.minus.score"}
SC8 = {k: load_score(v) for k, v in EV.items()}

if PAPER:
    print("\n=== Figure S7 (전체 풀링 AUPRC) 순위 ↔ 캐시 점수로 계산한 풀링 AUPRC 순위 ===")
    S = {k: load_score(FIG5_CSV[k]) for k in FIGS7_ORDER}
    pooled = {k: average_precision_score(y, S[k]) for k in FIGS7_ORDER}
    disc7 = [(a, b) for a, b in itertools.combinations(FIGS7_ORDER, 2) if pooled[a] <= pooled[b]]
    check(f"모델 {len(FIGS7_ORDER)}개의 순위", len(disc7) == 0, f"어긋나는 쌍 {len(disc7)}/{len(FIGS7_ORDER) * (len(FIGS7_ORDER) - 1) // 2} {disc7}")
    print("  풀링 AUPRC 순서:", " > ".join(f"{k} {pooled[k]:.3f}" for k in sorted(pooled, key=lambda k: -pooled[k])))

    print("\n=== Figure S8 (Evo2 LLR, consequence별) 막대 순서 ↔ 캐시 점수로 계산한 가중 평균 AUPRC 순서 (사전 지정: 공개 CSV와 같은 지표) ===")
    for n, order in FIGS8_ORDER.items():
        m = MASK[n]
        v = {k: wavg(y[m], SC8[k][m], chrom[m]) for k in order}
        d8 = [(a, b) for a, b in itertools.combinations(order, 2) if v[a] <= v[b]]
        check(f"{n}: {' > '.join(order)}", len(d8) == 0, f"어긋나는 쌍 {len(d8)}/6 {d8} | 계산값 " + ", ".join(f"{k} {v[k]:.3f}" for k in order))

    # ---------- (사후 분석) Figure S8가 쓴 지표 식별 ----------
    # 위 S8 순서 점검에서 가중 평균 지표는 5′UTR에서 1쌍(CADD vs Evo2 7B, 0.589 vs 0.596)이 어긋났다. 이미지 픽셀로 막대 끝을 재어(x축 눈금을 보정에 사용, ±0.01 수준)
    # AUPRC로 환산한 값을 값 파일의 figs8_endx와 px_tick에 옮겼다(막대가 너무 짧은 것은 재지 못했다). 값 = 막대 한 줄 한 줄의 같은 색 구간 끝 x의 중앙값(+1 px).
    # 이미지는 저장소에 없어 이 측정은 재현할 수 없고, 캐시 값을 본 *뒤에* 한 비맹검 측정이다. 이 식별은 결과를 본 뒤에 한 것이므로 '사후'로 표시한다. 아래 0.01 기준은 측정 정밀도(±0.01)에 맞춘 사후 기준이다.
    FIGS8_PX = {n: {k: 0.25 + (x + 1 - PX_TICK[0]) / PX_TICK[1] for k, x in d.items()} for n, d in FIGS8_ENDX.items()}
    print("\n=== (사후) Figure S8의 막대 길이(픽셀 측정) ↔ 가중 평균 AUPRC·풀링 AUPRC ===")
    dw, dp, nn = [], [], 0
    for n, d in FIGS8_PX.items():
        m = MASK[n]
        for k, v in d.items():
            dw.append(abs(wavg(y[m], SC8[k][m], chrom[m]) - v))
            dp.append(abs(average_precision_score(y[m], SC8[k][m]) - v))
    print(f"  막대 {len(dw)}개의 평균 |차이|: 가중 평균 {np.mean(dw):.3f} (최대 {np.max(dw):.3f}) | 풀링 {np.mean(dp):.3f} (최대 {np.max(dp):.3f})")
    check("그림의 지표는 풀링 AUPRC에 더 가깝다", np.mean(dp) < 0.01 and np.mean(dp) < np.mean(dw), f"풀링 {np.mean(dp):.3f} < 0.01, 가중 평균 {np.mean(dw):.3f}", kind="post")
    for n, order in FIGS8_ORDER.items():
        m = MASK[n]
        v = {k: average_precision_score(y[m], SC8[k][m]) for k in order}
        d8 = [(a, b) for a, b in itertools.combinations(order, 2) if v[a] <= v[b]]
        check(f"{n} 막대 순서(풀링 AUPRC)", len(d8) == 0, f"어긋나는 쌍 {len(d8)}/6 | " + ", ".join(f"{k} {v[k]:.3f}" for k in order), kind="post")

print("\n=== (사후) K3을 풀링 AUPRC로: Evo2 1B → 40B 이득 (문턱은 K3과 같은 0.10/0.10/0.05, 부트스트랩은 부분집합 안의 match_group 복원추출 1,000회) ===")
gain_p = {}
for idx, name in enumerate(["proximal", "ncRNA", "distal"]):
    mask = MASK[name]
    ug_s = np.unique(grp[mask])
    gi = {g: np.where((grp == g) & mask)[0] for g in ug_s}
    rs3 = np.random.RandomState(100 + idx)
    gb = []
    for _ in range(1000):
        ii = np.concatenate([gi[g] for g in rs3.choice(ug_s, len(ug_s), replace=True)])
        gb.append(average_precision_score(y[ii], SC8["Evo2 40B"][ii]) - average_precision_score(y[ii], SC8["Evo2 1B"][ii]))
    a1, a40 = average_precision_score(y[mask], SC8["Evo2 1B"][mask]), average_precision_score(y[mask], SC8["Evo2 40B"][mask])
    gain_p[name] = a40 - a1
    lo, hi = np.percentile(gb, [2.5, 97.5])
    print(f"  {name:9s} 양성 {int(y[mask].sum()):3d} | 1B {a1:.3f} → 40B {a40:.3f}: 이득 {a40 - a1:+.3f} [{lo:+.3f}, {hi:+.3f}]")
k3p = gain_p["proximal"] >= 0.10 and gain_p["ncRNA"] >= 0.10 and gain_p["distal"] <= 0.05
print(f"  → K3(풀링; proximal ≥ 0.10, ncRNA ≥ 0.10, distal ≤ 0.05): {'지지' if k3p else '지지되지 않음'}  [가중 평균 지표의 K3은 distal +0.053으로 '지지되지 않음']")

print(f"\n요약: 논문↔캐시 {sum(RES['paper'])}/{len(RES['paper'])} 일치 | 공개 CSV 재계산 {sum(RES['public'])}/{len(RES['public'])} | 논문 내부 일관성(제한적) {sum(RES['internal'])}/{len(RES['internal'])} | 사후 분석 {sum(RES['post'])}/{len(RES['post'])}")
if PAPER is None:
    print("논문 값 파일이 없어 논문↔캐시와 논문 내부 일관성 점검은 하지 않았다.")
print("불일치 항목:", "없음" if all(all(v) for k, v in RES.items() if k != "post") else "있음 — 위 [불일치] 줄 참조 (사전 지정 규칙은 바꾸지 않았다)")
