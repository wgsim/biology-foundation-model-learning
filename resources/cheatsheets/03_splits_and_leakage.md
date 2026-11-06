# 치트시트 03 — 분할과 누출

**원칙**: 분할 이름이 아니라 *막으려는 누출 경로*를 먼저 쓰고, 분할 뒤에 그 경로가 막혔는지 **숫자로** 잰다. `python` 블록은 합성/소형 데이터로 실행해 확인했다(2026-11-06, 저장소 `.venv`; RDKit 2026.09.1). MMseqs2는 이 Mac에 설치되어 있지 않아 `bash` 블록은 **실행하지 않았다**(옵션은 공식 문서로 확인).

## 1. 도메인별 분할 단위
| 데이터 | 막으려는 경로 | 분할 단위 | 분할 후 재볼 것 | 이 저장소 |
|---|---|---|---|---|
| 단백질 서열 | 상동 서열이 train과 test에 | 서열 동일도 클러스터 / 가족 | test→train 최대 동일도 분포, 임계값 | 모듈 01 노트북 03 |
| 변이 효과(DMS, 한 단백질) | 같은 위치의 다른 치환 | 위치 블록 | test 위치가 train에 있는 비율, 거리 | 모듈 02, 07 |
| DNA 서열 | 겹치는/반복 구간, 상동 구간 | 염색체 / 구간 / 클러스터 | 겹침 길이, 반복 서열 비율 | 모듈 03 |
| 단일세포 | 같은 기증자/연구/배치의 세포 | 기증자 / 연구 | 기증자·배치와 라벨의 교락 표 | 모듈 04 |
| 분자 | 같은 scaffold, 비슷한 분자 | scaffold / 시간 | test→train 최근접 Tanimoto | 모듈 05 |
| 텍스트–서열 정렬 | 서열 유사성으로 옮겨 붙은 주석 | 서열 클러스터 + 텍스트 중복 제거 | 근접 중복 텍스트 비율 | 모듈 05 |
| 병리 이미지 | 같은 의료기관/슬라이드의 패치 | 환자 / 기관 | 기관 신호 선형 분리 가능성 | 모듈 05 (읽기) |
| 사전학습 ↔ 평가 | 평가 데이터가 사전학습 코퍼스에 있음 | (분할 불가) | 중복 조회, 못 하면 한계로 기재 | 전 모듈 |

**공통**: ① 같은 서열·분자·텍스트는 **분할 전에** 병합한다(모듈 05: 중복 서열 33개가 train과 test에 걸쳤다). ② 설정(층·에폭·임계값)은 test로 고르지 않는다. ③ 표준화·PCA·특징 선택은 학습 접힘 안에서만 적합한다.

## 2. 위치 블록 분할 + 누출 수치 (변이 데이터)
```python
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

rng = np.random.default_rng(0)
L = 120
pos = rng.integers(1, L + 1, size=1000)                  # 변이 1000개의 위치 (1-based)

def block_groups(pos, block=10):
    return (np.asarray(pos) - 1) // block               # 연속 `block`개 위치가 한 그룹

def leakage_report(pos, train_idx, test_idx):
    tr_pos, te_pos = set(pos[train_idx]), pos[test_idx]
    in_train = float(np.mean([p in tr_pos for p in te_pos]))
    trp = np.array(sorted(tr_pos))
    dist = np.array([np.abs(trp - p).min() for p in te_pos])
    return dict(test_frac=len(test_idx) / (len(train_idx) + len(test_idx)),
                test_pos_in_train=in_train, median_dist_to_train=float(np.median(dist)),
                n_test_groups=len(np.unique(block_groups(pos[test_idx]))))

idx = np.arange(len(pos))
# (a) 무작위 분할: 같은 위치의 다른 치환이 train에 있다
tr, te = GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=idx).__next__()
rand = leakage_report(pos, tr, te)
# (b) 위치 블록 분할
tr, te = GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=block_groups(pos)).__next__()
blk = leakage_report(pos, tr, te)
assert rand["test_pos_in_train"] > 0.95 and blk["test_pos_in_train"] == 0.0
assert blk["median_dist_to_train"] >= 1
print("random:", {k: round(v, 2) for k, v in rand.items()})
print("block :", {k: round(v, 2) for k, v in blk.items()})
```
- 블록 분할은 *같은 위치* 경로를 막지만 **인접 위치**(거리 수 칸)와 **같은 치환 유형**은 여전히 train에 있다. 막지 못한 경로를 보고서에 적는다.
- test 비율이 목표(20%)와 다를 수 있다(블록 크기가 균일하지 않을 때). 실제 비율을 보고한다.
- **test 군집(블록) 수가 2~3개면** 그룹 부트스트랩은 무의미하다 → 치트시트 02 §6.
- 블록이 너무 크면 그룹이 적어 내부 CV(`GroupKFold`)가 성립하지 않는다 (길이 55 단백질에서 블록 15는 그룹이 4개뿐이었다).

## 3. 서열 클러스터로 분할 (MMseqs2 → 그룹)
```bash
# 실행하지 않음 (mmseqs 미설치). 옵션은 MMseqs2 공식 README/위키 기준.
mmseqs easy-cluster seqs.fasta clusterRes tmp --min-seq-id 0.3 -c 0.8 --cov-mode 1
# 출력: clusterRes_cluster.tsv  (각 줄: 대표 서열 <TAB> 구성원)
```
```python
import io, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# clusterRes_cluster.tsv 형식의 합성 예시 (대표 \t 구성원)
tsv = "A\tA\nA\tB\nA\tC\nD\tD\nD\tE\nF\tF\nG\tG\nG\tH\n"
cl = pd.read_csv(io.StringIO(tsv), sep="\t", header=None, names=["rep", "member"])
seq_ids = sorted(cl.member.unique())
group_of = dict(zip(cl.member, cl.rep))
groups = np.array([group_of[s] for s in seq_ids])
tr, te = GroupShuffleSplit(1, test_size=0.4, random_state=1).split(seq_ids, groups=groups).__next__()
assert set(groups[tr]).isdisjoint(set(groups[te]))      # 같은 클러스터는 한쪽에만
print("train", [seq_ids[i] for i in tr], "| test", [seq_ids[i] for i in te])
```
- 임계값(`--min-seq-id`)이 느슨할수록 보수적이다. **임계값과 coverage 옵션을 보고한다**.
- 대표 서열 기반 클러스터링은 *서로 다른 클러스터의 서열 쌍도 임계값 이상으로 유사*할 수 있다. 엄격한 분리가 필요하면 분할 뒤 test→train 최대 동일도를 검색으로 직접 확인한다 (모듈 01 §5).

## 4. Scaffold 분할 + 최근접 유사도 (분자)
```python
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

smiles = ["c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1O", "C1CCNCC1CC", "C1CCNCC1O", "c1ccc2ccccc2c1C", "c1ccc2ccccc2c1N", "CCO", "CCN", "CC(=O)O"]
mols = [Chem.MolFromSmiles(s) for s in smiles]
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]

# scaffold 단위로 그룹을 만들어 큰 그룹부터 train에 채우고 나머지를 test로
from collections import defaultdict
by_scaf = defaultdict(list)
for i, s in enumerate(scaf):
    by_scaf[s].append(i)
order = sorted(by_scaf.values(), key=len, reverse=True)
train, test = [], []
for g in order:
    (train if len(train) + len(g) <= 0.7 * len(smiles) else test).extend(g)
assert set(scaf[i] for i in train).isdisjoint(scaf[i] for i in test)

# 누출을 숫자로: test 분자마다 train에서의 최근접 Tanimoto (ECFP4, 2048비트)
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)    # ECFP4 (2048비트)
fps = [fpgen.GetFingerprint(m) for m in mols]
nn = [max(DataStructs.BulkTanimotoSimilarity(fps[j], [fps[i] for i in train])) for j in test]
print("scaffold 개수", len(by_scaf), "| train", len(train), "test", len(test), "| test→train 최근접 Tanimoto 중앙값", round(float(np.median(nn)), 2))
assert len(nn) == len(test) and all(0.0 <= v <= 1.0 for v in nn)
```
- *scaffold가 다르다고 분자가 다른 것은 아니다*. 모듈 05에서 scaffold 분할에서도 Lipophilicity test의 26%가 train에 Tanimoto ≥ 0.7인 분자를 가졌다(최근접 유사도 중앙값 0.57). 그래서 분할 뒤 위처럼 **최근접 유사도를 측정해 보고**한다.
- 옛 `AllChem.GetMorganFingerprintAsBitVect`는 이 RDKit(2026.09.1)에서 `DEPRECATION WARNING: please use MorganGenerator`를 냈다(같은 분자에서 같은 지문을 냄). 그래서 `rdFingerprintGenerator`를 쓴다 (모듈 05 노트북과 같다).

## 5. 기증자(그룹) 단위 분할 (단일세포)
```python
from sklearn.model_selection import GroupKFold

donor = np.repeat(np.arange(8), [50, 40, 60, 30, 45, 55, 35, 25])    # 세포 340개, 기증자 8명
X = np.zeros((len(donor), 1))
for k, (tr, te) in enumerate(GroupKFold(n_splits=4).split(X, groups=donor)):
    assert set(donor[tr]).isdisjoint(set(donor[te]))
    if k == 0:
        print("fold0 test 기증자", sorted(int(d) for d in set(donor[te])), "세포", len(te))
```
- 세포 무작위 분할은 같은 기증자의 세포가 양쪽에 들어가 배치 신호를 학습한 모델이 유리해진다. **기증자=배치=질병 상태가 겹치는지**를 먼저 표로 확인한다 (모듈 04).
- 그룹 수가 적으면(기증자 13명 등) 구간이 넓다. 기증자 단위 부트스트랩은 조건부 구간이다.

## 6. 점검 한 줄
- [ ] 막으려는 경로 / **막지 못하는 경로**를 적었다
- [ ] 분할 *후* 누출을 수치로 쟀다 (위치 공유율, 최근접 유사도, 기증자 겹침)
- [ ] 같은 항목(서열·분자·텍스트)의 중복을 분할 전에 병합했다
- [ ] test의 군집(위치 블록·가족·기증자) 수를 확인했다
- [ ] 사전학습 코퍼스와의 중복을 조회했거나, 못 했음을 한계로 적었다
