# Cheatsheet 03 — Splits and Leakage

> **English translation.** AI-assisted translation of the [Korean original](03_splits_and_leakage.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original. In the code blocks only the comments and message strings were translated; the executable code is unchanged.

**Principle**: Write first not the name of the split but *the leakage path you want to block*, and after splitting, measure **with numbers** whether that path was blocked. The `python` blocks were run on synthetic/small data and confirmed (2026-11-06, repository `.venv`; RDKit 2026.09.1). MMseqs2 is not installed on this Mac, so the `bash` block was **not run** (the options were checked in the official documentation).

## 1. Split unit by domain
| Data | Path to block | Split unit | What to re-check after splitting | This repository |
|---|---|---|---|---|
| Protein sequences | Homologous sequences in train and test | Sequence-identity cluster / family | The distribution of the test→train maximum identity, the threshold | Module 01 notebook 03 |
| Variant effects (DMS, one protein) | Other substitutions at the same position | Position block | The fraction of test positions in train, the distance | Modules 02, 07 |
| DNA sequences | Overlapping/repeated regions, homologous regions | Chromosome / region / cluster | Overlap length, the fraction of repeated sequences | Module 03 |
| Single-cell | Cells of the same donor/study/batch | Donor / study | A confounding table of donor, batch and label | Module 04 |
| Molecules | The same scaffold, similar molecules | scaffold / time | The test→train nearest Tanimoto | Module 05 |
| Text–sequence alignment | Annotations transferred on the basis of sequence similarity | Sequence cluster + removal of duplicate texts | The fraction of near-duplicate texts | Module 05 |
| Pathology images | Patches of the same medical center/slide | Patient / center | Linear separability of the center signal | Module 05 (reading) |
| Pretraining ↔ evaluation | The evaluation data is in the pretraining corpus | (cannot be split) | A duplicate lookup; if you could not do it, write it as a limitation | All modules |

**Common**: ① Group the same sequence, molecule or text into one item **before splitting**. Merge them or put them on the same side (module 05: 33 duplicate sequences straddled train and test). If the labels conflict when merging, write the handling rule (for measured values, merge them after checking that the assays and units are compatible). ② Do not choose settings (layer, epochs, threshold) with the test set. ③ Fit standardization, PCA and feature selection only inside the training fold.

## 2. Position-block split + leakage numbers (variant data)
```python
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

rng = np.random.default_rng(0)
L = 120
pos = rng.integers(1, L + 1, size=1000)                  # the positions of 1000 variants (1-based)

def block_groups(pos, block=10):
    return (np.asarray(pos) - 1) // block               # `block` consecutive positions form one group

def leakage_report(pos, groups, train_idx, test_idx):
    """groups: the group array actually used for the split (one per row for a random split)."""
    tr_pos, te_pos = set(pos[train_idx]), pos[test_idx]
    in_train = float(np.mean([p in tr_pos for p in te_pos]))
    trp = np.array(sorted(tr_pos))
    dist = np.array([np.abs(trp - p).min() for p in te_pos])
    return dict(test_frac=len(test_idx) / (len(train_idx) + len(test_idx)),
                test_pos_in_train=in_train, median_dist_to_train=float(np.median(dist)),
                n_test_positions=len(set(te_pos)), n_test_groups=len(np.unique(np.asarray(groups)[test_idx])))

idx = np.arange(len(pos))
# (a) Random split: other substitutions at the same position are in train
tr, te = GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=idx).__next__()
rand = leakage_report(pos, idx, tr, te)
# (b) Position-block split
tr, te = GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=block_groups(pos)).__next__()
blk = leakage_report(pos, block_groups(pos), tr, te)
assert rand["test_pos_in_train"] > 0.95 and blk["test_pos_in_train"] == 0.0
assert blk["median_dist_to_train"] >= 1
print("random:", {k: round(v, 2) for k, v in rand.items()})
print("block :", {k: round(v, 2) for k, v in blk.items()})
```
- The block split blocks the *same-position* path, but **adjacent positions** (a distance of a few positions) and **the same substitution type** are still in train. Write the paths that were not blocked in the report.
- The test fraction may differ from the target (20%) (when the block sizes are not uniform). Report the actual fraction.
- `n_test_groups` in the output is the number of *groups used for the split* (for a random split, each row is a group, so it equals the number of test rows). **If the number of test clusters (blocks) is 2–3**, the group bootstrap is meaningless. See cheatsheet 02 §6.
- If the blocks are too large, there are too few groups and the inner CV (`GroupKFold`) does not hold (for a length-55 protein, block 15 gave only 4 groups).

## 3. Splitting by sequence cluster (MMseqs2 → groups)
```bash
# not run (mmseqs not installed). The options follow the official MMseqs2 README/wiki.
mmseqs easy-cluster seqs.fasta clusterRes tmp --min-seq-id 0.3 -c 0.8 --cov-mode 0
# output: clusterRes_cluster.tsv  (each line: representative sequence <TAB> member)
```
```python
import io, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# a synthetic example in the clusterRes_cluster.tsv format (representative \t member)
tsv = "A\tA\nA\tB\nA\tC\nD\tD\nD\tE\nF\tF\nG\tG\nG\tH\n"
cl = pd.read_csv(io.StringIO(tsv), sep="\t", header=None, names=["rep", "member"])
seq_ids = sorted(cl.member.unique())
group_of = dict(zip(cl.member, cl.rep))
groups = np.array([group_of[s] for s in seq_ids])
tr, te = GroupShuffleSplit(1, test_size=0.4, random_state=1).split(seq_ids, groups=groups).__next__()
assert set(groups[tr]).isdisjoint(set(groups[te]))      # the same cluster goes to only one side
print("train", [seq_ids[i] for i in tr], "| test", [seq_ids[i] for i in te])
```
- The **lower** `--min-seq-id` is (e.g., 0.3 < 0.9), the larger the clusters and the stricter the split. **Report the threshold and the coverage options (`-c`, `--cov-mode`)**.
- `--cov-mode 0` requires coverage of both the query and the target, which suits full-length proteins, and `--cov-mode 1` (target coverage) suits fragments (a summary of the official wiki FAQ). If you use mode 1 for splitting full-length proteins, short sequences are absorbed into long ones, whereas long sequences that share the same domain stay separate, so **domain-level leakage** can arise†.
- Representative-sequence-based clustering can leave *pairs of sequences from different clusters that are at least as similar as the threshold*. If strict separation is needed, directly check the test→train maximum identity by search after splitting (module 01 §5).

## 4. Scaffold split + nearest similarity (molecules)
```python
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

smiles = ["c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1O", "C1CCNCC1CC", "C1CCNCC1O", "c1ccc2ccccc2c1C", "c1ccc2ccccc2c1N", "CCO", "CCN", "CC(=O)O"]
mols = [Chem.MolFromSmiles(s) for s in smiles]
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
# a molecule without rings has an empty-string scaffold → give each molecule its own group so that they do not become one giant group (module 05)
group = [s if s else f"__acyclic_{i}" for i, s in enumerate(scaf)]

# make groups by scaffold, fill train from the largest group and put the rest in test
from collections import defaultdict
by_scaf = defaultdict(list)
for i, s in enumerate(group):
    by_scaf[s].append(i)
order = sorted(by_scaf.values(), key=len, reverse=True)
train, test = [], []
for g in order:
    (train if len(train) + len(g) <= 0.7 * len(smiles) else test).extend(g)
assert train and test, "if it cannot be split at the target ratio, check the cases where train/test end up empty"
assert set(group[i] for i in train).isdisjoint(group[i] for i in test)

# leakage in numbers: for each test molecule, the nearest Tanimoto to train (ECFP4, 2048 bits)
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)    # ECFP4 (2048 bits)
fps = [fpgen.GetFingerprint(m) for m in mols]
nn = [max(DataStructs.BulkTanimotoSimilarity(fps[j], [fps[i] for i in train])) for j in test]
print("number of groups", len(by_scaf), "| train", len(train), "test", len(test), f"(test fraction {len(test)/len(smiles):.2f})", "| median test→train nearest Tanimoto", round(float(np.median(nn)), 2))
assert len(nn) == len(test) and all(0.0 <= v <= 1.0 for v in nn)
```
- The split above is a greedy implementation for explanation (fill train from the largest group and put the rest in test). If the largest group exceeds the target size or there are few groups, the ratio deviates greatly from the target or train/test becomes empty, so **report the actual ratio**.
- *Different scaffolds do not mean different molecules*. In module 05, even with the scaffold split, 26% of the Lipophilicity test set had a molecule in train with Tanimoto ≥ 0.7 (median nearest similarity 0.57). So, after splitting, **measure and report the nearest similarity**, as above.
- The old `AllChem.GetMorganFingerprintAsBitVect` printed `DEPRECATION WARNING: please use MorganGenerator` in this RDKit (2026.09.1) (it gave the same fingerprint for the same molecule). So `rdFingerprintGenerator` is used (the same as the module 05 notebook).

## 5. Donor (group) split (single-cell)
```python
from sklearn.model_selection import GroupKFold

donor = np.repeat(np.arange(8), [50, 40, 60, 30, 45, 55, 35, 25])    # 340 cells, 8 donors
X = np.zeros((len(donor), 1))
for k, (tr, te) in enumerate(GroupKFold(n_splits=4).split(X, groups=donor)):
    assert set(donor[tr]).isdisjoint(set(donor[te]))
    if k == 0:
        print("fold0 test donors", sorted(int(d) for d in set(donor[te])), "cells", len(te))
```
- In a cell-level random split, cells of the same donor go to both sides, which favors a model that learned the batch signal. First check in a table **whether donor = batch = disease status overlap** (module 04).
- With few groups (e.g., 13 donors), the interval is wide. A donor-level bootstrap is a conditional interval.

## 6. One-line checks
- [ ] You wrote the path you want to block / **the paths that cannot be blocked**
- [ ] After splitting, you measured leakage in numbers (the fraction of shared positions, the nearest similarity, donor overlap)
- [ ] You grouped duplicates of the same item (sequence, molecule, text) before splitting (merge or the same split), and wrote the label-conflict rule
- [ ] You checked the number of test clusters (position blocks, families, donors)
- [ ] You looked up the overlap with the pretraining corpus, or wrote down that you could not, as a limitation

<!-- i18n-source: resources/cheatsheets/03_splits_and_leakage.md blob:a91233377d387d67a9e1edcd32d88ffd41269d60 -->
