# Cheatsheet 01 — Embedding and Zero-Shot Score Recipes (Protein LMs)

> **English translation.** AI-assisted translation of the [Korean original](01_embedding_recipes.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original. In the code blocks only the comments and message strings were translated; the executable code is unchanged.

This is a condensed version of the patterns used in notebooks 01 and 02 of module 02. The `python` code blocks of this file were **run in order and confirmed in the repository environment (`.venv`, ESM-2 8M cache)** (2026-11-06, transformers 5.22.1, torch 2.15.0, macOS MPS). They were not checked with other models or other transformers versions.

## 0. Preparation — revision pinning and device
```python
import os
from pathlib import Path
os.environ.setdefault("HF_HOME", str(Path(".cache/hf").resolve()))   # it only takes effect if set **before** importing transformers (the same cache as the notebooks; delete the line if you like)

import numpy as np, torch
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
NAME = "facebook/esm2_t6_8M_UR50D"
REV = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"                      # pinned with the full sha (do not use a concatenated short prefix)

tok = AutoTokenizer.from_pretrained(NAME, revision=REV)
enc_model = EsmModel.from_pretrained(NAME, revision=REV, add_pooling_layer=False).to(DEVICE).eval()
mlm_model = EsmForMaskedLM.from_pretrained(NAME, revision=REV).to(DEVICE).eval()
```
- When opening by repository ID, transformers has pulled in the automatic-conversion files of `refs/pr/N` for repositories that have only a `.bin` (module 05). For a new model, it is safer to run `from_pretrained` on the **local path** downloaded with `huggingface_hub.snapshot_download(..., revision=...)`.
- Check the weight license and whether access approval is needed in the model card/metadata before use (module 06 §7).

## 1. Mean embedding — excluding padding and special tokens
```python
@torch.no_grad()
def embed_mean(seqs, layer=-1, batch=32):
    """A (d,) mean embedding per sequence. Padding and <cls>/<eos> are left out of the mean.
    layer: hidden_states index (0 = embedding-layer output, -1 = last layer)."""
    out = []
    for s in range(0, len(seqs), batch):
        enc = tok(list(seqs[s:s + batch]), return_tensors="pt", padding=True).to(DEVICE)
        res = enc_model(**enc, output_hidden_states=True)
        h = res.hidden_states[layer]                                   # (b, T, d)
        m = enc["attention_mask"].clone()
        m[:, 0] = 0                                                    # <cls>
        m[torch.arange(m.size(0)), enc["attention_mask"].sum(1) - 1] = 0   # <eos>
        out.append(((h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)).cpu())
    return torch.cat(out).numpy()

seqs = ["MKTAYIAKQRQISFVKSHFSRQ", "MKT", "GSHMLEDPVDAFLLRTKQALAESD"]
E_batch = embed_mean(seqs)
E_alone = np.vstack([embed_mean([s]) for s in seqs])
assert E_batch.shape == (3, enc_model.config.hidden_size)
assert np.allclose(E_batch, E_alone, atol=1e-4), "if the batch (with padding) and the single result differ, the mask handling is wrong"
print("hidden_size", E_batch.shape[1], "| max batch-vs-single difference", float(np.abs(E_batch - E_alone).max()))
```
- **Always check**: the result must be the same whether you feed a sequence alone or in a padded batch.
- Whether to include `<cls>`/`<eos>` in the mean, and which layer to use instead of the last, should be decided **before seeing the results** or chosen by validation inside the training data (module 07).

## 2. Layer choice
```python
with torch.no_grad():
    enc = tok(seqs[0], return_tensors="pt").to(DEVICE)
    hs = enc_model(**enc, output_hidden_states=True).hidden_states
n_layers = enc_model.config.num_hidden_layers
assert len(hs) == n_layers + 1                                          # embedding-layer output + one per layer
print("hidden_states", len(hs), "=", n_layers, "layers + embedding layer")
```
- In this environment (transformers 5.22.1, ESM-2 8M), `last_hidden_state` was exactly the same as `hidden_states[-1]` (maximum difference 0). It may differ with other models and versions, so check before using it. Notebook 02 used `last_hidden_state` when `layer=-1` and `hidden_states[layer]` otherwise.

## 3. Residue embedding at the variant position
```python
AAS = "ACDEFGHIKLMNPQRSTVWY"

def check_seq(seq, pos1=None):
    """Input checks: the 20 standard residues, the maximum length, the 1-based position range. (Position 0, negative or out-of-range positions silently pick a different token under Python indexing)"""
    assert set(seq) <= set(AAS), "characters other than the 20 standard ones (X, B, Z, U, O, etc.) must be handled separately"
    assert len(seq) <= enc_model.config.max_position_embeddings - 2, "exceeds the maximum length including <cls>/<eos>"
    if pos1 is not None:
        assert 1 <= pos1 <= len(seq), "pos1 is 1-based and must be in 1..L"

@torch.no_grad()
def residue_embedding(seq, pos1):
    """pos1: 1-based residue position. <cls> is index 0, so token index = pos1."""
    check_seq(seq, pos1)
    enc = tok(seq, return_tensors="pt").to(DEVICE)
    h = enc_model(**enc).last_hidden_state[0]
    return h[pos1].cpu().numpy()

seq = seqs[0]
assert tok.convert_ids_to_tokens(tok(seq)["input_ids"])[5] == seq[4]    # token 5 = residue 5 (1-based)
print(residue_embedding(seq, 5).shape)
for bad in (0, -1, len(seq) + 1):
    try:
        residue_embedding(seq, bad); raise RuntimeError("the check must not pass")
    except AssertionError:
        pass
```

## 4. Masked-marginal zero-shot score (single substitution)
```python
AA_IDS = torch.tensor(tok.convert_tokens_to_ids(list(AAS)), device=DEVICE)

@torch.no_grad()
def masked_marginal_table(seq):
    """An (L, 20) table: the log-probabilities of the 20 amino acids when each position is masked with <mask>."""
    check_seq(seq)
    ids = tok(seq, return_tensors="pt")["input_ids"].to(DEVICE)         # (1, L+2)
    L = len(seq)
    batch = ids.repeat(L, 1)
    batch[torch.arange(L), torch.arange(L) + 1] = tok.mask_token_id     # mask position i
    logits = mlm_model(input_ids=batch).logits
    lp = torch.log_softmax(logits[torch.arange(L), torch.arange(L) + 1].float(), dim=-1)
    return lp[:, AA_IDS].cpu().numpy()

def mm_score(table, seq, pos1, mt):
    check_seq(seq, pos1)
    wt = seq[pos1 - 1]
    return table[pos1 - 1, AAS.index(mt)] - table[pos1 - 1, AAS.index(wt)]

table = masked_marginal_table(seq)
assert table.shape == (len(seq), 20)
assert abs(mm_score(table, seq, 5, seq[4])) < 1e-6                     # wt → wt is 0 (a weak check: subtracting the same element from itself)
# Independent check: the value computed in one go with only position 5 masked must equal the row of the table
ids = tok(seq, return_tensors="pt")["input_ids"].to(DEVICE)
assert tok.convert_ids_to_tokens(ids[0, 5].item()) == seq[4]           # the token before masking = the residue at position 5
single = ids.clone(); single[0, 5] = tok.mask_token_id
with torch.no_grad():
    lp5 = torch.log_softmax(mlm_model(input_ids=single).logits[0, 5].float(), dim=-1)[AA_IDS].cpu().numpy()
assert np.allclose(table[4], lp5, atol=1e-4)
print("e.g. position 5", seq[4], "→ A:", round(float(mm_score(table, seq, 5, "A")), 3))
```
- The score is a **proxy** for "how plausible it is under the natural sequence distribution". The agreement with actual fitness differs by assay and protein, so confirm it with a benchmark such as ProteinGym (module 02 §4).
- For a long sequence, the implementation above puts L sequences on at once and uses a lot of memory. Notebook 02 processed the positions in chunks of 32.
- Summing the scores for multiple substitutions ignores epistasis (module 02 §4.1).

## 5. Common mistakes
| Symptom | Cause |
|---|---|
| The batch and single embeddings differ | Padding was included in the mean / the attention mask was missing |
| The variant position is off by one | Because of `<cls>`, token index = 1-based position. It was mistaken for 0-based |
| OOM on a long sequence | The cause was processing all the positions in one batch. Split it into chunks |
| Numerical differences of `log_softmax` on MPS | Compute after lifting to `.float()` |
| Different results on each re-run | The revision was not pinned, `model.eval()` was missing (dropout) |
| The cache location setting has no effect | The cause was setting `HF_HOME` **after** importing `transformers`/`huggingface_hub` (actually experienced when this file was first run) |
| The embedding cache is from a different model | Put the model ID, revision and layer in the cache file name |

<!-- i18n-source: resources/cheatsheets/01_embedding_recipes.md blob:fba069f6c3472ea5f1bb1851401a519aebb43535 -->
