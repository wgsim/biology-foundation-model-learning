# 치트시트 01 — 임베딩·zero-shot 점수 레시피 (단백질 LM)

모듈 02의 노트북 01·02에서 쓴 패턴을 줄였다. 이 파일의 `python` 코드 블록은 **저장소 환경(`.venv`, ESM-2 8M 캐시)에서 순서대로 실행해 확인했다** (2026-11-06, transformers 5.22.1, torch 2.15.0, macOS MPS). 다른 모델·다른 transformers 버전에서는 확인하지 않았다.

## 0. 준비 — 리비전 고정과 장치
```python
import os
from pathlib import Path
os.environ.setdefault("HF_HOME", str(Path(".cache/hf").resolve()))   # transformers import **전에** 설정해야 적용된다 (노트북과 같은 캐시; 원하면 줄 삭제)

import numpy as np, torch
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
NAME = "facebook/esm2_t6_8M_UR50D"
REV = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"                      # 전체 sha로 고정 (짧은 접두어를 이어 쓰지 말 것)

tok = AutoTokenizer.from_pretrained(NAME, revision=REV)
enc_model = EsmModel.from_pretrained(NAME, revision=REV, add_pooling_layer=False).to(DEVICE).eval()
mlm_model = EsmForMaskedLM.from_pretrained(NAME, revision=REV).to(DEVICE).eval()
```
- 저장소 ID로 열 때 `.bin`만 있는 저장소는 transformers가 `refs/pr/N`의 자동 변환 파일을 끌어온 적이 있다(모듈 05). 새 모델은 `huggingface_hub.snapshot_download(..., revision=...)`로 받은 **로컬 경로**에서 `from_pretrained`하는 편이 안전하다.
- 가중치 라이선스·접근 승인 여부는 사용 전에 모델 카드/메타데이터로 확인한다 (모듈 06 §7).

## 1. 평균 임베딩 — 패딩과 특수 토큰 제외
```python
@torch.no_grad()
def embed_mean(seqs, layer=-1, batch=32):
    """서열마다 (d,) 평균 임베딩. 패딩과 <cls>/<eos>는 평균에서 뺀다.
    layer: hidden_states 인덱스 (0 = 임베딩 층 출력, -1 = 마지막 층)."""
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
assert np.allclose(E_batch, E_alone, atol=1e-4), "배치(패딩 포함)와 단독 결과가 다르면 마스크 처리가 틀린 것"
print("hidden_size", E_batch.shape[1], "| 배치-단독 최대 차이", float(np.abs(E_batch - E_alone).max()))
```
- **항상 점검**: 같은 서열을 단독으로 넣을 때와 패딩된 배치로 넣을 때 결과가 같아야 한다.
- 평균에 `<cls>`/`<eos>`를 넣을지, 마지막 층 대신 어느 층을 쓸지는 **결과를 보기 전에** 정하거나 학습 데이터 안의 검증으로 고른다 (모듈 07).

## 2. 층 선택
```python
with torch.no_grad():
    enc = tok(seqs[0], return_tensors="pt").to(DEVICE)
    hs = enc_model(**enc, output_hidden_states=True).hidden_states
n_layers = enc_model.config.num_hidden_layers
assert len(hs) == n_layers + 1                                          # 임베딩 층 출력 + 층마다 하나
print("hidden_states", len(hs), "개 =", n_layers, "층 + 임베딩 층")
```
- 이 환경(transformers 5.22.1, ESM-2 8M)에서는 `last_hidden_state`가 `hidden_states[-1]`과 정확히 같았다(최대 차이 0). 다른 모델·버전에서는 다를 수 있으니 확인하고 쓴다. 노트북 02는 `layer=-1`일 때 `last_hidden_state`, 아니면 `hidden_states[layer]`를 썼다.

## 3. 변이 위치의 잔기 임베딩
```python
AAS = "ACDEFGHIKLMNPQRSTVWY"

def check_seq(seq, pos1=None):
    """입력 점검: 표준 20종, 최대 길이, 1-based 위치 범위. (0·음수·범위 밖 위치는 파이썬 인덱싱이 조용히 다른 토큰을 고른다)"""
    assert set(seq) <= set(AAS), "표준 20종 이외의 문자(X, B, Z, U, O 등)는 따로 처리해야 한다"
    assert len(seq) <= enc_model.config.max_position_embeddings - 2, "<cls>/<eos>를 포함한 최대 길이 초과"
    if pos1 is not None:
        assert 1 <= pos1 <= len(seq), "pos1은 1-based이며 1..L 이어야 한다"

@torch.no_grad()
def residue_embedding(seq, pos1):
    """pos1: 1-based 잔기 위치. <cls>가 인덱스 0이므로 토큰 인덱스 = pos1."""
    check_seq(seq, pos1)
    enc = tok(seq, return_tensors="pt").to(DEVICE)
    h = enc_model(**enc).last_hidden_state[0]
    return h[pos1].cpu().numpy()

seq = seqs[0]
assert tok.convert_ids_to_tokens(tok(seq)["input_ids"])[5] == seq[4]    # 토큰 5 = 잔기 5 (1-based)
print(residue_embedding(seq, 5).shape)
for bad in (0, -1, len(seq) + 1):
    try:
        residue_embedding(seq, bad); raise RuntimeError("점검이 통과해서는 안 된다")
    except AssertionError:
        pass
```

## 4. Masked-marginal zero-shot 점수 (단일 치환)
```python
AA_IDS = torch.tensor(tok.convert_tokens_to_ids(list(AAS)), device=DEVICE)

@torch.no_grad()
def masked_marginal_table(seq):
    """(L, 20) 표: 각 위치를 <mask>로 가렸을 때 20개 아미노산의 log-확률."""
    check_seq(seq)
    ids = tok(seq, return_tensors="pt")["input_ids"].to(DEVICE)         # (1, L+2)
    L = len(seq)
    batch = ids.repeat(L, 1)
    batch[torch.arange(L), torch.arange(L) + 1] = tok.mask_token_id     # 위치 i를 가림
    logits = mlm_model(input_ids=batch).logits
    lp = torch.log_softmax(logits[torch.arange(L), torch.arange(L) + 1].float(), dim=-1)
    return lp[:, AA_IDS].cpu().numpy()

def mm_score(table, seq, pos1, mt):
    check_seq(seq, pos1)
    wt = seq[pos1 - 1]
    return table[pos1 - 1, AAS.index(mt)] - table[pos1 - 1, AAS.index(wt)]

table = masked_marginal_table(seq)
assert table.shape == (len(seq), 20)
assert abs(mm_score(table, seq, 5, seq[4])) < 1e-6                     # wt → wt 는 0 (약한 점검: 같은 원소끼리 뺀 값)
# 독립 점검: 위치 5만 따로 가려 한 번에 계산한 값이 표의 행과 같아야 한다
ids = tok(seq, return_tensors="pt")["input_ids"].to(DEVICE)
assert tok.convert_ids_to_tokens(ids[0, 5].item()) == seq[4]           # 가리기 전 토큰 = 위치 5의 잔기
single = ids.clone(); single[0, 5] = tok.mask_token_id
with torch.no_grad():
    lp5 = torch.log_softmax(mlm_model(input_ids=single).logits[0, 5].float(), dim=-1)[AA_IDS].cpu().numpy()
assert np.allclose(table[4], lp5, atol=1e-4)
print("예: 위치 5", seq[4], "→ A:", round(float(mm_score(table, seq, 5, "A")), 3))
```
- 점수는 "자연 서열 분포에서 얼마나 그럴듯한가"의 **대리 지표**다. 실제 fitness와의 일치는 assay·단백질마다 달라 ProteinGym 같은 벤치마크로 확인한다 (모듈 02 §4).
- 서열이 길면 위 구현은 L개 서열을 한 번에 올려 메모리가 많이 든다. 노트북 02는 위치를 32개씩 나눠 처리했다.
- 다중 치환의 합산은 상위성을 무시한다 (모듈 02 §4.1).

## 5. 흔한 실수
| 증상 | 원인 |
|---|---|
| 배치와 단독 임베딩이 다름 | 패딩을 평균에 포함 / attention mask 누락 |
| 변이 위치가 한 칸 어긋남 | `<cls>` 때문에 토큰 인덱스 = 1-based 위치. 0-based로 착각 |
| 서열 길이가 긴데 OOM | 위치 전체를 한 배치로 처리 → 청크로 나눈다 |
| MPS에서 `log_softmax` 수치 차이 | `.float()`로 올려 계산 |
| 재실행마다 다른 결과 | 리비전 미고정, `model.eval()` 누락(드롭아웃) |
| 캐시 위치 설정이 안 먹음 | `HF_HOME`을 `transformers`/`huggingface_hub` import **뒤에** 설정 (이 파일을 처음 돌릴 때 실제로 겪음) |
| 임베딩 캐시가 다른 모델의 것 | 캐시 파일명에 모델 ID·리비전·층을 넣는다 |
