# 치트시트 04 — 계산량·메모리·환경

모듈 06(노트북 03)과 SETUP.md에서 **측정하거나 확인한 것**과, 측정하지 않고 공식(근사)으로 어림한 것을 구분해 모았다. 표기: **측정** = 이 저장소에서 실행한 값, **공식** = 계산식이며 활성값 등을 뺀 어림, † = 확인하지 못한 서술. `python` 블록은 저장소 `.venv`에서 실행해 확인했다(2026-11-06).

## 1. 메모리 예산 (활성값 제외, 공식)
| 상황 | 바이트/매개변수 | 설명 |
|---|---|---|
| 추론, fp32 | 4 | 가중치 |
| 추론, bf16/fp16 | 2 | 가중치 |
| 전체 파인튜닝, 순수 fp32 AdamW | 16 | 가중치 4 + 기울기 4 + Adam 1·2차 모멘트 8 |
| 혼합 정밀도 전체 파인튜닝 | 16~20† | fp32 마스터 사본이 필요해 총 바이트는 순수 fp32와 비슷하거나 더 클 수 있다(반정밀도 사본·기울기 보관 방식에 따라)†. 이득은 주로 연산 속도와 활성값 메모리 |
| LoRA (동결 가중치 fp32) | 동결 4 + 학습분 16 | 학습 매개변수가 전체의 1% 미만이면 거의 4P |
| LoRA (동결 가중치 bf16) | 동결 2 + 학습분 16 | 동결 가중치만 bf16, LoRA 매개변수는 fp32 |

```python
def budget_gb(n_params, mode="full_fp32", n_trainable=None):
    """활성값을 뺀 메모리 어림(GB, 1 GB = 1e9 바이트). 측정값이 아니라 공식이다."""
    if mode == "infer_fp32": b = 4 * n_params
    elif mode == "infer_bf16": b = 2 * n_params
    elif mode == "full_fp32": b = 16 * n_params
    elif mode in ("lora_fp32", "lora_bf16"):
        frozen = (4 if mode == "lora_fp32" else 2) * (n_params - n_trainable)
        b = frozen + 16 * n_trainable
    else: raise ValueError(mode)
    return b / 1e9

assert abs(budget_gb(650e6, "full_fp32") - 10.4) < 0.01                   # 모듈 06 §5: 650M 전체 파인튜닝 ≈ 10.4 GB
assert abs(budget_gb(650e6, "lora_fp32", 1.35e6) - 2.6) < 0.1             # LoRA(q,v r=8) ≈ 2.6 GB
assert abs(budget_gb(650e6, "lora_bf16", 1.35e6) - 1.3) < 0.1             # 동결 가중치 bf16 ≈ 1.3 GB
print({m: round(budget_gb(650e6, m, 1.35e6), 2) for m in ["infer_fp32", "infer_bf16", "full_fp32", "lora_fp32", "lora_bf16"]})
```
- **활성값**은 배치·서열 길이·층 수에 따라 크게 달라져 위 공식에 없다. 작은 모델·짧은 서열에서도 설정에 따라 달라지므로 **해당 설정에서 측정**하고, 긴 서열에서는 지배적일 수 있다 (아래 §3).
- 8M·35M 모델에서는 방법 간 차이가 0.1~0.5 GB라 장비 차이에 묻힌다 (모듈 06).

## 2. 연산량 어림
- 트랜스포머 학습 연산량 $C \approx 6ND$ (N 매개변수, D 토큰)†, 순전파 $\approx 2ND$†. 모듈 06은 이 식과 Mac MPS 측정 처리량으로 시간을 어림했다.
- 시간 어림 = 연산량 ÷ **같은 장비·같은 구현에서 측정한 처리량**(FLOP/s 또는 token/s; 단위를 맞춘다). 다른 장비로 옮길 때는 다시 측정한다. 모듈 04에서 Colab T4가 Mac MPS보다 빠르다고 가정할 수 없음을 측정했다: Geneformer V2-104M 추론(fp32)이 T4 788초, Mac 673초였고, CPU 학습인 scVI는 Colab 2코어에서 757초, Mac에서 134초였다. 단일 실행이며 반복 측정이 아니다.

## 3. 측정값 (Mac MPS, ESM-2 35M, 배치 4, forward+backward 중앙값 3회)
메모리는 MPS **드라이버 할당량**이라 peak가 아닌 *대리 지표*다. 같은 설정의 시간이 실행마다 약 10% 흔들렸다.

| 서열 길이 | eager fp32 | sdpa fp32 | sdpa bf16 autocast | sdpa fp32 + 체크포인팅 |
|---|---|---|---|---|
| 128 | 0.11 s · 0.63 GB | 0.11 s · 0.63 GB | 0.12 s · 0.68 GB | 0.13 s · 0.41 GB |
| 512 | 0.41 s · 3.50 GB | 0.41 s · 3.50 GB | 0.47 s · 2.45 GB | 0.51 s · 1.35 GB |
| 1024 | 1.00 s · 7.84 GB | 0.98 s · 7.85 GB | 1.11 s · 7.90 GB | 1.28 s · 2.48 GB |

(모듈 06 README §5-C의 표와 같다.) 읽는 법:
- 서열 길이 128 → 1024에서 메모리가 약 12배 늘었다 → 활성값이 지배한다. 길이 1024에서 체크포인팅은 메모리를 −68%, 시간을 +31% 바꿨다.
- 이 Mac에서 `sdpa`와 `eager`의 대리 메모리는 같았다. **MPS에는 FlashAttention 커널이 없다**†. CUDA에서의 효과는 측정하지 않았다.
- bf16 autocast는 이 장비에서 시간 이득이 없었다 (+9~20%).

```python
import gc
import numpy as np, torch

def mps_mem_gb():
    """MPS 드라이버 할당량(GB). peak가 아니라 캐시를 포함한 대리 지표."""
    if not torch.backends.mps.is_available():
        return float("nan")
    return torch.mps.driver_allocated_memory() / 1e9

gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()             # 측정 전에 정리하지 않으면 직전 작업의 버퍼가 섞인다 (모듈 06에서 겪음)
before = mps_mem_gb()
x = torch.zeros(256, 1024, 1024, device="mps" if torch.backends.mps.is_available() else "cpu")   # 1 GiB(=1.07 GB, 십진 GB 기준) fp32
after = mps_mem_gb()
print("MPS 대리 지표 변화(GB):", None if np.isnan(after) else round(after - before, 2))
del x; gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

## 4. 환경별로 되는 것 / 안 되는 것 (이 저장소에서 확인한 범위)
| 환경 | 확인한 것 | 확인하지 못했거나 안 되는 것 |
|---|---|---|
| 🟢 Mac (MPS/CPU) | ESM-2 8M~650M 임베딩·zero-shot, NT 500M 임베딩, Geneformer, 소형 LoRA/전체 파인튜닝, SAE, DDP 동등성(CPU 2프로세스) | FlashAttention(MPS에 없음†), 대형 모델(Evo 2 7B 이상), CUDA 전용 패키지 |
| 🟡 Colab T4 | 모듈 04 노트북 01~03을 오류 없이 실행(총 약 56분, 약 1 compute unit) | T4가 Mac보다 빠르다는 보장 없음(위 §2). 모듈 05·06은 Colab 미검증. scVI 같은 확률적 학습은 수치가 ±0.01 정도 달라질 수 있다 |
| 🔴 SLURM | `slurm_ddp.sbatch`는 `bash -n` 문법 점검만 | 실제 클러스터 실행, NCCL, 다중 노드 — **실행하지 못했다** |
| 가중치가 큰 모델 | Evo 2: `evo2_1b_base`·40B는 FP8(Transformer Engine)·Hopper GPU 필요, `evo2_7b`는 bf16 가능(공식 README, 2026-11 확인). ESMFold2 가중치 약 27 GB(모델 카드) | 이 저장소는 둘 다 직접 실행하지 않았다 |

## 5. 장비 특이점 (겪은 것)
| 증상 | 원인/해결 |
|---|---|
| `torchrun --standalone`이 300초 뒤 타임아웃 (macOS) | 호스트 이름이 `1.0.0.127.in-addr.arpa`로 풀림 → `--local-addr=127.0.0.1` |
| 메모리 측정이 직전 실행의 영향을 받음 | `gc.collect()` + `torch.mps.empty_cache()` 후 측정 |
| 시간 측정이 흔들림 | 다른 GPU/CPU 작업과 겹치지 않게 단독 실행, 중앙값 3회 이상 |
| 짧은 데이터에서 DDP가 2 step 만에 끝남 | 에폭 루프와 `sampler.set_epoch(epoch)` 사용 |
| 패키지 버전이 달라 결과가 다름 | `requirements/base.lock.txt`로 재현, 모델은 리비전 고정 |
| 큰 파일이 git에 들어감 | `.cache/`(가중치·데이터)는 `.gitignore` — 커밋 전 `git status` 확인 |

## 6. 실행 전 체크
- [ ] 필요한 다운로드(파일·출처·크기)를 적고 승인을 받았다
- [ ] 메모리 공식으로 어림하고, 활성값(서열 길이·배치)을 따로 점검했다
- [ ] 시간·메모리를 보고할 때 *대리 지표*임과 장비를 적었다
- [ ] 실행하지 못한 환경(클러스터, 다른 GPU)을 "미검증"으로 적었다
