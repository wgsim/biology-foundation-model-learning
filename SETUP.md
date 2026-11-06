# 환경 설정

실행 등급: 🟢 Mac(CPU/MPS) · 🟡 Colab(GPU) · 🔴 기관 GPU 클러스터

## 1. Mac 로컬 (🟢)
```bash
# uv 설치 여부 확인 (없으면 https://docs.astral.sh/uv/ 참고)
uv --version

uv venv --python 3.12 .venv      # 3.12에서 base.txt 설치와 모듈 01 노트북 실행을 확인함 (시스템 Python 3.15는 미시도)
source .venv/bin/activate
uv pip install -r requirements/base.txt
python -m ipykernel install --user --name bio-fm
```

MPS 사용 가능 여부 확인:
```python
import torch
print(torch.backends.mps.is_available())
```
- MPS에서 일부 연산이 미지원이면 `PYTORCH_ENABLE_MPS_FALLBACK=1` 환경변수로 CPU 폴백을 허용한다.
- 🟢 노트북은 소형 체크포인트(예: ESM-2 8M~35M급)만 사용하도록 설계한다.

## 2. Google Colab (🟡)
노트북 첫 셀에서:
```python
!pip install -q -r https://raw.githubusercontent.com/<your-account>/<repo>/main/requirements/colab.txt
```
또는 저장소를 clone한 뒤 `requirements/colab.txt`를 설치한다. 런타임 유형은 GPU(T4 이상)로 설정.

## 3. 기관 클러스터 (🔴)
Assumption: SLURM 스케줄러, `module` 시스템 사용. 실제 파티션명·모듈명은 기관마다 다르므로 확인 후 수정한다.
```bash
#!/bin/bash
#SBATCH --job-name=bio-fm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

source .venv/bin/activate
python train.py
```
`flash-attn` 등 CUDA 의존 패키지는 클러스터의 CUDA/컴파일러 버전에 맞춰 개별 설치한다.

## 4. 재현성
- `requirements/base.lock.txt`: 모듈 01~06 노트북 실행을 확인한 환경(Python 3.12.13, macOS arm64)의 `uv pip freeze` 결과 (모듈 04에서 `harmonypy`, `scvi-tools` 등 22개 패키지, 모듈 05에서 `rdkit`, 모듈 06에서 `peft` 추가, 기존 패키지 버전 변경 없음). 정확히 재현하려면 `uv pip install -r requirements/base.lock.txt`.
- 이후 모듈에서 패키지를 추가하면 노트북 실행 확인 후 lock 파일을 갱신한다.
- 모델/데이터셋은 노트북 상단에 ID와 확인 날짜를 기록한다.

## 5. 모듈 03 노트북 03(Enformer) 격리 환경
`enformer-pytorch`는 `transformers==4.56.2`로 고정하므로 기본 `.venv`(transformers 5.x)에 설치하면 다른 노트북이 깨질 수 있다. 별도 환경을 만든다.
```bash
uv venv --python 3.12 .venv-enformer
uv pip install --python .venv-enformer/bin/python -r requirements/enformer.txt
.venv-enformer/bin/jupyter lab   # 이 환경의 커널로 03_enformer_tracks.ipynb 실행
```
- `.venv-enformer/`는 git에서 제외된다. 검증 환경: Python 3.12, torch 2.15.0, transformers 4.56.2.
- 모듈 03 노트북 01·02는 기본 `.venv`에서 실행한다.
- 모듈 03에서 내려받는 모델·데이터 중 **Nucleotide Transformer v1 가중치는 CC BY-NC-SA 4.0(비상업)**이다.
