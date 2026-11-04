# 환경 설정

실행 등급: 🟢 Mac(CPU/MPS) · 🟡 Colab(GPU) · 🔴 기관 GPU 클러스터

## 1. Mac 로컬 (🟢)
```bash
# uv 설치 여부 확인 (없으면 https://docs.astral.sh/uv/ 참고)
uv --version

uv venv .venv
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
- 노트북 실행이 검증되면 `uv pip freeze > requirements/base.lock.txt`로 버전을 고정한다.
- 모델/데이터셋은 노트북 상단에 ID와 확인 날짜를 기록한다.
