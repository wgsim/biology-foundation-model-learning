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
또는 저장소를 clone한 뒤 `requirements/colab.txt`를 설치한다. 런타임 유형은 GPU(T4 이상)로 설정한다.

**Colab T4에서 확인한 것** (2026-11-06, GPU 15.6 GB, vCPU 2개, RAM 약 12 GB, Python 3.13, torch 2.14.1, transformers 5.22.1):
- 이미지에 `Bio`·`rdkit`·`peft`가 없다. `pip install biopython==1.88 rdkit==2026.9.1 peft==0.21.2`로 설치한다.
- 이미지의 `torchao 0.10.0`이 `peft 0.21.2`와 충돌한다(`ImportError: ... torchao`). `pip uninstall -y torchao` 후 커널을 다시 시작한다.
- `.cache/`가 없으므로 모듈 05·06·07 노트북이 가정하는 캐시(ProteinGym v1.3, 모듈 05의 UniProt 표, 모듈 06/03의 ESM-2 150M·650M `config.json`)를 먼저 받거나 올려야 한다. 모듈 02 노트북 01이 ProteinGym을 받고, 05 노트북 02가 UniProt 표를 받는다.
- 메모리: ESMFold v1은 가중치를 CPU로 먼저 불러오면 RAM 12 GB를 넘겨 커널이 죽을 수 있다(모듈 02 노트북 04는 `device_map`으로 GPU에 바로 올린다).
- 노트북별 Colab 실행 시간과 Mac과의 수치 차이는 각 모듈 README(05 §9, 06 §8, 07 §5)에 있다.

## 3. 기관 클러스터 (🔴)
Assumption: SLURM 스케줄러와 `module` 시스템을 사용한다. 실제 파티션명·모듈명은 기관마다 다르므로 확인 후 수정한다.
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
- `requirements/base.lock.txt`: 모듈 01~06 노트북 실행을 확인한 환경(Python 3.12.13, macOS arm64)의 `uv pip freeze` 결과다. 모듈 04에서 `harmonypy`, `scvi-tools` 등 22개 패키지, 모듈 05에서 `rdkit`, 모듈 06에서 `peft`를 추가했고 기존 패키지 버전은 바뀌지 않았다. 정확히 재현하려면 `uv pip install -r requirements/base.lock.txt`를 실행한다.
- 이후 모듈에서 패키지를 추가하면 노트북 실행 확인 후 lock 파일을 갱신한다.
- 모델/데이터셋은 노트북 상단에 ID와 확인 날짜를 기록한다.

## 5. 모듈 03 노트북 03(Enformer) 격리 환경
`enformer-pytorch`는 `transformers==4.56.2`로 고정하므로 기본 `.venv`(transformers 5.x)에 설치하면 다른 노트북이 깨질 수 있다. 별도 환경을 만든다.
```bash
uv venv --python 3.12 .venv-enformer
uv pip install --python .venv-enformer/bin/python -r requirements/enformer.txt
.venv-enformer/bin/jupyter lab   # 이 환경의 커널로 03_enformer_tracks.ipynb 실행
```
- `.venv-enformer/`는 git에서 제외된다. 검증 환경은 Python 3.12, torch 2.15.0, transformers 4.56.2이다.
- 모듈 03 노트북 01·02는 기본 `.venv`에서 실행한다.
- 모듈 03에서 내려받는 모델·데이터 중 **Nucleotide Transformer v1 가중치는 CC BY-NC-SA 4.0(비상업)**이다.

## 6. 웹 책으로 보기 (선택)
Markdown·노트북을 내용 변경 없이 목차·검색이 있는 웹 책(Jupyter Book 2 / MyST)으로 만든다. 원본은 건드리지 않고 `.book_build/`(git 제외)에 복사해 빌드한다.
```bash
uv venv --python 3.12 .venv-book
uv pip install --python .venv-book/bin/python jupyter-book   # 검증: jupyter-book 2.1.7, Node 필요
bash book/build.sh             # 문서만 빌드 (노트북은 출력 없이 실림)
EXEC=1 bash book/build.sh      # Mac .venv로 실행 가능한 노트북 17개를 실행해 출력·그림을 싣는다 (바뀐 것만 다시 실행)
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory .book_build/_build/html   # http://localhost:8765
```
- 첫 빌드는 사이트 테마(`book-theme`)를 GitHub에서 내려받는다(네트워크 필요).
- 문서를 추가하면 `book/myst.yml`의 `toc`에도 추가한다. 목차에 없는 문서는 책에 나오지 않는다.
- Colab·격리 환경 전용 노트북(02/04 ESMFold, 03/03 Enformer)은 실행하지 않고 출력 없이 싣는다.
- `book/stage_fix_links.py`가 복사본에서 폴더 링크와 한글 제목 앵커를 책에 맞게 고친다(MyST는 제목 id에서 한글을 버린다).

