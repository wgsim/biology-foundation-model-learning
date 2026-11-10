#!/bin/bash
# 웹 책(Jupyter Book 2 / MyST) 빌드. 저장소 원본은 건드리지 않고 .book_build/ 에 복사해 만든다.
# 실행(저장소 루트): bash book/build.sh          # 문서만 복사해 빌드 (노트북은 이미 실행된 복사본이 있으면 그것을 씀)
#                   EXEC=1 bash book/build.sh   # Mac .venv 로 실행 가능한 노트북을 실행해 출력을 싣는다 (원본이 바뀐 것만 다시 실행)
# 필요: .venv (노트북 실행), .venv-book (`uv venv --python 3.12 .venv-book && uv pip install --python .venv-book/bin/python jupyter-book`)
# 첫 빌드는 사이트 테마(book-theme)를 GitHub에서 내려받는다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B="$ROOT/.book_build"
cd "$ROOT"
mkdir -p "$B"

# 1) 문서·참조 파일 복사 (git이 추적하는 것만)
git ls-files '*.md' '*.py' '*.txt' '*.sbatch' | while read -r f; do
  mkdir -p "$B/$(dirname "$f")"; cp "$f" "$B/$f"
done
cp book/myst.yml "$B/myst.yml"
# 폴더를 가리키는 링크( ](00_orientation/) )는 책에서 그 폴더의 README로 보낸다 (복사본에서만 고침)
"$ROOT/.venv/bin/python" "$ROOT/book/stage_fix_links.py" "$B"

# 2) 노트북: 실행된 복사본이 없으면 원본(출력 없음)을 복사, EXEC=1이면 실행
SKIP_EXEC_LIST="02_protein/notebooks/04_esmfold_plddt_colab.ipynb 03_dna_genome/notebooks/03_enformer_tracks.ipynb"   # Colab·격리 환경 전용
git ls-files '*.ipynb' | while read -r nb; do
  out="$B/$nb"; mkdir -p "$(dirname "$out")"
  runnable=1; for s in $SKIP_EXEC_LIST; do [ "$s" = "$nb" ] && runnable=0; done
  if [ "${EXEC:-0}" = "1" ] && [ "$runnable" = "1" ] && { [ ! -f "$out.executed" ] || [ "$nb" -nt "$out.executed" ]; }; then
    echo "[exec] $nb"; t0=$(date +%s)
    if (cd "$(dirname "$nb")" && HF_HOME="$ROOT/.cache/hf" HF_HUB_OFFLINE=1 "$ROOT/.venv/bin/jupyter" nbconvert --to notebook --execute \
          --ExecutePreprocessor.timeout=7200 --output "$out" "$ROOT/$nb" >"$out.log" 2>&1); then
      touch "$out.executed"; echo "  ok $(( $(date +%s) - t0 ))s"
    else
      echo "  실패 (로그: $out.log) — 출력 없는 원본을 싣는다"; cp "$nb" "$out"
    fi
  elif [ ! -f "$out.executed" ]; then
    cp "$nb" "$out"
  fi
done

# 3) 빌드
cd "$B"
"$ROOT/.venv-book/bin/jupyter-book" build --html 2>&1 | grep -v "Duplicate identifier\|^   In files:" || true
echo "빌드 결과: $B/_build/html"
