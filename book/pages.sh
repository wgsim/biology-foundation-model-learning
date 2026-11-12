#!/bin/bash
# GitHub Pages에 올릴 사이트를 만든다. 저장소 원본은 건드리지 않는다.
# 실행(저장소 루트): bash book/pages.sh <GitHub 저장소 이름>
#   1) BASE_URL=/<저장소 이름> 으로 웹 책을 빌드한다. 프로젝트 페이지는 https://<아이디>.github.io/<저장소 이름>/ 아래에 놓인다.
#   2) 결과물에 로컬 경로·계정 이름·비밀값 패턴이 남아 있는지 검사한다. 하나라도 나오면 중단한다.
#   3) .pages_site/ 에 .nojekyll 과 함께 모으고, 별도 git 저장소(gh-pages 브랜치)로 커밋한다.
# 이 스크립트는 push하지 않는다. 마지막에 출력되는 명령을 확인한 뒤 직접 실행한다.
# 필요: build.sh와 같다. 노트북 출력은 이미 실행된 복사본(.book_build/**/*.ipynb)을 쓴다.
set -euo pipefail
REPO="${1:?사용법: bash book/pages.sh <GitHub 저장소 이름>}"
case "$REPO" in */*) echo "저장소 이름에는 /를 쓸 수 없다: $REPO"; exit 1;; esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B="$ROOT/.book_build"
H="$B/_build/html"
S="$ROOT/.pages_site"
TRASH=/opt/homebrew/opt/trash/bin/trash
cd "$ROOT"

# 1) 하위 경로용 빌드
BASE_URL="/$REPO" bash "$ROOT/book/build.sh"
[ -f "$H/index.html" ] || { echo "중단: $H/index.html 이 없다. 빌드가 실패했다."; exit 1; }
grep -q "/$REPO/_assets/" "$H/index.html" || { echo "중단: 페이지가 /$REPO/ 아래 경로를 가리키지 않는다. BASE_URL이 적용되지 않았다."; exit 1; }

# 2) 공개 전 검사
user="$(basename "$HOME")"
if grep -rIlE -e "$HOME" -e "/Users/" -e "/private/var" -e "/var/folders" -e "$user" "$H"; then
  echo "중단: 위 파일에 로컬 경로나 계정 이름이 남아 있다."; exit 1
fi
if grep -rIlE --include='*.html' --include='*.json' --include='*.ipynb' --include='*.md' --include='*.py' \
     '(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-|AIza[0-9A-Za-z_-]{30,})' "$H"; then
  echo "중단: 위 파일에 비밀값 패턴이 있다."; exit 1
fi
echo "[검사] 로컬 경로·계정 이름·비밀값 패턴 없음"

# 3) 게시용 폴더. .nojekyll이 없으면 GitHub가 밑줄로 시작하는 _assets 폴더를 서빙하지 않는다.
[ -e "$S" ] && "$TRASH" "$S"
mkdir "$S"
cp -R "$H"/. "$S"/
touch "$S/.nojekyll"
git -C "$S" init -q -b gh-pages
git -C "$S" add -A
git -C "$S" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" commit -q -m "Publish web book"
echo "게시용 사이트: $S ($(find "$S" -name '*.html' | wc -l | tr -d ' ')쪽, $(du -sh "$S" | cut -f1))"
echo "저장소를 만든 뒤 직접 실행한다:"
echo "  git -C .pages_site push --force https://github.com/<아이디>/$REPO.git gh-pages"
echo "GitHub 저장소 Settings > Pages 에서 Source를 'Deploy from a branch', 브랜치를 gh-pages / (root)로 고른다."
