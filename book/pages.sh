#!/bin/bash
# GitHub Pages에 올릴 사이트를 만든다. 저장소 원본은 건드리지 않는다.
# 실행(저장소 루트): bash book/pages.sh <GitHub 저장소 이름>
#   1) BASE_URL=/<저장소 이름> 으로 웹 책을 빌드한다. 프로젝트 페이지는 https://<아이디>.github.io/<저장소 이름>/ 아래에 놓인다.
#   2) 빌드가 완전한지(쪽 수), 노트북 실행본이 최신인지 확인한다.
#   3) 결과물에 로컬 경로·계정 이름·비밀값 패턴이 남아 있는지 검사한다. 일치하거나 검사가 오류로 끝나면 중단한다.
#   4) .pages_site/ 에 .nojekyll, 라이선스 파일과 함께 모으고, 별도 git 저장소(gh-pages 브랜치)로 커밋한다.
#      커밋 작성자 이메일은 GitHub noreply 주소여야 한다. 아니면 중단한다.
# 이 스크립트는 push하지 않는다. 마지막에 출력되는 명령을 확인한 뒤 직접 실행한다.
# 필요: build.sh와 같다. 노트북 출력은 이미 실행된 복사본(.book_build/**/*.ipynb)을 쓴다.
set -euo pipefail
REPO="${1:?사용법: bash book/pages.sh <GitHub 저장소 이름>}"
[[ "$REPO" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "중단: 저장소 이름은 영문자·숫자·.·_·- 만 쓸 수 있고 영문자나 숫자로 시작해야 한다: $REPO"; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B="$ROOT/.book_build"
H="$B/_build/html"
S="$ROOT/.pages_site"
TRASH=/opt/homebrew/opt/trash/bin/trash
[ -x "$TRASH" ] || { echo "중단: $TRASH 가 없다. 이 스크립트는 rm 대신 trash를 쓴다."; exit 1; }
cd "$ROOT"

# 공개 커밋에 개인 이메일이 실리지 않도록, 작성자 이메일이 GitHub noreply 주소인지 먼저 확인한다.
email="$(git config user.email || true)"
case "$email" in
  *@users.noreply.github.com) ;;
  *) echo "중단: git user.email이 GitHub noreply 주소(…@users.noreply.github.com)가 아니다."; exit 1;;
esac

# grep을 부른다. 일치하면 1, 일치가 없으면 0을 돌려주고, grep 자체가 오류(상태 2 이상)로 끝나면 스크립트를 중단한다.
scan() {
  local rc=0
  grep "$@" || rc=$?
  case "$rc" in
    0) return 1;;
    1) return 0;;
    *) echo "중단: 검사(grep)가 오류로 끝났다 (상태 $rc)."; exit 1;;
  esac
}

# 1) 하위 경로용 빌드
BASE_URL="/$REPO" bash "$ROOT/book/build.sh"
[ -f "$H/index.html" ] || { echo "중단: $H/index.html 이 없다. 빌드가 실패했다."; exit 1; }
grep -qF "/$REPO/_assets/" "$H/index.html" || { echo "중단: 페이지가 /$REPO/ 아래 경로를 가리키지 않는다. BASE_URL이 적용되지 않았다."; exit 1; }

# 2) 빌드가 완전한지 확인한다. build.sh는 빌드 실패를 숨기므로 쪽 수를 목차와 맞춰 본다.
want="$(grep -cE '^\s*- file:' "$ROOT/book/myst.yml")"
have="$(find "$H" -name index.html | wc -l | tr -d ' ')"
[ "$want" = "$have" ] || { echo "중단: 목차는 ${want}쪽인데 HTML은 ${have}쪽이다. 빌드가 불완전하다."; exit 1; }
# 원본 노트북이 실행본보다 새로우면(코드가 바뀐 경우) 오래된 출력이 실린다. EXEC=1로 다시 실행해야 한다.
stale=0
while read -r nb; do
  if [ -f "$B/$nb.executed" ] && [ "$ROOT/$nb" -nt "$B/$nb.executed" ]; then echo "오래된 실행본: $nb"; stale=1; fi
done < <(git ls-files '*.ipynb')
[ "$stale" = 0 ] || { echo "중단: 코드가 바뀐 노트북이 있다. EXEC=1 bash book/build.sh 로 다시 실행한 뒤 시도한다."; exit 1; }

# 3) 공개 전 검사. 계정 이름은 바이너리 파일까지 포함해 문자열 그대로(-F) 찾는다.
user="$(basename "$HOME")"
scan -rlaF -e "$HOME" -e "/Users/" -e "/private/var" -e "/var/folders" -e "$user" "$H" \
  || { echo "중단: 위 파일에 로컬 경로나 계정 이름이 남아 있다."; exit 1; }
scan -rIlE --include='*.html' --include='*.json' --include='*.ipynb' --include='*.md' --include='*.py' \
     '(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-|AIza[0-9A-Za-z_-]{30,})' "$H" \
  || { echo "중단: 위 파일에 비밀값 패턴이 있다."; exit 1; }
echo "[검사] 로컬 경로·계정 이름·비밀값 패턴 없음"

# 4) 게시용 폴더. .nojekyll이 없으면 GitHub가 밑줄로 시작하는 _assets 폴더를 서빙하지 않는다.
if [ -e "$S" ]; then "$TRASH" "$S"; fi
mkdir "$S"
cp -R "$H"/. "$S"/
touch "$S/.nojekyll"
# robots.txt와 sitemap은 빌드가 자리표시자 도메인(http://pre.render)으로 만든 것이라 올리지 않는다.
for f in robots.txt sitemap.xml sitemap_style.xsl; do
  if [ -e "$S/$f" ]; then "$TRASH" "$S/$f"; fi
done
# 이 사이트의 라이선스와, 번들된 제3자 구성요소 안내를 함께 둔다.
cp "$ROOT/LICENSE" "$S/LICENSE.txt"
cp "$ROOT/LICENSE-docs.md" "$S/LICENSE-docs.txt"
cat > "$S/NOTICE.txt" <<'EOF'
This site was generated with Jupyter Book 2 (MyST).
The JavaScript and CSS bundles (for example under /_assets and the *.thebe-*.js files) are
third-party open-source components of those projects and keep their own licenses.
See https://github.com/jupyter-book/mystmd and https://github.com/jupyter-book/myst-theme.
The licenses of the curriculum itself are in LICENSE.txt (code) and LICENSE-docs.txt (prose).
EOF
git -C "$S" init -q -b gh-pages
git -C "$S" add -A
git -C "$S" -c user.name="$(git config user.name)" -c user.email="$email" commit -q -m "Publish web book"
echo "게시용 사이트: $S ($(find "$S" -name '*.html' | wc -l | tr -d ' ')쪽, $(du -sh "$S" | cut -f1))"
echo "저장소를 만든 뒤 직접 실행한다:"
echo "  git -C .pages_site push --force https://github.com/<아이디>/$REPO.git gh-pages"
echo "GitHub 저장소 Settings > Pages 에서 Source를 'Deploy from a branch', 브랜치를 gh-pages / (root)로 고른다."
