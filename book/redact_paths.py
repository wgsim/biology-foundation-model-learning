"""실행된 노트북 복사본의 출력에서 로컬 절대경로를 지운다.

웹 책을 공개하기 전에 계정 이름이 담긴 경로가 HTML에 실리지 않게 하려는 처리다.
저장소 루트 경로는 `<repo>`로, 홈 디렉터리는 `~`로 바꾼다. 여러 번 실행해도 결과가 같다.

사용: redact_paths.py <book_build 폴더> <저장소 루트> <홈 디렉터리>
"""
import pathlib
import sys

B, ROOT, HOME = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
files = hits = 0
for p in sorted(B.rglob("*.ipynb")):
    if "_build" in p.parts:
        continue
    text = p.read_text(encoding="utf-8")
    new = text.replace(ROOT, "<repo>").replace(HOME, "~")
    if new != text:
        files += 1
        hits += text.count(HOME)
        p.write_text(new, encoding="utf-8")
print(f"[redact] 노트북 {files}개에서 로컬 경로 {hits}곳을 지웠다")
