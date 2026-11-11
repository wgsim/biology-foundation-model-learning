"""실행된 노트북 복사본에 원본의 설명(마크다운) 셀만 덮어쓴다.

사용: python overlay_markdown.py <원본 ipynb> <실행된 복사본 ipynb>
- 두 노트북의 셀 수·셀 종류 순서·코드 셀 소스가 모두 같을 때만 덮어쓰고 종료 코드 0을 돌려준다.
- 코드 셀이 하나라도 다르면 아무것도 바꾸지 않고 종료 코드 1을 돌려준다(그때는 다시 실행해야 한다).
설명 셀만 바뀐 경우 노트북을 다시 실행하지 않아도 출력이 그대로 유효하다.
"""
import json
import pathlib
import sys

src_path, out_path = map(pathlib.Path, sys.argv[1:3])
src = json.loads(src_path.read_text(encoding="utf8"))
out = json.loads(out_path.read_text(encoding="utf8"))

if len(src["cells"]) != len(out["cells"]):
    sys.exit(1)
for a, b in zip(src["cells"], out["cells"]):
    if a["cell_type"] != b["cell_type"]:
        sys.exit(1)
    if a["cell_type"] == "code" and a["source"] != b["source"]:
        sys.exit(1)

for a, b in zip(src["cells"], out["cells"]):
    if a["cell_type"] == "markdown":
        b["source"] = a["source"]

out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
