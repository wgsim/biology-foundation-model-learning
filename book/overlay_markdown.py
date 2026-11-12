"""실행된 노트북 복사본에 원본의 설명(마크다운) 셀과 코드 셀의 주석을 덮어쓴다.

사용: python overlay_markdown.py <원본 ipynb> <실행된 복사본 ipynb>
- 두 노트북의 셀 수·셀 종류 순서가 같고, 코드 셀의 소스가 같거나 주석만 다를 때 덮어쓰고 종료 코드 0을 돌려준다.
- 코드가 하나라도 다르면 아무것도 바꾸지 않고 종료 코드 1을 돌려준다(그때는 다시 실행해야 한다).
설명 셀이나 주석만 바뀐 경우 노트북을 다시 실행하지 않아도 출력이 그대로 유효하다.
"주석만 다르다"는 파이썬 토큰열(주석·빈 줄 제외)과 줄 수가 같다는 뜻이다. 문자열 리터럴이 바뀌면 출력이 달라질 수 있으므로 다른 코드로 본다.
토큰화할 수 없는 셀(매직 명령 등)은 소스가 완전히 같아야 한다.
"""
import io
import json
import pathlib
import sys
import tokenize

SKIP = {tokenize.COMMENT, tokenize.NL}


def code_tokens(text):
    return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type not in SKIP]


def same_code(a, b):
    if a == b:
        return True
    if a.count("\n") != b.count("\n"):
        return False
    try:
        return code_tokens(a) == code_tokens(b)
    except (tokenize.TokenError, SyntaxError):  # IndentationError는 SyntaxError의 하위 클래스
        return False


src_path, out_path = map(pathlib.Path, sys.argv[1:3])
src = json.loads(src_path.read_text(encoding="utf8"))
out = json.loads(out_path.read_text(encoding="utf8"))

if len(src["cells"]) != len(out["cells"]):
    sys.exit(1)
for a, b in zip(src["cells"], out["cells"]):
    if a["cell_type"] != b["cell_type"]:
        sys.exit(1)
    if a["cell_type"] == "code" and not same_code("".join(a["source"]), "".join(b["source"])):
        sys.exit(1)

for a, b in zip(src["cells"], out["cells"]):
    if a["cell_type"] in ("markdown", "code"):
        b["source"] = a["source"]

out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
