""".book_build/ 의 복사본에서만 링크를 고친다 (저장소 원본은 그대로).
1) 폴더 링크 `](00_orientation/)` -> 그 폴더의 README.md (README가 없으면 첫 노트북·첫 파일)
2) 제목 앵커: MyST는 제목 id에서 한글·원문자를 버리므로 GitHub식 앵커(#c③-...)가 맞지 않는다.
   링크가 가리키는 제목 앞에 ASCII 라벨 `(h-xxxxxxxx)=` 을 넣고 링크를 그 라벨로 바꾼다.
"""
import hashlib, pathlib, re, sys, unicodedata

B = pathlib.Path(sys.argv[1])
LINK = re.compile(r"\]\(([^)\s]+)\)")
files = [p for p in B.rglob("*.md") if "_build" not in p.parts]
text = {p: p.read_text() for p in files}


def gh_slug(h):
    h = re.sub(r"`|\*", "", h.strip().lower())
    return unicodedata.normalize("NFC", "".join(c for c in h if c.isalnum() or c in " -_").replace(" ", "-"))


wanted = {}  # (target file, github slug) -> label
for p in files:
    def fix(m, p=p):
        t = m.group(1)
        if re.match(r"https?:|mailto:", t):
            return m.group(0)
        path, _, anc = t.partition("#")
        tgt = (p.parent / path).resolve() if path else p.resolve()
        if not anc:
            if tgt.is_dir() and (tgt / "README.md").exists():
                return f"]({path.rstrip('/')}/README.md)"
            if tgt.is_dir():  # README가 없는 폴더(notebooks/, templates/): 첫 노트북, 없으면 첫 파일로
                first = sorted(tgt.glob("*.ipynb")) or sorted(f for f in tgt.iterdir() if f.is_file())
                if first:
                    return f"]({path.rstrip('/')}/{first[0].name})"
            return m.group(0)
        if tgt.suffix != ".md" or not tgt.exists():
            return m.group(0)
        slug = unicodedata.normalize("NFC", anc)
        label = wanted.setdefault((tgt, slug), "h-" + hashlib.sha1(f"{tgt}#{slug}".encode()).hexdigest()[:8])
        return f"]({path}#{label})"
    text[p] = LINK.sub(fix, text[p])

missing = []
for (tgt, slug), label in wanted.items():
    key = next((p for p in files if p.resolve() == tgt), None)
    lines, done, fence = text[key].split("\n"), False, False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            fence = not fence
        m = re.match(r"^#+ (.+)$", line)
        if m and not fence and gh_slug(m.group(1)) == slug:
            lines.insert(i, f"({label})=")
            done = True
            break
    if not done:
        missing.append((str(tgt.relative_to(B.resolve())), slug))
    text[key] = "\n".join(lines)

for p in files:
    p.write_text(text[p])
print(f"앵커 라벨 {len(wanted) - len(missing)}개 삽입" + (f", 대상 제목을 찾지 못함: {missing}" if missing else ""))
