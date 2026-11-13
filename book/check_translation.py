"""한국어 Markdown 문서와 영어 번역본(이름.en.md)을 비교해 번역 중 생긴 어긋남을 찾는다.

번역본은 원본 옆에 `이름.en.md`로 둔다(예: 07_capstone/README.md -> 07_capstone/README.en.md).
번역본에는 다음 두 가지를 넣는다.
  1) 제목 바로 아래의 안내 인용문. 예:
     > **English translation.** AI-assisted translation of [한국어 원문](README.md). The Korean original is authoritative and may be newer.
  2) 파일 맨 끝의 출처 표식 `<!-- i18n-source: <원본 경로> blob:<원본 파일의 git blob 해시> -->` (`stamp` 명령이 쓴다).
     요약본처럼 1:1 번역이 아닌 파일은 `kind:summary`로 적고 구조 비교를 건너뛴다.

실행(저장소 루트):
  python3 -I book/check_translation.py status                 # 번역 진행 현황과 오래된 번역
  python3 -I book/check_translation.py check [경로 ...]       # 구조 비교. 경로가 없으면 .en.md 전부
  python3 -I book/check_translation.py stamp 경로.md [--summary]  # 경로.en.md 끝에 출처 표식을 쓴다
ERROR가 하나라도 있으면 종료 상태 1이다. WARN은 사람이 확인할 항목이다.

비교하는 것(제목마다 나눈 절 단위로 비교해 어긋난 위치를 알려 준다).
  ERROR  제목 수준 순서, 표의 수와 행·열 수, 코드 블록 수와 언어, 한글이 없는 코드 블록의 내용,
         링크 대상(앵커 제외, .en.md는 .md와 같게 본다), 숫자(원문에 있는 숫자가 번역에 없을 때),
         한글이 없는 인라인 코드, 주의·판정 표식(⚠ † ✏ ✅ ❌ ★ 🟢🟡🔴 ①~⑩ 등)의 개수, 목록 항목 수,
         존재하지 않는 링크 대상과 앵커, 번역본의 출처 표식과 안내 인용문
  WARN   번역본에만 있는 숫자·링크, 굵은 글씨 수, 인용문 줄 수, 오래된 번역, 용어집(book/i18n/glossary.tsv) 위반
숫자를 일부러 바꿔 쓴 곳은 번역본에 `<!-- i18n-allow-numbers: 10,1 -->` 주석으로 허용할 수 있다.
이 스크립트는 구조와 용어만 본다. 의미가 보존됐는지는 사람이나 다른 모델이 원문과 대조해 확인해야 한다.
"""
import hashlib
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = ROOT / "book" / "i18n" / "glossary.tsv"
HANGUL = re.compile(r"[가-힣]")
MARK = re.compile(r"<!--\s*i18n-source:\s*(\S+)\s+(blob:[0-9a-f]{40}|kind:summary)\s*-->")
ALLOW = re.compile(r"<!--\s*i18n-allow-numbers:\s*([0-9.,\s]*)-->")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w+#.-]*)")
HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LINK = re.compile(r'\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)')
AUTO = re.compile(r"<(https?://[^>\s]+)>")
CODE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
NUM = re.compile(r"\d+(?:\.\d+)?")
SYMBOLS = "⚠†‡✏✅❌★◆○🟢🟡🔴①②③④⑤⑥⑦⑧⑨⑩"


def blob_sha(path):
    """`git hash-object`와 같은 값. git 없이 계산한다."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def gh_slug(h):
    h = re.sub(r"`|\*", "", h.strip().lower())
    return unicodedata.normalize("NFC", "".join(c for c in h if c.isalnum() or c in " -_").replace(" ", "-"))


def slug_list(heads):
    seen, out = Counter(), []
    for _, title in heads:
        s = gh_slug(title)
        out.append(s if seen[s] == 0 else f"{s}-{seen[s]}")
        seen[s] += 1
    return out


def parse(text):
    """제목 단위 절, 코드 블록, 제목 목록으로 나눈다. 코드 블록 안은 절 본문에 넣지 않는다."""
    sections = [{"title": "(top)", "lines": []}]
    fences, heads, fence = [], [], None
    for line in text.split("\n"):
        m = FENCE.match(line)
        if fence is not None:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not line.strip().strip(fence[0]):
                fence = None
            else:
                fences[-1]["body"].append(line)
            continue
        if m:
            fence = m.group(1)
            fences.append({"lang": m.group(2), "body": []})
            continue
        h = HEAD.match(line)
        if h:
            heads.append((len(h.group(1)), h.group(2)))
            sections.append({"title": h.group(2), "lines": []})
            continue
        sections[-1]["lines"].append(line)
    return sections, fences, heads


def norm_target(t):
    if re.match(r"(https?:|mailto:)", t):
        return t.rstrip("/")
    p, _, _ = t.partition("#")
    if p.endswith(".en.md"):
        p = p[:-6] + ".md"
    return p or "#"


def facts(lines):
    text = re.sub(r"<!--.*?-->", "", "\n".join(lines), flags=re.S)
    tables, cur, items, quotes = [], None, 0, 0
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("|"):
            n = len(re.split(r"(?<!\\)\|", s)) - (2 if s.endswith("|") else 1)
            if cur is None:
                cur = []
                tables.append(cur)
            cur.append(n)
        else:
            cur = None
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s", ln):
            items += 1
        if s.startswith(">"):
            quotes += 1
    links = Counter(norm_target(t) for t in LINK.findall(text) + AUTO.findall(text))
    body = LINK.sub(lambda m: m.group(0)[: m.group(0).index("](") + 2] + ")", text)
    body = re.sub(r"https?://\S+", "", AUTO.sub("", body))
    body = re.sub(r"(?<=\d),(?=\d{3})", "", body)
    return {"tables": tables, "items": items, "quotes": quotes, "links": links,
            "nums": Counter(NUM.findall(body)), "codes": [m.group(2).strip() for m in CODE.finditer(body)],
            "bold": body.count("**")}


def fmt(c):
    return ", ".join(f"{k}x{v}" if v > 1 else k for k, v in sorted(c.items()))


def compare(ko, en):
    """텍스트 두 개를 비교해 (수준, 위치, 메시지) 목록을 돌려준다."""
    issues = []
    err = lambda where, msg: issues.append(("ERROR", where, msg))
    warn = lambda where, msg: issues.append(("WARN", where, msg))
    ksec, kfen, khead = parse(ko)
    esec, efen, ehead = parse(en)
    allowed = set()
    for m in ALLOW.finditer(en):
        allowed |= {x.strip() for x in re.split(r"[,\s]+", m.group(1)) if x.strip()}

    kl, el = [l for l, _ in khead], [l for l, _ in ehead]
    same_heads = kl == el
    if not same_heads:
        i = next((i for i, (a, b) in enumerate(zip(kl, el)) if a != b), min(len(kl), len(el)))
        err("headings", f"제목 수준 순서가 다르다: 원문 {len(kl)}개, 번역 {len(el)}개, 처음 어긋난 곳 {i + 1}번째"
            f" (원문 {khead[i][1]!r} / 번역 {ehead[i][1]!r})" if i < min(len(khead), len(ehead)) else
            f"제목 수준 순서가 다르다: 원문 {len(kl)}개, 번역 {len(el)}개")

    if [f["lang"] for f in kfen] != [f["lang"] for f in efen]:
        err("code blocks", f"코드 블록 수나 언어가 다르다: 원문 {[f['lang'] for f in kfen]}, 번역 {[f['lang'] for f in efen]}")
    else:
        for i, (a, b) in enumerate(zip(kfen, efen), 1):
            if not HANGUL.search("\n".join(a["body"])):
                if a["body"] != b["body"]:
                    err(f"code block #{i}", "한글이 없는 코드 블록의 내용이 원문과 다르다")
            elif len(a["body"]) != len(b["body"]):
                warn(f"code block #{i}", f"한글이 있는 코드 블록의 줄 수가 다르다: 원문 {len(a['body'])}, 번역 {len(b['body'])}")
            if a["lang"] in ("markdown", "md") and HANGUL.search("\n".join(a["body"])):
                # 템플릿처럼 마크다운 코드 블록 안에 한글 문서가 든 경우에는 블록 안의 구조도 같은 기준으로 비교한다
                issues += [(lv, f"code block #{i} (markdown) / {w}", m) for lv, w, m in compare("\n".join(a["body"]), "\n".join(b["body"]))]

    if same_heads:
        for i, (ks, es) in enumerate(zip(ksec, esec)):
            where = f"section {i} {es['title']!r}"
            kf, ef = facts(ks["lines"]), facts(es["lines"])
            if kf["tables"] != ef["tables"]:
                i_t = next((j for j, (a, b) in enumerate(zip(kf["tables"], ef["tables"])) if a != b), None)
                detail = (f"{i_t + 1}번째 표의 행별 열 수가 원문 {kf['tables'][i_t][:8]}, 번역 {ef['tables'][i_t][:8]}" if i_t is not None
                          else f"표의 수가 원문 {len(kf['tables'])}개, 번역 {len(ef['tables'])}개")
                err(where, f"표의 모양이 다르다: {detail}")
            if kf["items"] != ef["items"]:
                err(where, f"목록 항목 수가 다르다: 원문 {kf['items']}, 번역 {ef['items']}")
            if kf["quotes"] != ef["quotes"]:
                warn(where, f"인용문 줄 수가 다르다: 원문 {kf['quotes']}, 번역 {ef['quotes']}")
            lost, extra = kf["links"] - ef["links"], ef["links"] - kf["links"]
            if lost:
                err(where, f"원문의 링크가 번역에 없다: {fmt(lost)}")
            if extra:
                warn(where, f"번역에만 있는 링크: {fmt(extra)}")
            lost = Counter({k: v for k, v in (kf["nums"] - ef["nums"]).items() if k not in allowed})
            extra = Counter({k: v for k, v in (ef["nums"] - kf["nums"]).items() if k not in allowed})
            if lost:
                err(where, f"원문의 숫자가 번역에 없다: {fmt(lost)}")
            if extra:
                warn(where, f"번역에만 있는 숫자: {fmt(extra)}")
            if kf["bold"] != ef["bold"]:
                warn(where, f"굵은 글씨(**) 수가 다르다: 원문 {kf['bold']}, 번역 {ef['bold']}")

    kcodes = Counter(c for s in ksec for c in facts(s["lines"])["codes"] if not HANGUL.search(c))
    ecodes = Counter(c for s in esec for c in facts(s["lines"])["codes"])
    lost = kcodes - ecodes
    if lost:
        err("inline code", f"원문의 인라인 코드가 번역에 없다: {fmt(lost)}")
    for ch in SYMBOLS:
        if ko.count(ch) != en.count(ch):
            err("markers", f"표식 {ch!r}의 개수가 다르다: 원문 {ko.count(ch)}, 번역 {en.count(ch)}")
    return issues


def check_links(text, path):
    """상대 링크의 대상 파일과 앵커가 실제로 있는지 본다."""
    issues, base, cache = [], Path(path).parent, {}

    def slugs(p):
        if p not in cache:
            cache[p] = slug_list(parse(p.read_text(encoding="utf-8"))[2])
        return cache[p]

    sections = parse(text)[0]
    for sec in sections:
        body = re.sub(r"<!--.*?-->", "", "\n".join(sec["lines"]), flags=re.S)
        for t in LINK.findall(body):
            if re.match(r"(https?:|mailto:)", t):
                continue
            p, _, a = t.partition("#")
            target = (base / unquote(p)).resolve() if p else Path(path).resolve()
            if p and not target.exists():
                issues.append(("ERROR", path, f"링크 대상이 없다: {t}"))
            elif a and target.suffix == ".md" and target.is_file() and unicodedata.normalize("NFC", unquote(a)) not in slugs(target):
                issues.append(("ERROR", path, f"링크 앵커에 해당하는 제목이 없다: {t}"))
    return issues


def load_glossary(path=GLOSSARY):
    rows = {}
    if not Path(path).exists():
        return []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line.strip() or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) < 2:
            continue
        rows[c[0]] = {"ko": [k for k in c[0].split("|") if k], "en": [v.strip() for v in c[1].split("|") if v.strip()],
                      "count": "count" in (c[2].split(",") if len(c) > 2 else [])}
    return list(rows.values())


def en_pattern(v):
    parts = r"\s+".join(re.escape(p) for p in v.split())
    if len(v) >= 4 and not v.isupper():
        return re.compile(r"(?<![A-Za-z])" + parts + r"[a-z]*", re.I)
    return re.compile(r"(?<![A-Za-z])" + parts + r"(?![A-Za-z])")


def count_terms(text, groups):
    """가장 긴 일치를 먼저 고르고 겹치는 일치는 버려 용어별 횟수를 센다."""
    spans = []
    for key, pats in groups:
        for p in pats:
            spans += [(m.start(), -(m.end() - m.start()), m.end(), key) for m in p.finditer(text)]
    spans.sort()
    taken, counts = -1, Counter()
    for start, _, end, key in spans:
        if start >= taken:
            counts[key] += 1
            taken = end
    return counts


def glossary_issues(ko, en, rows):
    def plain(t):
        return "\n".join("\n".join(s["lines"]) for s in parse(t)[0])

    kt, et = plain(ko), plain(en)
    kc = count_terms(kt, [(i, [re.compile(re.escape(k)) for k in r["ko"]]) for i, r in enumerate(rows)])
    ec = count_terms(et, [(i, [en_pattern(v) for v in r["en"]]) for i, r in enumerate(rows)])
    out = []
    for i, r in enumerate(rows):
        k, e = kc[i], ec[i]
        if k and not e:
            out.append(("WARN", "glossary", f"원문의 {'/'.join(r['ko'])!r} x{k}에 해당하는 영어({r['en'][0]!r} 등)가 번역에 없다"))
        elif r["count"] and e < k:
            out.append(("WARN", "glossary", f"원문의 {'/'.join(r['ko'])!r}는 {k}번인데 번역의 대응어({r['en'][0]!r} 등)는 {e}번이다. 판정·유보 표현이 빠졌는지 본다"))
    return out


def en_of(ko):
    return ko[:-3] + ".en.md"


def ko_of(en):
    return en[:-6] + ".md"


def check_header(en_text, en_rel):
    """출처 표식과 안내 인용문. 표식의 종류(blob 또는 summary)를 함께 돌려준다."""
    issues, m = [], MARK.search(en_text)
    kind = None
    if not m:
        return [("ERROR", en_rel, "출처 표식(<!-- i18n-source: ... -->)이 없다. `stamp`로 쓴다")], None
    src, kind = m.group(1), m.group(2)
    if src != ko_of(en_rel):
        issues.append(("ERROR", en_rel, f"출처 표식의 경로가 {src!r}인데 {ko_of(en_rel)!r}여야 한다"))
    elif not (ROOT / src).exists():
        issues.append(("ERROR", en_rel, f"출처 파일이 없다: {src}"))
    elif kind.startswith("blob:") and kind[5:] != blob_sha(ROOT / src):
        issues.append(("WARN", en_rel, "오래된 번역: 번역 뒤에 한국어 원문이 바뀌었다. 변경을 반영한 뒤 `stamp`를 다시 실행한다"))
    if kind != "kind:summary":
        head = [l for l in en_text.split("\n") if l.strip()][:12]
        if not any(l.lstrip().startswith(">") and "translation" in l.lower() for l in head):
            issues.append(("ERROR", en_rel, "제목 아래에 번역 안내 인용문(> **English translation.** ...)이 없다"))
    return issues, kind


def strip_notice(en_text):
    """제목 아래의 번역 안내 인용문은 원문에 없는 줄이므로 비교에서 뺀다."""
    lines, seen = en_text.split("\n"), 0
    for i, l in enumerate(lines):
        if l.strip():
            seen += 1
        if seen > 12:
            break
        if l.lstrip().startswith(">") and "translation" in l.lower():
            return "\n".join(lines[:i] + lines[i + 1:])
    return en_text


def check_pair(ko_rel, en_rel, rows):
    ko_p, en_p = ROOT / ko_rel, ROOT / en_rel
    if not en_p.exists():
        return [("ERROR", en_rel, "번역본이 없다")]
    if not ko_p.exists():
        return [("ERROR", en_rel, f"원문이 없다: {ko_rel}")]
    ko, en = ko_p.read_text(encoding="utf-8"), en_p.read_text(encoding="utf-8")
    issues, kind = check_header(en, en_rel)
    issues += check_links(en, en_p)
    if kind != "kind:summary":
        en = strip_notice(en)
        issues += compare(ko, en)
        issues += glossary_issues(ko, en, rows)
    return [(lv, w if w.startswith(en_rel) else f"{en_rel}: {w}", m) for lv, w, m in issues]


def tracked(pattern):
    out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [l for l in out.split("\n") if l]


def cmd_check(args):
    rows = load_glossary()
    if args:
        pairs = [(ko_of(a), a) if a.endswith(".en.md") else (a, en_of(a)) for a in args]
    else:
        pairs = [(ko_of(e), e) for e in tracked("*.en.md")]
    errors = warns = 0
    for ko, en in pairs:
        for lv, where, msg in check_pair(ko, en, rows):
            print(f"{lv:5s} {where}: {msg}")
            errors += lv == "ERROR"
            warns += lv == "WARN"
    print(f"\n{len(pairs)}개 파일: ERROR {errors}, WARN {warns}")
    return 1 if errors else 0


def cmd_status():
    toc = (ROOT / "book" / "myst.yml").read_text(encoding="utf-8")
    kos = [f for f in tracked("*.md") if not f.endswith(".en.md")]
    done = total = 0
    print(f"{'상태':10s} {'한글 글자':>8s}  파일")
    for f in kos:
        n = len(HANGUL.findall((ROOT / f).read_text(encoding="utf-8")))
        if not n:
            continue
        en = en_of(f)
        total += n
        if not (ROOT / en).exists():
            st = "missing"
        else:
            m = MARK.search((ROOT / en).read_text(encoding="utf-8"))
            st = "no-marker" if not m else "summary" if m.group(2) == "kind:summary" else \
                "stale" if m.group(2)[5:] != blob_sha(ROOT / f) else "ok"
            done += n if st == "ok" else 0
            if f"file: {en}" not in toc:
                st += "+no-toc"
        print(f"{st:10s} {n:8d}  {f}")
    orphans = [e for e in tracked("*.en.md") if not (ROOT / ko_of(e)).exists()]
    print(f"\n번역 완료(ok) {done}/{total} 글자 ({100 * done / max(total, 1):.0f}%). 출처 없는 번역본: {orphans or '없음'}")
    return 0


def cmd_stamp(args):
    summary = "--summary" in args
    paths = [a for a in args if not a.startswith("--")]
    if len(paths) != 1 or not paths[0].endswith(".md") or paths[0].endswith(".en.md"):
        print("사용법: stamp 원문경로.md [--summary]")
        return 2
    ko, en = ROOT / paths[0], ROOT / en_of(paths[0])
    if not ko.exists() or not en.exists():
        print(f"원문 또는 번역본이 없다: {ko} / {en}")
        return 2
    marker = f"<!-- i18n-source: {paths[0]} {'kind:summary' if summary else 'blob:' + blob_sha(ko)} -->"
    text = MARK.sub("", en.read_text(encoding="utf-8")).rstrip("\n")
    en.write_text(text + "\n\n" + marker + "\n", encoding="utf-8")
    print(marker)
    return 0


def main(argv):
    if not argv or argv[0] not in ("status", "check", "stamp"):
        print(__doc__)
        return 2
    return {"status": lambda: cmd_status(), "check": lambda: cmd_check(argv[1:]), "stamp": lambda: cmd_stamp(argv[1:])}[argv[0]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
