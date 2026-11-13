"""book/check_translation.py의 시험. 실행(저장소 루트): python3 -I book/test_check_translation.py"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("ct", Path(__file__).resolve().parent / "check_translation.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)

KO = """# 제목

> 인용 한 줄

표본은 3,380개이고 `main.py`를 쓴다. [원문](other.md)과 [외부](https://example.com/a) 참고. ⚠️ 확인하지 못했다. **중요**

## 절 하나

| a | b |
|---|---|
| 1 | 2 |
| 3 | 4 |

- 항목 하나
- 항목 둘

```bash
echo hi
```
"""
EN = """# Title

> A quote line

There are 3380 samples and it uses `main.py`. See [the original](other.en.md) and [external](https://example.com/a). ⚠️ Could not be verified. **Important**

## Section one

| a | b |
|---|---|
| 1 | 2 |
| 3 | 4 |

- item one
- item two

```bash
echo hi
```
"""


def errors(ko, en):
    return [m for lv, _, m in ct.compare(ko, en) if lv == "ERROR"]


class Compare(unittest.TestCase):
    def test_faithful_translation_passes(self):
        self.assertEqual(ct.compare(KO, EN), [])

    def test_table_row_missing(self):
        self.assertTrue(any("표의 모양" in m for m in errors(KO, EN.replace("| 3 | 4 |\n", ""))))

    def test_table_column_missing(self):
        self.assertTrue(any("표의 모양" in m for m in errors(KO, EN.replace("| 1 | 2 |", "| 1 |"))))

    def test_number_changed(self):
        e = errors(KO, EN.replace("3380", "3308"))
        self.assertTrue(any("숫자" in m and "3380" in m for m in e))

    def test_number_allowed(self):
        en = EN.replace("3380", "3308") + "<!-- i18n-allow-numbers: 3380, 3308 -->\n"
        self.assertEqual([m for m in errors(KO, en) if "숫자" in m], [])

    def test_link_target_changed(self):
        self.assertTrue(any("링크" in m for m in errors(KO, EN.replace("other.en.md", "another.md"))))

    def test_url_changed(self):
        self.assertTrue(any("링크" in m for m in errors(KO, EN.replace("example.com/a", "example.com/b"))))

    def test_caveat_marker_dropped(self):
        self.assertTrue(any("⚠" in m for m in errors(KO, EN.replace("⚠️ ", ""))))

    def test_inline_code_dropped(self):
        self.assertTrue(any("인라인 코드" in m for m in errors(KO, EN.replace("`main.py`", "main.py"))))

    def test_code_block_changed(self):
        self.assertTrue(any("코드 블록" in m for m in errors(KO, EN.replace("echo hi", "echo hello"))))

    def test_heading_level_changed(self):
        self.assertTrue(any("제목" in m for m in errors(KO, EN.replace("## Section one", "### Section one"))))

    def test_heading_added(self):
        self.assertTrue(any("제목" in m for m in errors(KO, EN + "\n## Extra\n")))

    def test_list_item_missing(self):
        self.assertTrue(any("목록" in m for m in errors(KO, EN.replace("- item two\n", ""))))

    def test_bold_is_only_a_warning(self):
        issues = ct.compare(KO, EN.replace("**Important**", "Important"))
        self.assertEqual([lv for lv, _, _ in issues], ["WARN"])

    def test_extra_number_is_only_a_warning(self):
        issues = ct.compare(KO, EN.replace("samples", "samples (about 7)"))
        self.assertEqual([lv for lv, _, _ in issues], ["WARN"])


class MarkdownInFence(unittest.TestCase):
    ko = "# 템플릿\n\n```markdown\n# 실험 [번호]\n- 날짜:\n\n## 결과\n| 설정 | 값 |\n|---|---|\n| 3,380 | ⚠️ |\n```\n"
    en = "# Template\n\n```markdown\n# Experiment [number]\n- Date:\n\n## Results\n| Setting | Value |\n|---|---|\n| 3380 | ⚠️ |\n```\n"

    def test_faithful_template_passes(self):
        self.assertEqual(ct.compare(self.ko, self.en), [])

    def test_changes_inside_the_fence_are_found(self):
        self.assertTrue(any("(markdown)" in w and "표의 모양" in m for _, w, m in ct.compare(self.ko, self.en.replace("|---|---|\n", ""))))
        self.assertTrue(any("(markdown)" in w and "⚠" in m for _, w, m in ct.compare(self.ko, self.en.replace("⚠️", "ok"))))
        self.assertTrue(any("(markdown)" in w and "숫자" in m for _, w, m in ct.compare(self.ko, self.en.replace("3380", "3308"))))


class Links(unittest.TestCase):
    def test_missing_target_and_anchor(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "b.md").write_text("# Top\n\n## Real section\n", encoding="utf-8")
            a = d / "a.md"
            text = "# A\n\n[ok](b.md#real-section) [bad anchor](b.md#nope) [gone](c.md) [self](#a) [bad self](#zzz)\n"
            a.write_text(text, encoding="utf-8")
            msgs = [m for _, _, m in ct.check_links(text, a)]
            self.assertEqual(len(msgs), 3, msgs)
            self.assertTrue(any("nope" in m for m in msgs) and any("c.md" in m for m in msgs) and any("zzz" in m for m in msgs))

    def test_duplicate_heading_slugs(self):
        self.assertEqual(ct.slug_list([(2, "Same"), (2, "Same")]), ["same", "same-1"])


class Glossary(unittest.TestCase):
    rows = [
        {"ko": ["지지되지 않음"], "en": ["not supported"], "count": True},
        {"ko": ["지지"], "en": ["supported", "support"], "count": True},
        {"ko": ["확인하지 못", "미확인"], "en": ["could not verify", "could not be verified", "unverified"], "count": True},
        {"ko": ["시험 실행"], "en": ["dry run"], "count": False},
    ]

    def test_verdict_overlap_is_counted_longest_first(self):
        ko = "판정은 지지다. 다른 판정은 지지되지 않음이다."
        self.assertEqual(ct.glossary_issues(ko, "The verdict is supported. The other is not supported.", self.rows), [])

    def test_dropped_negative_verdict(self):
        ko = "하나는 지지, 다른 하나는 지지되지 않음이다."
        out = ct.glossary_issues(ko, "One is supported, the other is supported.", self.rows)
        self.assertTrue(any("지지되지 않음" in m for _, _, m in out), out)

    def test_caveat_dropped(self):
        ko = "이것은 확인하지 못했다. 저것은 미확인이다."
        out = ct.glossary_issues(ko, "This is true. That is also true.", self.rows)
        self.assertTrue(any("확인하지 못" in m for _, _, m in out), out)

    def test_term_absent(self):
        out = ct.glossary_issues("시험 실행을 했다.", "We ran it.", self.rows)
        self.assertTrue(any("시험 실행" in m for _, _, m in out), out)

    def test_real_glossary_loads(self):
        rows = ct.load_glossary()
        self.assertGreater(len(rows), 100)
        self.assertTrue(all(r["ko"] and r["en"] for r in rows))


class Notice(unittest.TestCase):
    def test_notice_line_is_ignored_by_the_comparison(self):
        en = EN.replace("# Title\n\n", "# Title\n\n> **English translation.** See the [Korean original](x.md).\n\n", 1)
        self.assertNotEqual(ct.compare(KO, en), [])
        self.assertEqual(ct.compare(KO, ct.strip_notice(en)), [])

    def test_only_a_notice_near_the_top_is_removed(self):
        body = "# T\n\n" + "line\n\n" * 20 + "> late translation quote\n"
        self.assertEqual(ct.strip_notice(body), body)


class Header(unittest.TestCase):
    def test_blob_sha_matches_git(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"hello\n")
            f.flush()
            self.assertEqual(ct.blob_sha(f.name), "ce013625030ba8dba906f756967f9e9ca394464a")

    def test_missing_marker(self):
        issues, kind = ct.check_header("# T\n\n> English translation.\n", "README.en.md")
        self.assertEqual(kind, None)
        self.assertEqual(issues[0][0], "ERROR")

    def test_summary_marker_needs_no_notice(self):
        issues, kind = ct.check_header("# T\n\ntext\n\n<!-- i18n-source: README.md kind:summary -->\n", "README.en.md")
        self.assertEqual((issues, kind), ([], "kind:summary"))

    def test_notice_required_for_translations(self):
        sha = ct.blob_sha(ct.ROOT / "README.md")
        text = f"# T\n\ntext\n\n<!-- i18n-source: README.md blob:{sha} -->\n"
        self.assertTrue(any("안내 인용문" in m for _, _, m in ct.check_header(text, "README.en.md")[0]))
        ok = f"# T\n\n> **English translation.** AI-assisted.\n\n<!-- i18n-source: README.md blob:{sha} -->\n"
        self.assertEqual(ct.check_header(ok, "README.en.md")[0], [])

    def test_stale_when_source_changed(self):
        text = "# T\n\n> English translation.\n\n<!-- i18n-source: README.md blob:" + "0" * 40 + " -->\n"
        self.assertTrue(any(lv == "WARN" and "오래된" in m for lv, _, m in ct.check_header(text, "README.en.md")[0]))


if __name__ == "__main__":
    unittest.main(verbosity=1)
