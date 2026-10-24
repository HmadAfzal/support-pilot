from pathlib import Path

from supportpilot.ingest.load_docs import clean, doc_url, find_title, split_front_matter

BASE = "https://docs.gitlab.com/"


def test_front_matter_split():
    text = "---\nstage: Verify\ntitle: Pipelines\n---\n# Pipelines\nBody"
    meta, body = split_front_matter(text)
    assert meta["title"] == "Pipelines"
    assert body.startswith("# Pipelines")


def test_front_matter_missing():
    meta, body = split_front_matter("# Just a doc")
    assert meta == {}
    assert body == "# Just a doc"


def test_clean_strips_noise_but_keeps_text():
    body = (
        "# Title\n\n"
        "{{< history >}}\n\n- Introduced in 16.0\n\n{{< /history >}}\n\n"
        "<!-- a comment -->\n"
        '{{< alert type="note" >}}\n\nKeep this note.\n\n{{< /alert >}}\n\n\n\n'
        "See [the guide](../guide.md) and ![diagram](img.png)."
    )
    out = clean(body)
    assert "Introduced in" not in out
    assert "comment" not in out
    assert "{{" not in out
    assert "Keep this note." in out
    assert "See the guide and diagram." in out
    assert "\n\n\n" not in out


def test_doc_url_variants():
    assert doc_url(Path("ci/yaml/index.md"), BASE) == BASE + "ci/yaml/"
    assert doc_url(Path("user/permissions.md"), BASE) == BASE + "user/permissions/"
    assert doc_url(Path("ci/_index.md"), BASE) == BASE + "ci/"


def test_find_title_fallbacks():
    assert find_title({"title": "From meta"}, "# H1", "stem") == "From meta"
    assert find_title({}, "text\n# From H1\nmore", "stem") == "From H1"
    assert find_title({}, "no heading", "stem") == "stem"
