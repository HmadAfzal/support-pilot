import pytest

from supportpilot.rag.chunkers import chunk_document, fixed_pieces, header_pieces, split_sections
from supportpilot.rag.tokenization import WhitespaceTokenizer

TOK = WhitespaceTokenizer()


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_fixed_windows_and_overlap():
    pieces = fixed_pieces(words(10), TOK, size=4, overlap=1)
    assert [p.text for p in pieces] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert all(p.n_tokens == 4 for p in pieces)


def test_fixed_last_window_may_be_short():
    pieces = fixed_pieces(words(9), TOK, size=4, overlap=1)
    assert pieces[-1].text == "w6 w7 w8"
    assert pieces[-1].n_tokens == 3


def test_fixed_short_and_empty_text():
    assert len(fixed_pieces("a b c", TOK, size=10, overlap=2)) == 1
    assert fixed_pieces("   \n ", TOK, size=10, overlap=2) == []


def test_fixed_covers_every_token():
    text = words(57)
    joined = " ".join(p.text for p in fixed_pieces(text, TOK, size=8, overlap=2))
    assert all(f"w{i}" in joined.split() for i in range(57))


@pytest.mark.parametrize("size,overlap", [(0, 0), (4, 4), (4, -1)])
def test_fixed_rejects_bad_params(size, overlap):
    with pytest.raises(ValueError):
        fixed_pieces("a b c", TOK, size=size, overlap=overlap)


def test_sections_heading_paths():
    text = "# Doc\nintro\n\n## A\nalpha\n\n### A1\ndeep\n\n## B\nbeta\n"
    paths = [p for p, _ in split_sections(text, root_title="Doc", max_level=3)]
    assert paths == ["Doc", "Doc > A", "Doc > A > A1", "Doc > B"]


def test_sections_ignore_headers_in_code_fences():
    text = "# Doc\n\n```bash\n# not a header\necho hi\n```\n"
    sections = split_sections(text, root_title="Doc")
    assert len(sections) == 1
    assert "# not a header" in sections[0][1]


def test_sections_respect_max_level():
    text = "# Doc\n## A\ntext\n### B\nmore\n"
    sections = split_sections(text, root_title="Doc", max_level=2)
    assert len(sections) == 2
    assert "### B" in sections[1][1]


def test_header_one_piece_per_section_when_min_zero():
    text = "# Doc\nintro\n\n## A\nalpha\n\n## B\nbeta\n"
    pieces = header_pieces(text, TOK, root_title="Doc", max_tokens=100, min_tokens=0, overlap=2)
    assert [p.heading_path for p in pieces] == ["Doc", "Doc > A", "Doc > B"]


def test_header_merges_tiny_sections():
    text = "# Doc\n\n## A\nshort\n\n## B\nalso short\n"
    pieces = header_pieces(text, TOK, root_title="Doc", max_tokens=100, min_tokens=10, overlap=2)
    assert len(pieces) == 1
    assert pieces[0].heading_path == "Doc"
    assert "## B" in pieces[0].text


def test_header_splits_oversized_section_and_keeps_path():
    text = f"# Doc\n\n## A\n{words(30)}\n"
    pieces = header_pieces(text, TOK, root_title="Doc", max_tokens=10, min_tokens=0, overlap=2)
    a_pieces = [p for p in pieces if p.heading_path == "Doc > A"]
    assert len(a_pieces) > 1
    assert all(p.n_tokens <= 10 for p in pieces)


def test_chunk_document_ids_and_metadata():
    doc = {
        "doc_id": "d1",
        "title": "T",
        "area": "ci_cd",
        "url": "u",
        "path": "p",
        "commit": "c",
        "text": words(10),
    }
    chunks = chunk_document(doc, "fixed_4", {"type": "fixed", "size": 4, "overlap": 1}, TOK)
    assert [c["chunk_id"] for c in chunks] == [
        "d1::fixed_4::0000",
        "d1::fixed_4::0001",
        "d1::fixed_4::0002",
    ]
    assert chunks[0]["area"] == "ci_cd" and chunks[0]["strategy"] == "fixed_4"


def test_chunk_document_unknown_type():
    with pytest.raises(ValueError):
        chunk_document({"doc_id": "d", "text": "a"}, "x", {"type": "nope"}, TOK)
