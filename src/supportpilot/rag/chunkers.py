from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from supportpilot.rag.tokenization import Tokenizer

_HEADER = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Piece:
    text: str
    n_tokens: int
    heading_path: str = ""


def fixed_pieces(text: str, tok: Tokenizer, size: int, overlap: int) -> list[Piece]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"need size > 0 and 0 <= overlap < size, got {size}, {overlap}")
    offs = tok.offsets(text)
    n = len(offs)
    if n == 0:
        return []
    pieces: list[Piece] = []
    start = 0
    while True:
        end = min(start + size, n)
        pieces.append(Piece(text[offs[start][0] : offs[end - 1][1]], end - start))
        if end == n:
            break
        start = end - overlap
    return pieces


def split_sections(text: str, root_title: str = "", max_level: int = 3) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    path = root_title
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((path, body))
        buf.clear()

    for line in text.splitlines():
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            buf.append(line)
            continue
        m = None if in_fence else _HEADER.match(line)
        if m and len(m.group(1)) <= max_level:
            flush()
            level, title = len(m.group(1)), m.group(2)
            stack = [(lv, t) for lv, t in stack if lv < level]
            stack.append((level, title))
            path = " > ".join(t for _, t in stack)
        buf.append(line)
    flush()
    return sections


def header_pieces(
    text: str,
    tok: Tokenizer,
    *,
    root_title: str = "",
    max_tokens: int = 400,
    min_tokens: int = 64,
    overlap: int = 80,
    max_level: int = 3,
) -> list[Piece]:
    pieces: list[Piece] = []
    buf_text, buf_n, buf_path = "", 0, ""

    def flush() -> None:
        nonlocal buf_text, buf_n, buf_path
        if buf_text:
            pieces.append(Piece(buf_text, buf_n, buf_path))
        buf_text, buf_n, buf_path = "", 0, ""

    for path, sec in split_sections(text, root_title, max_level):
        n = len(tok.offsets(sec))
        if n == 0:
            continue
        if n > max_tokens:
            flush()
            pieces.extend(
                replace(p, heading_path=path) for p in fixed_pieces(sec, tok, max_tokens, overlap)
            )
        elif buf_text and buf_n < min_tokens and buf_n + n <= max_tokens:
            buf_text += "\n\n" + sec
            buf_n += n
        else:
            flush()
            buf_text, buf_n, buf_path = sec, n, path
    flush()
    return pieces


def chunk_document(
    doc: dict[str, Any], name: str, spec: dict[str, Any], tok: Tokenizer
) -> list[dict[str, Any]]:
    kind = spec["type"]
    if kind == "fixed":
        pieces = fixed_pieces(doc["text"], tok, spec["size"], spec["overlap"])
    elif kind == "header":
        pieces = header_pieces(
            doc["text"],
            tok,
            root_title=doc.get("title", ""),
            max_tokens=spec["max_tokens"],
            min_tokens=spec["min_tokens"],
            overlap=spec["overlap"],
            max_level=spec["max_level"],
        )
    else:
        raise ValueError(f"unknown chunking type: {kind!r}")
    return [
        {
            "chunk_id": f"{doc['doc_id']}::{name}::{i:04d}",
            "doc_id": doc["doc_id"],
            "strategy": name,
            "index": i,
            "text": p.text,
            "n_tokens": p.n_tokens,
            "heading_path": p.heading_path,
            "title": doc.get("title", ""),
            "area": doc.get("area", ""),
            "url": doc.get("url", ""),
            "path": doc.get("path", ""),
            "commit": doc.get("commit", ""),
        }
        for i, p in enumerate(pieces)
    ]
