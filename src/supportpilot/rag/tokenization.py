from __future__ import annotations

import re
from typing import Protocol


class Tokenizer(Protocol):
    def offsets(self, text: str) -> list[tuple[int, int]]: ...


class WhitespaceTokenizer:
    _pattern = re.compile(r"\S+")

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in self._pattern.finditer(text)]


class HFTokenizer:
    def __init__(self, name: str) -> None:
        from tokenizers import Tokenizer as _HFTokenizer

        tok = _HFTokenizer.from_pretrained(name)
        tok.no_truncation()
        tok.no_padding()
        self._tok = tok

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok.encode(text, add_special_tokens=False)
        return [(int(s), int(e)) for s, e in enc.offsets]
