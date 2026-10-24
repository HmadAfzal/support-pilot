from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed/documents.jsonl")
MIN_CHARS = 200
SKIP_NAMES = {"AGENTS.md", "CLAUDE.md"}


@dataclass(frozen=True)
class Source:
    repo: str
    subpath: str
    area: str
    root: str
    url_base: str


GITLAB = "https://docs.gitlab.com/"
RUNNER = "https://docs.gitlab.com/runner/"

SOURCES = [
    Source("gitlab", "doc/ci", "ci_cd", "doc", GITLAB),
    Source("gitlab", "doc/user/project/merge_requests", "merge_requests", "doc", GITLAB),
    Source("gitlab-runner", "docs", "runner", "docs", RUNNER),
    Source("gitlab", "doc/user/permissions.md", "permissions", "doc", GITLAB),
    Source("gitlab", "doc/subscriptions", "subscriptions", "doc", GITLAB),
    Source("gitlab", "doc/user/project/repository", "projects_repo", "doc", GITLAB),
    Source("gitlab", "doc/user/project/issues", "projects_repo", "doc", GITLAB),
    Source("gitlab", "doc/topics/git", "projects_repo", "doc", GITLAB),
]

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
HISTORY = re.compile(r"\{\{<\s*history\s*>\}\}.*?\{\{<\s*/history\s*>\}\}", re.DOTALL)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
BLANKS = re.compile(r"\n{3,}")
H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


@dataclass
class Document:
    doc_id: str
    title: str
    area: str
    repo: str
    path: str
    url: str
    commit: str
    text: str


def commit_of(repo: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(RAW / repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t", "-")):
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, text[match.end() :]


def clean(body: str) -> str:
    body = HISTORY.sub("", body)
    body = COMMENT.sub("", body)
    body = SHORTCODE.sub("", body)
    body = IMAGE.sub(r"\1", body)
    body = LINK.sub(r"\1", body)
    body = BLANKS.sub("\n\n", body)
    return body.strip()


def doc_url(rel: Path, url_base: str) -> str:
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] in ("index", "_index"):
        parts.pop()
    return url_base + "".join(p + "/" for p in parts)


def find_title(meta: dict[str, str], body: str, fallback: str) -> str:
    if meta.get("title"):
        return meta["title"]
    match = H1.search(body)
    return match.group(1).strip() if match else fallback


def iter_files(src: Source) -> list[Path]:
    target = RAW / src.repo / src.subpath
    if target.is_file():
        return [target]
    return sorted(target.rglob("*.md"))


def load_source(src: Source, commit: str) -> list[Document]:
    site_root = RAW / src.repo / src.root
    docs: list[Document] = []
    for path in iter_files(src):
        if path.name in SKIP_NAMES:
            continue
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
        if "redirect_to" in meta:
            continue
        text = clean(body)
        if len(text) < MIN_CHARS:
            continue
        rel = path.relative_to(site_root)
        docs.append(
            Document(
                doc_id=f"{src.repo}:{rel.as_posix()}",
                title=find_title(meta, text, path.stem),
                area=src.area,
                repo=src.repo,
                path=f"{src.root}/{rel.as_posix()}",
                url=doc_url(rel, src.url_base),
                commit=commit,
                text=text,
            )
        )
    return docs


def main() -> None:
    commits = {repo: commit_of(repo) for repo in {s.repo for s in SOURCES}}
    seen: set[str] = set()
    docs: list[Document] = []
    for src in SOURCES:
        for doc in load_source(src, commits[src.repo]):
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                docs.append(doc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for doc in docs:
        stats[doc.area][0] += 1
        stats[doc.area][1] += len(doc.text)

    print(f"{'area':<16}{'docs':>6}{'chars':>12}{'est_chunks':>12}")
    total_docs = total_chars = 0
    for area, (n, chars) in sorted(stats.items()):
        est = round(chars / 4 / 450)
        print(f"{area:<16}{n:>6}{chars:>12}{est:>12}")
        total_docs += n
        total_chars += chars
    print(f"{'TOTAL':<16}{total_docs:>6}{total_chars:>12}{round(total_chars / 4 / 450):>12}")
    print(f"Wrote {total_docs} documents to {OUT}")
    for repo, commit in sorted(commits.items()):
        print(f"commit {repo}: {commit}")


if __name__ == "__main__":
    main()
