from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

ANSWERS = Path("data/eval/generation_dev_answers.jsonl")
JUDGE = Path("data/eval/generation_dev_judge.jsonl")
SAMPLE = Path("data/eval/spotcheck_sample.jsonl")
LABELS = Path("data/eval/spotcheck_labels.jsonl")
REPORT = Path("evals/rag/judge_spotcheck")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def judged_faithfulness(
    gens: list[dict[str, Any]], judges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scores = {
        (j["config"], j["id"]): j
        for j in judges
        if j["metric"] == "faithfulness" and j["score"] is not None
    }
    out = []
    for g in gens:
        j = scores.get((g["config"], g["id"]))
        if g["refused"] or j is None:
            continue
        out.append({**g, "judge_score": j["score"], "judge_unsupported": j.get("unsupported", [])})
    return out


def draw_sample(
    rows: list[dict[str, Any]], n_flagged: int, n_perfect: int, n_code: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    def key(r: dict[str, Any]) -> tuple[str, str]:
        return r["id"], r["config"]

    flagged = sorted((r for r in rows if r["judge_score"] < 1.0), key=key)
    perfect = sorted((r for r in rows if r["judge_score"] >= 1.0), key=key)
    rng.shuffle(flagged)
    rng.shuffle(perfect)
    used: set[str] = set()
    chosen: list[dict[str, Any]] = []

    def take(pool: list[dict[str, Any]], n: int) -> None:
        taken = 0
        for r in pool:
            if taken >= n:
                break
            if r["id"] in used:
                continue
            used.add(r["id"])
            chosen.append(r)
            taken += 1

    take(flagged, n_flagged)
    take([r for r in perfect if "```" in r["text"]], n_code)
    take(perfect, n_perfect - sum(1 for r in chosen if r["judge_score"] >= 1.0))
    rng.shuffle(chosen)
    return chosen


def agreement(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    n = len(pairs)
    tp = sum(1 for j, h in pairs if j and h)
    tn = sum(1 for j, h in pairs if not j and not h)
    fp = sum(1 for j, h in pairs if j and not h)
    fn = sum(1 for j, h in pairs if not j and h)
    if n == 0:
        return {"n": 0}
    po = (tp + tn) / n
    pj, ph = (tp + fp) / n, (tp + fn) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = (po - pe) / (1 - pe) if pe < 1 else (1.0 if po == 1 else 0.0)
    return {
        "n": n,
        "both_faithful": tp,
        "both_unsupported": tn,
        "judge_lenient": fp,
        "judge_strict": fn,
        "agreement": po,
        "kappa": kappa,
        "judge_faithful_confirmed": tp / (tp + fp) if tp + fp else None,
        "judge_flagged_confirmed": tn / (tn + fn) if tn + fn else None,
    }


def ask(msg: str, valid: str) -> str:
    while True:
        a = input(msg).strip().lower()
        if a and a[0] in valid:
            return a[0]


def cmd_sample(args: argparse.Namespace) -> None:
    if SAMPLE.exists() and not args.force:
        raise SystemExit(
            f"{SAMPLE} exists. Use --force to redraw (this drops your labels' sample)."
        )
    rows = judged_faithfulness(read_jsonl(ANSWERS), read_jsonl(JUDGE))
    if args.exclude:
        old = {r["id"] for r in read_jsonl(Path(args.exclude))}
        rows = [r for r in rows if r["id"] not in old]
    sample = draw_sample(rows, args.flagged, args.perfect, args.code, args.seed)
    SAMPLE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample))
    n_flag = sum(1 for r in sample if r["judge_score"] < 1.0)
    print(f"{len(rows)} judged answers; sampled {len(sample)} ({n_flag} judge-flagged)")


def cmd_label(_: argparse.Namespace) -> None:
    sample = read_jsonl(SAMPLE)
    labeled = {(x["config"], x["id"]) for x in read_jsonl(LABELS)}
    pending = [r for r in sample if (r["config"], r["id"]) not in labeled]
    print(
        "Faithful = EVERY claim (steps, settings, commands, code) is stated in the passages or"
        " follows directly from them. Ignore citation style. Do not use outside knowledge."
    )
    try:
        for n, r in enumerate(pending, start=1):
            cited = set(r["cited_chunk_ids"])
            print("\n" + "=" * 78)
            print(f"[{n}/{len(pending)}]  QUESTION: {r['question']}\n")
            print(f"ANSWER:\n{r['text']}\n")
            for i, p in enumerate(r["passages"], start=1):
                tag = " (cited)" if p["chunk_id"] in cited else ""
                print(f"--- [{i}] {p['title']} | {p['heading_path']}{tag}\n{p['text']}\n")
            action = ask(
                "[f] all supported  [u] something unsupported  [s] skip  [q] quit > ", "fusq"
            )
            if action == "q":
                break
            if action == "s":
                continue
            note = input("which claim (optional): ").strip() if action == "u" else ""
            rec = {
                "config": r["config"],
                "id": r["id"],
                "human_faithful": action == "f",
                "note": note,
            }
            with LABELS.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except (EOFError, KeyboardInterrupt):
        print()
    print(f"\nlabeled {len(read_jsonl(LABELS))}/{len(sample)}")


def cmd_report(_: argparse.Namespace) -> None:
    sample = {(r["config"], r["id"]): r for r in read_jsonl(SAMPLE)}
    labels = [x for x in read_jsonl(LABELS) if (x["config"], x["id"]) in sample]
    pairs = [
        (sample[(x["config"], x["id"])]["judge_score"] >= 1.0, x["human_faithful"]) for x in labels
    ]
    stats = agreement(pairs)
    if not stats["n"]:
        raise SystemExit("No labels yet.")
    disagreements = [
        {
            "id": x["id"],
            "config": x["config"],
            "judge_score": sample[(x["config"], x["id"])]["judge_score"],
        }
        for x in labels
        if (sample[(x["config"], x["id"])]["judge_score"] >= 1.0) != x["human_faithful"]
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.with_suffix(".json").write_text(
        json.dumps({"stats": stats, "disagreements": disagreements}, indent=2)
    )

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0%}"

    md = (
        f"Judge spot-check, {stats['n']} hand-labeled dev answers (oversampled: judge-flagged answers are"
        " over-represented, so overall agreement is not a population rate).\n\n"
        f"- Overall agreement: {pct(stats['agreement'])}, Cohen's kappa {stats['kappa']:.2f}\n"
        f"- Judge said faithful, human confirmed: {pct(stats['judge_faithful_confirmed'])}"
        f" ({stats['both_faithful']} of {stats['both_faithful'] + stats['judge_lenient']})\n"
        f"- Judge flagged, human confirmed unsupported: {pct(stats['judge_flagged_confirmed'])}"
        f" ({stats['both_unsupported']} of {stats['both_unsupported'] + stats['judge_strict']})\n"
        f"- Judge too lenient: {stats['judge_lenient']}, too strict: {stats['judge_strict']}\n"
    )
    REPORT.with_suffix(".md").write_text(md)
    print(md)
    print("Disagreements:", [(d["id"], d["config"]) for d in disagreements])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--flagged", type=int, default=12)
    s.add_argument("--perfect", type=int, default=18)
    s.add_argument("--code", type=int, default=4)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--force", action="store_true")
    s.add_argument("--exclude", help="sample file whose questions must not be reused")
    s.set_defaults(func=cmd_sample)
    sub.add_parser("label").set_defaults(func=cmd_label)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
