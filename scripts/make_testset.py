"""Folder of documents -> verified, multilingual Q&A test sets. One command.

Usage:
    python scripts/make_testset.py --input-dir docs/                       # sets A and B, 100 each, en+fr
    python scripts/make_testset.py --input-dir /path/to/library --sets A --keep 200 --languages en,fr,es
    python scripts/make_testset.py --input-dir docs/ --sets trial --keep 10  # cheap dry run

Steps: 01_extract -> 02_generate_testset (Ragas) -> 03_verify_testset (source checks + translation).
Several sets use disjoint pages of the same documents (first set: even pages, second: odd).
Output per set: data/testset_<SET>.jsonl / .csv / _review.csv
"""
import argparse
import json
import math
import subprocess
import sys

from llm import ROOT

PY = sys.executable
SCRIPTS = ROOT / "scripts"


def run(script: str, *args: str) -> None:
    print(f"\n=== {script} {' '.join(args)}")
    subprocess.run([PY, str(SCRIPTS / script), *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(ROOT / "docs"))
    ap.add_argument("--sets", default="A,B", help="one or two names; two names get disjoint pages")
    ap.add_argument("--keep", type=int, default=100, help="verified English questions per set")
    ap.add_argument("--languages", default="en,fr")
    ap.add_argument("--oversample", type=float, default=1.4, help="raw cases per kept case (some get dropped)")
    ap.add_argument("--max-chunks", type=int, default=120)
    args = ap.parse_args()

    sets = [s.strip() for s in args.sets.split(",")]
    if len(sets) > 2:
        raise SystemExit("--sets takes one or two names")
    run("01_extract.py", "--input-dir", args.input_dir)

    n_docs = len({json.loads(l)["source"] for l in (ROOT / "data" / "pages.jsonl").open(encoding="utf-8")})
    per_doc = max(1, min(8, math.ceil(args.max_chunks / n_docs)))
    raw = math.ceil(args.keep * args.oversample)
    halves = ["first", "second"] if len(sets) == 2 else ["all"]
    for name, half in zip(sets, halves):
        run("02_generate_testset.py", "--set", name, "--half", half, "--size", str(raw),
            "--pages-per-doc", str(per_doc), "--max-chunks", str(args.max_chunks))
        run("03_verify_testset.py", "--set", name, "--keep", str(args.keep), "--languages", args.languages)


if __name__ == "__main__":
    main()
