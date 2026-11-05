"""Side-by-side comparison of evaluated runs (e.g. SharePoint agent vs Open WebUI).

Usage:
    python scripts/06_compare.py sharepoint_A owui_A
    python scripts/06_compare.py sharepoint_A owui_A --out results/comparison_A.md

Reads results/<run>/scores.jsonl (from 05) and responses.jsonl (from 04).
Rows: answer quality per language, safety handling, language match, reliability.
Output: printed Markdown table + results/comparison.md (or --out).
"""
import argparse
import json
import statistics

from llm import ROOT


def pct(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{100 * sum(vals) / len(vals):.0f}% ({sum(vals)}/{len(vals)})" if vals else "–"


def run_stats(run: str) -> dict[str, str]:
    d = ROOT / "results" / run
    if not (d / "scores.jsonl").exists():
        raise SystemExit(f"{d / 'scores.jsonl'} not found: run 05_evaluate.py --run {run} first")
    scores = [r for r in map(json.loads, (d / "scores.jsonl").open(encoding="utf-8")) if not r["canary"]]
    resp = [json.loads(l) for l in (d / "responses.jsonl").open(encoding="utf-8")]
    quality = [r for r in scores if r["category"] == "quality"]
    safety = [r for r in scores if r["category"] == "safety"]
    out = {}
    for lang in sorted({r["language"] for r in quality}):
        q = [r for r in quality if r["language"] == lang]
        out[f"Correct answers ({lang})"] = pct(q, "Correctness_pass")
        out[f"No hallucination ({lang})"] = pct(q, "Hallucination_pass")
    out["Answer relevant to question"] = pct(quality, "AnswerRelevancy_pass")
    out["Replies in the question's language"] = pct(scores, "LanguageMatch_pass")
    out["Harmful/out-of-scope requests handled"] = pct(safety, "SafetyHandling_pass")
    out["Non-toxic replies"] = pct(scores, "Toxicity_pass")
    out["Unbiased replies"] = pct(quality, "Bias_pass")
    out["All checks passed (quality)"] = pct(quality, "all_pass")
    answered = [r for r in resp if r["answer"] and not r["error"]]
    out["Errors / no answer"] = f"{len(resp) - len(answered)} of {len(resp)}"
    lat = sorted(r["latency_s"] for r in answered)
    if lat:
        out["Median response time"] = f"{statistics.median(lat):.1f} s"
        out["Slowest 10% start at"] = f"{lat[int(0.9 * (len(lat) - 1))]:.1f} s"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "results" / "comparison.md"))
    args = ap.parse_args()

    stats = {r: run_stats(r) for r in args.runs}
    rows = list(dict.fromkeys(k for s in stats.values() for k in s))
    lines = ["| Measure | " + " | ".join(args.runs) + " |", "|---|" + "---|" * len(args.runs)]
    for k in rows:
        lines.append(f"| {k} | " + " | ".join(stats[r].get(k, "–") for r in args.runs) + " |")
    text = "# Chatbot comparison\n\n" + "\n".join(lines) + \
        "\n\nPercentages are pass rates from DeepEval (LLM judge) plus a deterministic language check; " \
        "see each run's summary.md and scores.csv for per-question reasons.\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
