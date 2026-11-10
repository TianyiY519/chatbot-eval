"""Check every raw Ragas case against its source, keep the best N, add translations.

Usage:
    python scripts/03_verify_testset.py --set A [--keep 100] [--languages en,fr]

Checks per case
  deterministic : source page found, every number in the answer appears in the
                  source text, question doesn't say "the document/context",
                  near-duplicate questions dropped
  LLM judge     : answer fully supported by the source, question self-contained
                  and unambiguous, answerable from the source alone

Translation: each kept case is translated (question + reference answer) into every
extra language; the copies share a pair_id so results can be compared per language.
Codes: en fr es ar zh ru (the six UN official languages).

Output
  data/testset_<SET>.jsonl         final cases, all languages (input to 04/05)
  data/testset_<SET>.csv           same, one row per case/language, for sharing
  data/testset_<SET>_review.csv    every raw case + verdict + reason, for a human spot check
"""
import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from langchain_core.messages import HumanMessage, SystemMessage

from llm import ROOT, langchain_chat

BANNED = re.compile(r"\b(the|this|given|provided) (document|context|text|page|passage|excerpt)\b"
                    r"|\b(chapter|box|figure|table|annex|section) \d", re.I)  # users don't know section numbers
NUM = re.compile(r"\d[\d,]*\.?\d*")

JUDGE_SYSTEM = """You audit test cases for a document-grounded chatbot.
Given SOURCE text, a QUESTION and a REFERENCE ANSWER, return JSON only:
{"supported": true|false,       // every claim in the answer is stated in SOURCE
 "self_contained": true|false,  // a user without the SOURCE in front of them could ask it; no 'the document/context'
 "unambiguous": true|false,     // one clear correct answer; names report/year/region when needed
 "useful": true|false,          // a realistic question a staff member would ask (not trivia about page layout)
 "reason": "<one short sentence>"}"""


LANGUAGES = {"en": "English", "fr": "French", "es": "Spanish", "ar": "Arabic", "zh": "Chinese (Simplified)",
             "ru": "Russian"}

TRANSLATE_SYSTEM = """Translate a chatbot test question and its reference answer from English into {lang}.
Keep every number, date, name and document title exact (official {lang} title if one exists, otherwise keep
the English title). Write the question the way a native {lang}-speaking UN staff member would ask it.
Return JSON only: {{"question": "...", "reference_answer": "..."}}"""


def parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def translate(model, case: dict, code: str) -> dict:
    msg = json.dumps({"question": case["question"], "reference_answer": case["reference_answer"]}, ensure_ascii=False)
    out = parse_json(model.invoke([SystemMessage(TRANSLATE_SYSTEM.format(lang=LANGUAGES[code])),
                                   HumanMessage(msg)]).content)
    if not out.get("question") or not out.get("reference_answer"):
        raise ValueError(f"translation to {code} failed for {case['id']}")
    return out


def same_fact(case: dict, kept: list[dict]) -> bool:
    """Same source page and a similar answer or a shared figure: Ragas often asks twice about one fact."""
    figs = lambda t: {n for n in re.findall(r"\d[\d.]*\d|\d", t.replace(",", "")) if len(n) > 1
                      and not re.fullmatch(r"(19|20)\d\d", n)}
    for k in kept:
        if k["sources"] != case["sources"]:
            continue
        if SequenceMatcher(None, k["reference_answer"].lower(), case["reference_answer"].lower()).ratio() > 0.5 \
                or figs(k["reference_answer"]) & figs(case["reference_answer"]):
            return True
    return False


def numbers(s: str) -> set[str]:
    return {n.rstrip(".").replace(",", "") for n in NUM.findall(s) if len(n.strip(".,")) > 0}


def judge(model, case: dict) -> dict:
    source = "\n\n---\n\n".join(case["reference_contexts"])[:12000]
    msg = f"SOURCE:\n{source}\n\nQUESTION: {case['question']}\n\nREFERENCE ANSWER: {case['reference_answer']}"
    raw = model.invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(msg)]).content
    return parse_json(raw) or {"reason": f"unparseable: {raw[:80]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--keep", type=int, default=100)
    ap.add_argument("--languages", default="en,fr", help=f"comma list from {','.join(LANGUAGES)}; en is the source")
    ap.add_argument("--workers", type=int, default=8, help="parallel LLM calls")
    args = ap.parse_args()
    langs = [l.strip() for l in args.languages.split(",")]
    if bad := [l for l in langs if l not in LANGUAGES]:
        raise SystemExit(f"Unknown language code(s): {bad}")

    raw = [json.loads(l) for l in (ROOT / "data" / f"testset_{args.set}_raw.jsonl").open(encoding="utf-8")]
    model = langchain_chat("judge", temperature=0)

    # deterministic checks first; only clean drafts go to the (slow, paid) judge, several at a time
    problems_of = {}
    for c in raw:
        problems = []
        if not c["reference_contexts"] or "UNMATCHED" in c["sources"]:
            problems.append("source page not traceable")
        ctx = " ".join(c["reference_contexts"]).replace(",", "")
        missing = [n for n in numbers(c["reference_answer"]) if n not in ctx]
        if missing:
            problems.append(f"numbers not in source: {', '.join(missing[:4])}")
        if BANNED.search(c["question"]):
            problems.append("refers to 'the document/context' or a section number")
        problems_of[c["id"]] = problems
    to_judge = [c for c in raw if not problems_of[c["id"]]]
    print(f"judging {len(to_judge)} of {len(raw)} drafts ({args.workers} at a time) ...", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        verdicts = dict(zip((c["id"] for c in to_judge), pool.map(lambda c: judge(model, c), to_judge)))

    reviewed, kept_questions, kept = [], [], []
    for i, c in enumerate(raw, 1):
        problems, verdict = problems_of[c["id"]], verdicts.get(c["id"], {})
        for k in ("supported", "self_contained", "unambiguous", "useful"):
            if not problems and verdict.get(k) is not True:
                problems.append(f"judge: not {k}")
        if not problems and any(SequenceMatcher(None, c["question"].lower(), q.lower()).ratio() > 0.85
                                for q in kept_questions):
            problems.append("near-duplicate question")
        if not problems and same_fact(c, kept):
            problems.append("same fact as another question on the same page")
        ok = not problems
        if ok:
            kept_questions.append(c["question"])
            kept.append(c)
        reviewed.append({**c, "ok": ok, "problems": "; ".join(problems), "judge_reason": verdict.get("reason", "")})
        print(f"[{i}/{len(raw)}] {'KEEP' if ok else 'DROP'} {c['id']} {'' if ok else '- ' + problems[0]}", flush=True)

    # keep up to N, round-robin across documents so no single report dominates
    passing = [r for r in reviewed if r["ok"]]
    by_doc: dict[str, list] = {}
    for r in passing:
        by_doc.setdefault(r["sources"][0].split("#")[0], []).append(r)
    final = []
    while len(final) < args.keep and any(by_doc.values()):
        for doc in sorted(by_doc):
            if by_doc[doc] and len(final) < args.keep:
                final.append(by_doc[doc].pop(0))

    pairs = [(f"{args.set}{n:03d}", r) for n, r in enumerate(final, 1)]
    jobs = [(pair, r, code) for pair, r in pairs for code in langs if code != "en"]
    print(f"translating {len(pairs)} cases into {', '.join(l for l in langs if l != 'en') or '-'} ...", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        translated = dict(zip(((p, c) for p, _, c in jobs),
                              pool.map(lambda j: translate(model, {**j[1], "id": j[0]}, j[2]), jobs)))
    cases = []
    for pair, r in pairs:
        base = {"pair_id": pair, "reference_contexts": r["reference_contexts"], "sources": r["sources"],
                "synthesizer": r["synthesizer"]}
        for code in langs:
            qa = ({"question": r["question"], "reference_answer": r["reference_answer"]} if code == "en"
                  else translated[(pair, code)])
            cases.append({"id": f"{pair}-{code}", "language": code, **qa, **base})

    out = ROOT / "data" / f"testset_{args.set}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "pair_id", "language", "question", "reference_answer", "sources"])
        for c in cases:
            w.writerow([c["id"], c["pair_id"], c["language"], c["question"], c["reference_answer"],
                        " | ".join(c["sources"])])

    review = ROOT / "data" / f"testset_{args.set}_review.csv"
    with review.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: opens cleanly in Excel
        w = csv.writer(f)
        w.writerow(["raw_id", "ok", "question", "reference_answer", "sources", "problems", "judge_reason", "human_ok"])
        for r in reviewed:
            w.writerow([r["id"], r["ok"], r["question"], r["reference_answer"], " | ".join(r["sources"]),
                        r["problems"], r["judge_reason"], ""])

    print(f"\n{len(passing)}/{len(raw)} passed checks; kept {len(final)} x {len(langs)} languages "
          f"= {len(cases)} cases -> {out}")
    print("per document:", dict(Counter(r["sources"][0].split("#")[0] for r in final)))
    print(f"review sheet -> {review}")
    if len(final) < args.keep:
        print(f"Only {len(final)} cases passed. Re-run 02 with a larger --size or --pages-per-doc.")


if __name__ == "__main__":
    main()
