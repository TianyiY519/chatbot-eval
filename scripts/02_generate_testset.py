"""Generate a raw question/reference-answer set with Ragas.

Usage (normally called by make_testset.py):
    python scripts/02_generate_testset.py --set A --half first   # ~140 raw -> verify keeps 100
    python scripts/02_generate_testset.py --set B --half second
    python scripts/02_generate_testset.py --set trial --size 15   # cheap trial run

--half first/second draw from *disjoint* pages of the same documents, so two
sets test different content from the same knowledge base.
Output: data/testset_<SET>_raw.jsonl (every case keeps source file + page).
"""
import argparse
import json
import random
import re
from collections import defaultdict

from langchain_core.documents import Document
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import (
    MultiHopSpecificQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)

from llm import ROOT, langchain_chat, langchain_embeddings

PAGES = ROOT / "data" / "pages.jsonl"

PERSONAS = [
    Persona(name="Programme officer",
            role_description="UN staff member preparing a briefing who needs precise figures, dates and findings from the reports."),
    Persona(name="Policy analyst",
            role_description="Analyst comparing trends and explaining causes and recommendations described in the reports."),
    Persona(name="New staff member",
            role_description="Recently joined colleague asking plain-language questions about what a report says on a topic."),
]

LLM_CONTEXT = (
    "The questions will be asked to an internal UN chatbot grounded on the organisation's documents "
    "(reports, papers, presentations). "
    "Each question must be self-contained: name the report, organisation or year when needed, never say "
    "'the document', 'the context', 'this page' or 'the text'. "
    "Each reference answer must be concise (1-3 sentences) and use only facts stated in the context."
)


def is_body_page(rec: dict, n_pages: int) -> bool:
    """Prose pages only: skip front matter, back matter (references/annex tables) and table-heavy pages."""
    pos = rec["page"] / n_pages
    if n_pages >= 20 and (pos < 0.06 or pos > 0.85):  # only long reports have front/back matter
        return False
    text = rec["text"]
    if not 800 <= len(text) <= 7000:
        return False
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    if letters / len(text) < 0.65 or digits / max(letters, 1) > 0.12:
        return False
    if re.search(r"\b(References|Bibliography|Notes)\b\s*\n|\bANNEX\b|\bAnnex \d|TECHNICAL NOTE|Statistical annex",
                 text[:200], re.IGNORECASE):
        return False
    return True


def sample_pages(half: str, pages_per_doc: int, seed: int, max_chunks: int = 10**9) -> list[dict]:
    by_doc = defaultdict(list)
    for line in PAGES.open(encoding="utf-8"):
        rec = json.loads(line)
        by_doc[rec["source"]].append(rec)
    picked = []
    for source, recs in sorted(by_doc.items()):
        n_pages = recs[0].get("n_units") or max(r["page"] for r in recs)
        body = [r for r in recs if is_body_page(r, n_pages)]
        random.Random(seed).shuffle(body)
        # fixed split: even positions -> first half, odd -> second (never overlap)
        pool = {"first": body[0::2], "second": body[1::2]}.get(half, body)
        chosen = pool[:pages_per_doc]
        picked += chosen
        print(f"{source}: {len(body)} body pages, using {len(chosen)}")
    if len(picked) > max_chunks:  # big libraries: spread the budget over a random subset of pages
        picked = random.Random(seed).sample(picked, max_chunks)
        print(f"capped at {max_chunks} pages (--max-chunks)")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="name of the test set, e.g. A")
    ap.add_argument("--half", choices=["first", "second", "all"], default="all",
                    help="which disjoint half of the pages to draw from")
    ap.add_argument("--size", type=int, default=140, help="raw cases to generate (verification drops some)")
    ap.add_argument("--pages-per-doc", type=int, default=8)
    ap.add_argument("--max-chunks", type=int, default=120, help="cap on pages sent to Ragas (cost control)")
    ap.add_argument("--multi-hop", type=float, default=0.2, help="share of multi-hop questions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pages = sample_pages(args.half, args.pages_per_doc, args.seed, args.max_chunks)
    chunks = [Document(page_content=p["text"], metadata={"source": p["source"], "page": p["page"]})
              for p in pages]

    llm = LangchainLLMWrapper(langchain_chat("gen", temperature=0.3))
    emb = LangchainEmbeddingsWrapper(langchain_embeddings())
    gen = TestsetGenerator(llm=llm, embedding_model=emb, persona_list=PERSONAS, llm_context=LLM_CONTEXT)

    dist = [(SingleHopSpecificQuerySynthesizer(llm=llm, llm_context=LLM_CONTEXT), 1 - args.multi_hop)]
    if args.multi_hop > 0:
        dist.append((MultiHopSpecificQuerySynthesizer(llm=llm, llm_context=LLM_CONTEXT), args.multi_hop))

    try:
        testset = gen.generate_with_chunks(chunks, testset_size=args.size, query_distribution=dist,
                                           run_config=RunConfig(max_workers=8, timeout=180),
                                           raise_exceptions=False)
    except ValueError as e:
        # multi-hop needs overlapping entities between pages; fall back to single-hop only
        if args.multi_hop == 0:
            raise
        print(f"Multi-hop unavailable ({e}); retrying single-hop only")
        dist = [(dist[0][0], 1.0)]
        testset = gen.generate(testset_size=args.size, query_distribution=dist,
                               run_config=RunConfig(max_workers=8, timeout=180), raise_exceptions=False)

    # map every reference context back to its page so sources can be checked
    page_of = {p["text"]: f'{p["source"]}#{p.get("loc", "page " + str(p["page"]))}' for p in pages}
    out = ROOT / "data" / f"testset_{args.set}_raw.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for s in testset.samples:
            es = s.eval_sample
            if not es.user_input or not es.reference:
                continue
            ctxs = es.reference_contexts or []
            n += 1
            f.write(json.dumps({
                "id": f"{args.set}{n:03d}",
                "question": es.user_input.strip(),
                "reference_answer": es.reference.strip(),
                "reference_contexts": ctxs,
                "sources": sorted({page_of.get(c, "UNMATCHED") for c in ctxs}),
                "synthesizer": s.synthesizer_name,
            }, ensure_ascii=False) + "\n")
    print(f"\n{n} raw cases -> {out}")


if __name__ == "__main__":
    main()
