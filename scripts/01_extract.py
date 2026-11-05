"""Extract text units from every document in a folder -> data/pages.jsonl.

Usage:
    python scripts/01_extract.py                       # default folder: docs/
    python scripts/01_extract.py --input-dir /path/to/library

Supported: .pdf (one unit per page), .pptx (consecutive slides merged),
.docx / .txt / .md (split into ~3,000-character blocks at paragraph breaks).
Each record keeps the file name and location so every generated test case can
be traced back to where its answer came from.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from llm import ROOT

MIN_CHARS = 400  # skip covers, blank pages, mostly-figure pages
BLOCK = 3000     # target size for non-PDF units


def clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def blocks(paragraphs: list[str], label: str) -> list[tuple[str, str]]:
    """Merge paragraphs (or slides) into ~BLOCK-sized units -> [(location, text)]."""
    out, buf, start = [], [], 1
    for i, p in enumerate(paragraphs, 1):
        buf.append(p)
        if sum(map(len, buf)) >= BLOCK or i == len(paragraphs):
            loc = f"{label} {start}" if start == i else f"{label}s {start}-{i}"
            out.append((loc, "\n\n".join(buf)))
            buf, start = [], i + 1
    return out


def read_pdf(path: Path) -> list[tuple[str, str]]:
    from pypdf import PdfReader

    units = []
    for i, page in enumerate(PdfReader(path).pages, 1):
        try:
            units.append((f"page {i}", page.extract_text() or ""))
        except Exception as e:  # some figure-heavy pages exceed pypdf limits
            print(f"  skip {path.name} p{i}: {type(e).__name__}")
    return units


def read_pptx(path: Path) -> list[tuple[str, str]]:
    from pptx import Presentation

    slides = []
    for slide in Presentation(path).slides:
        texts = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            texts.append("Notes: " + slide.notes_slide.notes_text_frame.text)
        slides.append("\n".join(texts))
    return blocks(slides, "slide")


def read_docx(path: Path) -> list[tuple[str, str]]:
    from docx import Document

    doc = Document(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        paras += [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
    return blocks(paras, "part")


def read_text(path: Path) -> list[tuple[str, str]]:
    paras = [p for p in path.read_text(encoding="utf-8", errors="ignore").split("\n\n") if p.strip()]
    return blocks(paras, "part")


READERS = {".pdf": read_pdf, ".pptx": read_pptx, ".docx": read_docx, ".txt": read_text, ".md": read_text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(ROOT / "docs"))
    ap.add_argument("--out", default=str(ROOT / "data" / "pages.jsonl"))
    args = ap.parse_args()

    files = sorted(p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in READERS)
    if not files:
        sys.exit(f"No {', '.join(READERS)} files found in {args.input_dir}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out.open("w", encoding="utf-8") as f:
        for path in files:
            units = READERS[path.suffix.lower()](path)
            n_doc = 0
            for n, (loc, text) in enumerate(units, 1):
                text = clean(text)
                if len(text) < MIN_CHARS:
                    continue
                f.write(json.dumps({"source": path.name, "page": n, "loc": loc, "n_units": len(units),
                                    "text": text}, ensure_ascii=False) + "\n")
                n_doc += 1
            kept += n_doc
            print(f"{path.name}: {len(units)} units, {n_doc} kept")
    print(f"\n{kept} units from {len(files)} files -> {out}")


if __name__ == "__main__":
    main()
