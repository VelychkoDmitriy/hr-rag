"""Ingestion: raw documents -> clean structured text -> deduplicated chunks with metadata.

    python main.py ingest

Steps: extract (per source type) -> normalize -> document dedup -> chunk (by structure; fixed windows as a
baseline) -> chunk dedup across versions -> metadata -> data/processed/.
Outputs:
    documents.jsonl      clean text per document (silver layer)
    chunks.jsonl         structure-aware chunks with metadata: the index reads this
    chunks_fixed.jsonl   fixed-size baseline, only for the comparison
    manifest.json        per-document status, hashes, chunk counts, dataset_version
    audit.jsonl          what happened to every document and why
    chunking_report.md   structure-aware vs fixed: numbers for the README
"""
import csv
import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import pymupdf4llm
import tiktoken
import yaml
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
PIPELINE_VERSION = "ingest-v2"   # v2: layout-aware PDF, stitched tables, nested FAQ, preamble merge

MAX_TOKENS = 450          # structure-aware chunk limit (cl100k tokens); see README "Chunking"
MIN_TOKENS = 40           # a smaller section is merged with the next one under the same parent
FIXED_TOKENS, FIXED_OVERLAP = 400, 60
NEAR_DUP_JACCARD = 0.8
MIN_CHARS_PER_PAGE = 50
CSV_ROWS_PER_CHUNK = 15

ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Doc:
    id: str
    meta: dict
    path: Path
    format: str
    raw_hash: str = ""
    text: str = ""
    status: str = "ok"            # ok | ocr_required | exact_duplicate | error
    note: str = ""
    doc_hash: str = ""
    near_duplicates: list = field(default_factory=list)
    chunks: list = field(default_factory=list)


# ================================================================ 1. extract

LAW_ARTICLE = re.compile(r"^(Стаття|Article)\s+\d+(?:[\-–]\d+)*\.?(\s|$)")
LAW_PART = re.compile(r"^(Розділ|РОЗДІЛ|Глава|ГЛАВА|Частина|ЧАСТИНА|Chapter|CHAPTER|Section|SECTION|Part|PART)\s+"
                      r"([IVXLC]+|\d+)(?:[\-–]\d+)?\b")


def is_upper_title(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and len(text) < 160 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def extract_zakon(html: str) -> str:
    """zakon.rada.gov.ua print view: <div id="article">, one <p> per paragraph; old acts are preformatted text."""
    soup = BeautifulSoup(html, "html.parser")
    art = soup.select_one("div#article") or soup.body
    for tag in art(["script", "style", "button"]):
        tag.decompose()
    paras = art.find_all("p")
    if len(paras) < 5:                         # ILO conventions: one <pre>-like block with hard line breaks
        return preformatted_to_text(art.get_text("\n"))
    lines: list[str] = []
    for p in paras:
        t = " ".join(p.get_text(" ").split())
        if not t:
            continue
        t = re.sub(r"^(Стаття|Article)\s+(\d+)\s*-\s*(\d+)\s*\.", r"\1 \2-\3.", t)   # "Стаття 42 - 7 ." -> "42-7."
        if LAW_ARTICLE.match(t):
            lines.append("### " + t)
        elif LAW_PART.match(t):
            lines.append("## " + t)
        elif lines and lines[-1].startswith("## ") and is_upper_title(t) and len(lines[-1]) < 40:
            lines[-1] += ". " + t              # "Глава III" + "ТРУДОВИЙ ДОГОВІР" on the next line
        else:
            lines.append(t)
    return "\n\n".join(lines)


def preformatted_to_text(raw: str) -> str:
    """Hard-wrapped text: an indented line or a blank line starts a paragraph; "Стаття N" lines are headings."""
    out: list[str] = []
    cur: list[str] = []

    def flush():
        if cur:
            out.append(" ".join(cur))
            cur.clear()

    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            flush()
            continue
        if re.match(r"^(Стаття|Article)\s+\d+\.?$", s):
            flush()
            out.append("### " + s)
            continue
        if LAW_PART.match(s) and len(s) < 60:
            flush()
            out.append("## " + s)
            continue
        if line.startswith("     ") and cur:
            flush()
        cur.append(s)
    flush()
    return "\n\n".join(out)


def html_blocks(container) -> list[str]:
    """Generic article HTML -> markdown-like blocks. A bold or question-like short paragraph becomes a heading."""
    blocks: list[str] = []
    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if el.find_parent("table") is not None and el.name != "table":
            continue
        if el.name == "table":
            rows = [[" ".join(c.get_text(" ").split()) for c in tr.find_all(["td", "th"])] for tr in el.find_all("tr")]
            rows = [r for r in rows if any(r)]
            if rows:
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
                blocks.append("\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows[1:]]))
            continue
        text = " ".join(el.get_text(" ").split())
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            level = {"h1": 2, "h2": 2, "h3": 3, "h4": 3}[el.name]
            blocks.append("#" * level + " " + text)
        elif el.name == "li":
            blocks.append("- " + text)
        else:
            strong = el.find(["strong", "b"])
            q = " ".join(strong.get_text(" ").split()) if strong else ""
            if q and q.endswith("?") and len(q) < 300:
                blocks.append("### " + q)              # FAQ: the bold question is the section
                rest = text[len(q):].strip() if text.startswith(q) else text.replace(q, "", 1).strip()
                if rest:
                    blocks.append(rest)
            elif q and q == text and len(text) < 200:
                blocks.append("### " + text)
            else:
                blocks.append(text)
    return blocks


def extract_web(html: str, doc_id: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "aside"]):
        tag.decompose()
    if doc_id == "dsp_south_faq":                      # nested accordions: topic panel -> question panels
        blocks = []
        for outer in soup.select(".wpsm_panel"):
            if outer.find_parent(class_="wpsm_panel") is not None:
                continue                                # a question panel, handled inside its topic
            title = outer.select_one(".ac_title_class")
            blocks.append("## " + " ".join(title.get_text(" ").split()))
            for inner in outer.select(".wpsm_panel"):
                q = inner.select_one(".ac_title_class")
                a = inner.select_one(".wpsm_panel-body")
                if q and a:
                    blocks.append("### " + " ".join(q.get_text(" ").split()))
                    paras = [" ".join(p.get_text(" ").split()) for p in a.find_all(["p", "li"])]
                    paras = [p for p in paras if p] or [" ".join(a.get_text(" ").split())]
                    blocks += paras
        return "\n\n".join(blocks)
    container = (soup.select_one(".print-content") or soup.select_one("section.single-main-content")
                 or soup.select_one(".entry-content") or soup.find("main") or soup.body)
    return "\n\n".join(html_blocks(container))


def extract_pdf(path: Path) -> tuple[str, bool]:
    """Layout-aware: pymupdf4llm turns font sizes into markdown headings and tables into markdown rows.
    The scan check runs on the plain text layer first: a page without text is flagged, not indexed."""
    doc = pymupdf.open(path)
    chars = sum(len(page.get_text().strip()) for page in doc)
    if chars < MIN_CHARS_PER_PAGE * max(1, doc.page_count):
        return "", True
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    return "\f".join(p["text"] for p in pages), False


DOCX_HEADINGS = {"Title": "# ", "Heading 1": "## ", "Heading 2": "### ", "Heading 3": "### "}
LIST_ITEM = re.compile(r"^(\d+\. |- )")


def table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def extract_docx(path: Path) -> str:
    """Walk the body in document order: paragraphs and tables; styles give headings and numbered lists."""
    doc = Document(path)
    blocks: list[str] = []
    number = 0
    for el in doc.element.body.iterchildren():
        if el.tag.endswith("}tbl"):
            t = Table(el, doc)
            blocks.append(table_to_markdown([[" ".join(c.text.split()) for c in row.cells] for row in t.rows]))
            number = 0
            continue
        if not el.tag.endswith("}p"):
            continue
        para = Paragraph(el, doc)
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style is not None else ""
        if style in DOCX_HEADINGS:
            blocks.append(DOCX_HEADINGS[style] + text)
            number = 0
        elif style in ("List Number", "List Bullet"):
            number += 1
            item = f"{number}. {text}" if style == "List Number" else f"- {text}"
            if blocks and LIST_ITEM.match(blocks[-1].split("\n")[-1]):
                blocks[-1] += "\n" + item               # list items stay together
            else:
                blocks.append(item)
        else:
            blocks.append(text)
            number = 0
    return "\n\n".join(blocks)


def extract_csv(path: Path, title: str) -> str:
    best = from_bytes(path.read_bytes()).best()
    rows = list(csv.reader(io.StringIO(str(best))))
    return f"# {title}\n\n" + table_to_markdown(rows)


def extract(doc: Doc) -> tuple[str, bool]:
    if doc.format == "html":
        html = str(from_bytes(doc.path.read_bytes()).best())
        if "zakon.rada.gov.ua" in doc.meta.get("url", ""):
            return extract_zakon(html), False
        return extract_web(html, doc.id), False
    if doc.format == "pdf":
        return extract_pdf(doc.path)
    if doc.format == "docx":
        return extract_docx(doc.path), False
    if doc.format == "csv":
        return extract_csv(doc.path, doc.meta["title"]), False
    raise ValueError(f"unsupported format {doc.format}")


# ================================================================ 2. normalize

PAGE_NUMBER = re.compile(r"^\s*(page\s+)?\d+\s*(of|/|з|із)\s*\d+\s*$", re.IGNORECASE)
INVISIBLE = re.compile("[​‌‍⁠﻿­]")
EDITORIAL = re.compile(r"\{[^{}]*\}", re.DOTALL)             # zakon.rada amendment notes: {Із змінами, ...}
LINK_ARTIFACT = re.compile(r"\(\s*\d{1,4}[-_][0-9a-zа-яіїєґ]{1,8}\s*\)", re.IGNORECASE)   # "( 2481-14 )"
PHONE = re.compile(r"(?<!\d)(?:\+?38[\s\-]?)?\(?0\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
HEADING = re.compile(r"^(#{1,6} |\d+\.\s+[A-ZА-ЯІЇЄҐ])")


def is_spaced(line: str) -> bool:
    """"La b o r C o d e o f U k r a in e": most tokens are 1-2 letters. Word boundaries are lost."""
    tokens = line.split()
    return len(tokens) >= 6 and sum(len(t) <= 2 for t in tokens) / len(tokens) > 0.7


def remove_repeated_lines(pages: list[str]) -> list[str]:
    """Headers and footers: a line found on more than half of the pages. Needs 2+ pages."""
    if len(pages) < 2:
        return pages
    counts: dict[str, int] = {}
    for page in pages:
        for line in {l.strip() for l in page.split("\n") if l.strip()}:
            counts[line] = counts.get(line, 0) + 1
    repeated = {line for line, n in counts.items()
                if n > len(pages) / 2 and len(line) < 120 and not line.startswith("|")}   # table rows are content
    return ["\n".join(l for l in page.split("\n") if l.strip() not in repeated) for page in pages]


def unwrap(text: str) -> str:
    """Join lines that a PDF broke inside a sentence; keep headings, list items and table rows."""
    out: list[str] = []
    for line in text.split("\n"):
        prev = out[-1] if out else ""
        joinable = (prev and line and len(prev) >= 50 and not HEADING.match(line) and not HEADING.match(prev)
                    and not line.lstrip().startswith(("- ", "|", "• ")) and not prev.lstrip().startswith("|")
                    and not re.match(r"^\d+\.\s", line) and not prev.rstrip().endswith((".", ":", "!", "?", ";")))
        if joinable:
            out[-1] = f"{prev.rstrip()} {line.lstrip()}"
        else:
            out.append(line)
    return "\n".join(out)


def stitch_tables(text: str) -> str:
    """A table cut by a page break comes back as a second table with the same number of columns and no header
    (the PDF parser takes its first data row for a header). Join it to the first table, keeping the real header."""
    blocks = re.split(r"\n\s*\n", text)
    out: list[str] = []
    for block in blocks:
        rows = block.strip().split("\n")
        is_table = len(rows) >= 2 and all(r.startswith("|") for r in rows) and re.match(r"^\|(-+\|)+$", rows[1])
        prev = out[-1].strip().split("\n") if out else []
        prev_table = len(prev) >= 2 and all(r.startswith("|") for r in prev)
        if is_table and prev_table and rows[0].count("|") == prev[0].count("|"):
            out[-1] = "\n".join(prev + [rows[0]] + rows[2:])   # drop the false separator line
        else:
            out.append(block)
    return "\n\n".join(out)


def mask_pii(text: str) -> tuple[str, int]:
    """Phones and real e-mails -> placeholders. The fictional company uses *.example addresses: those stay."""
    n = 0

    def email(m):
        nonlocal n
        if m.group(0).endswith(".example"):
            return m.group(0)
        n += 1
        return "[EMAIL]"

    text = EMAIL.sub(email, text)
    text, k = PHONE.subn("[PHONE]", text)
    return text, n + k


def normalize(text: str, title: str) -> tuple[str, int]:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = INVISIBLE.sub("", text)
    pages = remove_repeated_lines(text.split("\f"))
    pages = ["\n".join(l for l in p.split("\n") if not PAGE_NUMBER.match(l)) for p in pages]
    text = "\n".join(pages)
    text = EDITORIAL.sub("", text)
    text = LINK_ARTIFACT.sub("", text)
    lines = text.split("\n")
    first = next((i for i, l in enumerate(lines) if l.strip()), None)
    if first is not None and is_spaced(lines[first]):
        lines[first] = "# " + title                     # the spaced PDF title: replace with the registry title
    text = "\n".join(l for l in lines if not (is_spaced(l) and len(l) < 120))
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # "reneg-\notiation" -> "renegotiation"
    text = text.replace("**", "")                          # markdown bold from the PDF parser
    text = re.sub(r"(?m)^((?:Стаття|Article)\s+\d+(?:-\d+)*\.[^\n]{0,180})$", r"### \1", text)  # PDF article lines
    text = "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n"))
    text = unwrap(text)
    text = stitch_tables(text)
    text, masked = mask_pii(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text.startswith("# "):
        text = f"# {title}\n\n{text}"
    return text, masked


# ================================================================ 3. dedup

def canonical(text: str) -> str:
    return " ".join(text.split()).casefold()


def shingles(text: str, k: int = 5) -> set:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


# ================================================================ 4. chunk

def split_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def split_long_block(block: str, max_tokens: int) -> list[str]:
    """A paragraph longer than the limit: cut at sentence ends, never inside a table row."""
    if block.lstrip().startswith("|"):
        rows = block.split("\n")
        head, body = rows[:2], rows[2:]
        parts, cur = [], []
        for r in body:
            if cur and n_tokens("\n".join(head + cur + [r])) > max_tokens:
                parts.append("\n".join(head + cur))
                cur = []
            cur.append(r)
        return parts + (["\n".join(head + cur)] if cur else [])
    units = []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ0-9«\"(])", block):      # sentences
        if n_tokens(s) <= max_tokens:
            units.append(s)
            continue
        for item in re.split(r"(?<=;)\s+", s):                            # list items "1) ...; 2) ..."
            if n_tokens(item) <= max_tokens:
                units.append(item)
            else:                                                         # last resort: cut between words
                words, cur = item.split(), []
                for w in words:
                    if cur and n_tokens(" ".join(cur + [w])) > max_tokens:
                        units.append(" ".join(cur))
                        cur = []
                    cur.append(w)
                units.append(" ".join(cur))
    parts, cur = [], ""
    for s in units:
        if cur and n_tokens(cur + " " + s) > max_tokens:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    return parts + ([cur] if cur else [])


def sections(text: str) -> list[dict]:
    """Split markdown-like text on headings; keep the heading path ("Глава III > Стаття 38")."""
    path = {1: "", 2: "", 3: ""}
    out, cur = [], {"path": [], "blocks": []}
    for block in split_blocks(text):
        m = re.match(r"^(#{1,3}) (.+)$", block.split("\n")[0])
        if m and "\n" not in block:
            if cur["blocks"]:
                out.append(cur)
            level = len(m.group(1))
            path[level] = m.group(2).strip()
            for deeper in range(level + 1, 4):
                path[deeper] = ""
            cur = {"path": [path[i] for i in (2, 3) if path[i]], "heading": block, "blocks": []}
        else:
            cur["blocks"].append(block)
    if cur["blocks"]:
        out.append(cur)
    return out


def chunk_by_structure(text: str, max_tokens: int = MAX_TOKENS, min_tokens: int = MIN_TOKENS) -> list[dict]:
    """Section first (chapter > article), then whole paragraphs up to max_tokens. Tables are cut only between rows,
    with the header row repeated. Tiny sections are merged with the next one under the same parent."""
    chunks: list[dict] = []
    for sec in sections(text):
        heading = sec.get("heading", "")
        pieces: list[str] = []
        cur = heading
        for block in sec["blocks"]:
            parts = split_long_block(block, max_tokens) if n_tokens(block) > max_tokens else [block]
            for part in parts:
                candidate = f"{cur}\n\n{part}".strip()
                if cur and cur != heading and n_tokens(candidate) > max_tokens:
                    pieces.append(cur)
                    cur = f"{heading}\n\n{part}".strip()     # the continuation keeps its heading
                else:
                    cur = candidate
        if cur and cur != heading:
            pieces.append(cur)
        section = " > ".join(sec["path"])
        for piece in pieces:
            prev = chunks[-1] if chunks else None
            same_parent = prev and (prev["section"] == "" or prev["section"].split(" > ")[0] == section.split(" > ")[0])
            if (same_parent and n_tokens(prev["text"]) < max(min_tokens, 100 if prev["section"] == "" else 0)
                    and n_tokens(prev["text"] + piece) <= max_tokens):     # the preamble joins the first section
                prev["text"] = f"{prev['text']}\n\n{piece}"
                prev["section"] = section or prev["section"]
            else:
                chunks.append({"text": piece, "section": section})
    return chunks


def chunk_fixed(text: str, size: int = FIXED_TOKENS, overlap: int = FIXED_OVERLAP) -> list[dict]:
    tokens = ENC.encode(text)
    out, start = [], 0
    while start < len(tokens):
        out.append({"text": ENC.decode(tokens[start:start + size]), "section": ""})
        if start + size >= len(tokens):
            break
        start += size - overlap
    return out


def chunk_csv(text: str, title: str) -> list[dict]:
    """A table document: groups of rows, the header row repeated in every chunk."""
    lines = [l for l in text.split("\n") if l.startswith("|")]
    head, rows = lines[:2], lines[2:]
    return [{"text": f"# {title}\n\n" + "\n".join(head + rows[i:i + CSV_ROWS_PER_CHUNK]),
             "section": f"рядки {i + 1}–{min(i + CSV_ROWS_PER_CHUNK, len(rows))}"}
            for i in range(0, len(rows), CSV_ROWS_PER_CHUNK)]


def chunk_stats(chunks: list[dict]) -> dict:
    def mid_sentence(t: str) -> bool:
        body = re.sub(r"^#+ [^\n]*\n+", "", t.strip())       # a continuation chunk starts with its heading
        return bool(body) and (body[0].islower() or body[0] in ",;)")

    def cut_table(t: str) -> bool:
        return any("|" in l and not (l.startswith("|") and l.endswith("|")) for l in t.strip().split("\n"))

    def mixed(t: str) -> bool:                                 # text of two different sections in one chunk
        heads = [m.start() for m in re.finditer(r"(?m)^#{2,3} ", t)]
        return len(heads) > 1 or (len(heads) == 1 and heads[0] > 0 and t[:heads[0]].strip() != "")

    sizes = [n_tokens(c["text"]) for c in chunks] or [0]
    n = max(1, len(chunks))
    return {"chunks": len(chunks), "avg_tokens": round(sum(sizes) / n), "max_tokens": max(sizes),
            "starts_mid_sentence_pct": round(100 * sum(mid_sentence(c["text"]) for c in chunks) / n, 1),
            "cut_table_rows_pct": round(100 * sum(cut_table(c["text"]) for c in chunks) / n, 1),
            "mixed_articles_pct": round(100 * sum(mixed(c["text"]) for c in chunks) / n, 1),
            "with_section_pct": round(100 * sum(bool(c["section"]) for c in chunks) / n, 1)}


# ================================================================ 5. run

def load_registry() -> list[Doc]:
    reg = yaml.safe_load((ROOT / "data" / "sources.yaml").read_text(encoding="utf-8"))
    docs = []
    for group, items in reg.items():
        for m in items:
            if group == "company":
                path = RAW / "company" / m["file"]
            elif group == "manual":
                path = RAW / "manual" / f"{m['id']}.pdf"
            else:
                path = RAW / group / f"{m['id']}.html"
            m = {**m, "group": group, "version": str(m["version"])}
            docs.append(Doc(id=m["id"], meta=m, path=path, format=path.suffix.lstrip(".").lower()))
    return docs


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    audit_lines: list[dict] = []

    def audit(event: str, doc_id: str, **extra):
        audit_lines.append({"ts": now, "event": event, "doc_id": doc_id, **extra})

    docs = load_registry()
    masked_total = 0
    for d in docs:
        if not d.path.exists():
            d.status, d.note = "error", "file not found: run scripts/fetch_corpus.py / make_company_docs.py"
            audit("error", d.id, error=d.note)
            continue
        d.raw_hash = hashlib.sha256(d.path.read_bytes()).hexdigest()
        try:
            raw, needs_ocr = extract(d)
        except Exception as e:                      # one broken file must not stop the batch
            d.status, d.note = "error", f"{type(e).__name__}: {e}"
            audit("error", d.id, error=d.note)
            continue
        if needs_ocr:
            d.status, d.note = "ocr_required", "no text layer: send to OCR, not to the index"
            audit("ocr_required", d.id)
            continue
        d.text, masked = normalize(raw, d.meta["title"])
        masked_total += masked
        if masked:
            audit("pii_masked", d.id, items=masked)
        d.doc_hash = sha256(canonical(d.text))

    # exact duplicates of whole documents: keep the first
    seen: dict[str, Doc] = {}
    for d in docs:
        if d.status != "ok":
            continue
        if d.doc_hash in seen:
            d.status, d.note = "exact_duplicate", f"same content as {seen[d.doc_hash].id}"
            audit("exact_duplicate", d.id, of=seen[d.doc_hash].id)
        else:
            seen[d.doc_hash] = d

    # near duplicates are reported, not dropped: they are versions, kept with is_current in the metadata
    active = [d for d in docs if d.status == "ok"]
    sh = {d.id: shingles(d.text) for d in active}
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            score = jaccard(sh[a.id], sh[b.id])
            if score >= NEAR_DUP_JACCARD:
                a.near_duplicates.append({"doc": b.id, "jaccard": round(score, 3)})
                b.near_duplicates.append({"doc": a.id, "jaccard": round(score, 3)})
                audit("near_duplicate", a.id, other=b.id, jaccard=round(score, 3),
                      action="kept both; is_current marks the version")

    # chunking
    fixed_all, structured_all = [], []
    for d in active:
        fixed = chunk_fixed(d.text)
        structured = chunk_csv(d.text, d.meta["title"]) if d.format == "csv" else chunk_by_structure(d.text)
        fixed_all += [{**c, "doc_id": d.id} for c in fixed]
        structured_all += [{**c, "doc_id": d.id} for c in structured]
        d.chunks = structured

    # chunk dedup: identical text in an outdated version adds nothing -> drop it there, keep the current one.
    # Identical chunks within current documents (repeated FAQ answers) -> keep the first.
    by_hash: dict[str, str] = {}
    dropped = 0
    for d in sorted(active, key=lambda x: not x.meta["is_current"]):      # current documents first
        kept = []
        for c in d.chunks:
            h = sha256(canonical(re.sub(r"^#+ [^\n]*\n+", "", c["text"])))
            if h in by_hash:
                dropped += 1
                continue
            by_hash[h] = d.id
            kept.append(c)
        d.chunks = kept
    audit("chunk_dedup", "*", dropped=dropped)

    # metadata
    records = []
    for d in active:
        m = d.meta
        for i, c in enumerate(d.chunks):
            header = f"{m['title']}" + (f" > {c['section']}" if c["section"] else "")
            records.append({
                "chunk_id": f"{d.id}#{i}", "doc_id": d.id, "title": m["title"], "section": c["section"],
                "header": header, "text": c["text"], "lang": m["lang"], "doc_type": m["doc_type"],
                "format": d.format, "version": m["version"], "version_group": m["version_group"],
                "is_current": m["is_current"], "source": m.get("url") or f"data/raw/company/{m.get('file', '')}",
                "doc_hash": d.doc_hash, "chunk_hash": sha256(canonical(c["text"])), "chunk_index": i,
                "n_tokens": n_tokens(c["text"]), "pipeline_version": PIPELINE_VERSION, "ingested_at": now})
        audit("ingested", d.id, chunks=len(d.chunks))

    files = {d.id: {"format": d.format, "lang": d.meta["lang"], "doc_type": d.meta["doc_type"],
                    "version": d.meta["version"], "is_current": d.meta["is_current"], "status": d.status,
                    "note": d.note, "raw_hash": d.raw_hash, "doc_hash": d.doc_hash, "chunks": len(d.chunks),
                    "near_duplicates": d.near_duplicates} for d in docs}
    dataset_version = sha256(json.dumps({"pipeline": PIPELINE_VERSION,
                                         "files": {k: v["raw_hash"] for k, v in files.items()}}, sort_keys=True))[:16]
    write_jsonl(OUT / "documents.jsonl", [{"doc_id": d.id, "title": d.meta["title"], "text": d.text} for d in active])
    write_jsonl(OUT / "chunks.jsonl", records)
    write_jsonl(OUT / "chunks_fixed.jsonl", fixed_all)
    write_jsonl(OUT / "audit.jsonl", audit_lines)
    stats = {"structure": chunk_stats(structured_all), "fixed": chunk_stats(fixed_all)}
    manifest = {"dataset_version": dataset_version, "pipeline_version": PIPELINE_VERSION, "created_at": now,
                "documents": len(docs), "indexed": len(active), "chunks": len(records), "chunks_deduplicated": dropped,
                "pii_masked": masked_total, "chunking": stats, "files": files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_chunking_report(manifest, structured_all, fixed_all, {d.id: d.meta["doc_type"] for d in active})
    return manifest


def write_chunking_report(m: dict, structured: list[dict], fixed: list[dict], doc_types: dict) -> None:
    """Structure-aware vs fixed windows, overall and per document type, plus one example side by side."""
    lines = ["# Chunking: structure-aware vs fixed windows", "",
             f"dataset `{m['dataset_version']}`, {m['indexed']} documents. Fixed: {FIXED_TOKENS} tokens, overlap "
             f"{FIXED_OVERLAP}. Structure: section > article > whole paragraphs up to {MAX_TOKENS} tokens.", "",
             "| set | strategy | chunks | avg tok | max tok | start mid-sentence | cut table rows | two sections in one chunk | knows its section |",
             "|---|---|---|---|---|---|---|---|---|"]
    for dtype in ["all"] + sorted(set(doc_types.values())):
        for name, chunks in (("structure", structured), ("fixed", fixed)):
            sel = chunks if dtype == "all" else [c for c in chunks if doc_types.get(c["doc_id"]) == dtype]
            if not sel:
                continue
            s = chunk_stats(sel)
            lines.append(f"| {dtype} | {name} | {s['chunks']} | {s['avg_tokens']} | {s['max_tokens']} | "
                         f"{s['starts_mid_sentence_pct']}% | {s['cut_table_rows_pct']}% | {s['mixed_articles_pct']}% | "
                         f"{s['with_section_pct']}% |")
    pairs = [x for x in m["files"].items() if x[1]["near_duplicates"]]
    lines += ["", "## Near-duplicate documents (kept: versions are marked with is_current)", ""]
    lines += [f"- `{k}` ~ " + ", ".join(f"`{n['doc']}` (Jaccard {n['jaccard']})" for n in v["near_duplicates"])
              for k, v in pairs] or ["- none above the threshold"]
    example = next((c for c in fixed if c["doc_id"] == "polozhennia_vidriadzhennia" and "|" in c["text"]), None)
    if example:
        struct = [c for c in structured if c["doc_id"] == "polozhennia_vidriadzhennia" and "|" in c["text"]]
        lines += ["", "## Example: the business-trip table (it runs across two PDF pages)", "",
                  "Fixed window (the chunk that holds the table):", "", "```", example["text"][:1200], "```", "",
                  "Structure-aware (the table chunk):", "", "```", (struct[0]["text"] if struct else "")[:1200], "```"]
    (OUT / "chunking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_summary(m: dict) -> None:
    print(f"{'document':<30} {'fmt':<5} {'status':<14} chunks  near-duplicates")
    for doc_id, f in m["files"].items():
        nd = ", ".join(f"{x['doc']} {x['jaccard']}" for x in f["near_duplicates"])
        print(f"{doc_id:<30} {f['format']:<5} {f['status']:<14} {f['chunks']:>6}  {nd}")
    s, fx = m["chunking"]["structure"], m["chunking"]["fixed"]
    print(f"\n{'chunking':<12} chunks  avg tok  max tok  mid-sentence  cut tables  mixed articles  with section")
    for name, x in (("structure", s), ("fixed", fx)):
        print(f"{name:<12} {x['chunks']:>6} {x['avg_tokens']:>8} {x['max_tokens']:>8} {x['starts_mid_sentence_pct']:>12}% "
              f"{x['cut_table_rows_pct']:>10}% {x['mixed_articles_pct']:>14}% {x['with_section_pct']:>12}%")
    print(f"\ndocuments {m['documents']} | indexed {m['indexed']} | chunks {m['chunks']} | "
          f"deduplicated chunks {m['chunks_deduplicated']} | PII masked {m['pii_masked']} | dataset {m['dataset_version']}")
