"""PDF ingestion: extraction, cleaning and structure-aware chunking.

Pipeline: PDF -> page blocks (text lines + tables as rows) -> logical units
(section / sub-section / FAQ item) -> size-bounded chunks carrying citation metadata.

Design notes
------------
* Chunks follow the document's own structure, so a chunk never mixes two sections and
  every chunk can be cited as "<document> - Section 6.2: <title> (page 3)".
* Tables are extracted as markdown rows (one row per line) instead of one cell per line,
  and are never split across chunks unless a single table exceeds the size budget.
* Each chunk text starts with a breadcrumb (document > section) so it is self-describing
  when retrieved on its own.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pdfplumber

from .utils import clean_text, word_count

logger = logging.getLogger(__name__)

CHUNKER_VERSION = "2"

_SECTION_RE = re.compile(r"^Section\s+(\d+(?:\.\d+)?):\s*(.+)$")
_SUBSECTION_RE = re.compile(r"^(\d+\.\d+)\s+([A-Z][^.]{2,100})$")
_FAQ_RE = re.compile(r"^(Q\d{3,4}):\s*(.+)$")
_FRONT_FIELD_RE = re.compile(r"^(Document Code|Effective Date|Issuing Authority|Classification|Version)\s*:\s*(.+)$", re.I)

_TABLE_TOP_MARGIN = 110.0
_TABLE_BOTTOM_MARGIN = 140.0


@dataclass(frozen=True)
class Block:
    """A line of prose or a whole table, with the page it came from."""

    page: int
    kind: str  # "line" | "table"
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass
class _Unit:
    section_id: str
    section_label: str
    parent_label: str
    faq_id: str
    page_start: int
    blocks: list[Block] = field(default_factory=list)

    @property
    def page_end(self) -> int:
        return max([self.page_start] + [block.page for block in self.blocks])


# --------------------------------------------------------------------------- extraction


def _repair_glyphs(page: "pdfplumber.page.Page") -> None:
    """Map symbol-font glyphs to their real characters.

    These PDFs draw the rupee sign with a ZapfDingbats 'n' (rendered as a black square by
    some extractors) and bullets as ``(cid:127)``. Fixing it per character is exact, unlike a
    regex over the final text.
    """
    for char in page.chars:
        text = char["text"]
        if "ZapfDingbats" in char.get("fontname", "") and text == "n":
            char["text"] = "\u20b9"
        elif text.startswith("(cid:"):
            char["text"] = "\u2022"


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [" ".join((cell or "").split()) for cell in row]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    lines = ["| " + " | ".join(cleaned[0]) + " |", "|" + " --- |" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in cleaned[1:])
    return "\n".join(lines)


def _in_bbox(obj: dict, boxes: list[tuple[float, float, float, float]]) -> bool:
    x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
    return any(
        x0 >= bx0 - 1 and x1 <= bx1 + 1 and top >= btop - 1 and bottom <= bbottom + 1
        for bx0, btop, bx1, bbottom in boxes
    )


def extract_blocks(pdf_path: Path) -> list[Block]:
    """Return the document as an ordered list of text-line and table blocks."""
    blocks: list[Block] = []
    previous_header: list[str] | None = None
    previous_bottom_gap = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            _repair_glyphs(page)
            tables = page.find_tables()
            boxes = [tuple(table.bbox) for table in tables]
            body = page.filter(lambda obj: obj["object_type"] != "char" or not _in_bbox(obj, boxes))

            positioned: list[tuple[float, Block]] = []
            for line in body.extract_text_lines(return_chars=False):
                text = clean_text(line["text"])
                if text:
                    positioned.append((line["top"], Block(page_number, "line", text)))

            for table in tables:
                rows = table.extract()
                header = [" ".join((cell or "").split()) for cell in rows[0]] if rows else []
                continues = (
                    previous_header is not None
                    and previous_bottom_gap is not None
                    and previous_bottom_gap < _TABLE_BOTTOM_MARGIN
                    and table.bbox[1] < _TABLE_TOP_MARGIN
                    and len(header) == len(previous_header)
                    and header != previous_header
                )
                if continues:
                    rows = [previous_header] + rows
                markdown = _table_to_markdown(rows)
                if markdown:
                    positioned.append((table.bbox[1], Block(page_number, "table", markdown)))
                previous_header = previous_header if continues else header
                previous_bottom_gap = page.height - table.bbox[3]

            if not tables:
                previous_bottom_gap = None
            positioned.sort(key=lambda item: item[0])
            blocks.extend(block for _, block in positioned)

    return blocks


# --------------------------------------------------------------------------- structure


def _split_front_matter(blocks: list[Block]) -> tuple[list[Block], list[Block]]:
    for index, block in enumerate(blocks):
        if block.kind == "line" and _SECTION_RE.match(block.text):
            return blocks[:index], blocks[index:]
    return blocks, []


def _document_title(front: list[Block], fallback: str) -> str:
    title_lines: list[str] = []
    for block in front:
        if block.kind != "line":
            continue
        text = block.text
        if _FRONT_FIELD_RE.match(text) or text.lower().startswith("table of contents"):
            break
        title_lines.append(text)
        if len(title_lines) == 3:
            break
    title = " ".join(title_lines).strip()
    return title or fallback


def _front_matter_text(front: list[Block]) -> str:
    fields = [b.text for b in front if b.kind == "line" and _FRONT_FIELD_RE.match(b.text)]
    return "\n".join(fields)


def _build_units(body: list[Block]) -> list[_Unit]:
    units: list[_Unit] = []
    current: _Unit | None = None
    top_label = ""
    top_number = ""
    faq_parent_label = ""

    def open_unit(section_id: str, label: str, parent: str, faq_id: str, page: int) -> _Unit:
        unit = _Unit(section_id, label, parent, faq_id, page)
        units.append(unit)
        return unit

    for block in body:
        if block.kind == "line":
            section = _SECTION_RE.match(block.text)
            if section:
                number, title = section.group(1), section.group(2).strip()
                label = f"Section {number}: {title}"
                if "." in number:
                    current = open_unit(number, label, top_label, "", block.page)
                else:
                    top_label, top_number = label, number
                    faq_parent_label = label
                    current = open_unit(number, label, "", "", block.page)
                continue
            sub = _SUBSECTION_RE.match(block.text)
            if sub and not block.text.rstrip().endswith((".", ":")):
                number, title = sub.group(1), sub.group(2).strip()
                parent = top_label if number.split(".")[0] == top_number else ""
                current = open_unit(number, f"Section {number}: {title}", parent, "", block.page)
                continue
            faq = _FAQ_RE.match(block.text)
            if faq:
                current = open_unit(top_number or "FAQ", faq_parent_label, "", faq.group(1), block.page)
                current.blocks.append(block)
                continue
        if current is None:
            current = open_unit(top_number or "0", top_label or "Introduction", "", "", block.page)
        current.blocks.append(block)

    return [unit for unit in units if any(b.text.strip() for b in unit.blocks)]


# --------------------------------------------------------------------------- chunking


def _pack_blocks(blocks: list[Block], max_words: int, overlap_words: int) -> list[list[Block]]:
    """Greedily pack blocks into groups of at most ``max_words`` words.

    Tables are atomic. When a group is flushed, trailing prose lines worth up to
    ``overlap_words`` words are repeated at the start of the next group.
    """
    groups: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for block in blocks:
        words = word_count(block.text)
        if current and size + words > max_words:
            groups.append(current)
            carry: list[Block] = []
            carried = 0
            for previous in reversed(current):
                if previous.kind != "line" or carried + word_count(previous.text) > overlap_words:
                    break
                carry.insert(0, previous)
                carried += word_count(previous.text)
            current, size = carry, carried
        current.append(block)
        size += words
    if current and (not groups or any(b not in groups[-1] for b in current)):
        groups.append(current)
    return groups


def _page_label(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def build_chunks(pdf_path: Path, max_words: int, overlap_words: int, id_prefix: str | None = None) -> list[Chunk]:
    """Parse one PDF into citation-ready chunks. ``id_prefix`` keeps chunk ids unique across folders."""
    blocks = extract_blocks(pdf_path)
    front, body = _split_front_matter(blocks)
    stem = id_prefix or pdf_path.stem
    title = _document_title(front, fallback=stem.replace("_", " ").title())
    chunks: list[Chunk] = []

    def add(text: str, section_id: str, section_label: str, faq_id: str, p0: int, p1: int) -> None:
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{stem}::{index:04d}",
                text=text,
                metadata={
                    "file_name": pdf_path.name,
                    "document_title": title,
                    "section_id": section_id,
                    "section": section_label,
                    "faq_id": faq_id,
                    "page_start": p0,
                    "page_end": p1,
                    "pages": _page_label(p0, p1),
                    "chunk_index": index,
                },
            )
        )

    front_text = _front_matter_text(front)
    if front_text:
        add(f"[{title}] Document information\n{front_text}", "0", "Document information", "", 1, 1)

    for unit in _build_units(body):
        prefix_parts = [f"[{title}]"]
        if unit.parent_label:
            prefix_parts.append(unit.parent_label)
        prefix_parts.append(unit.section_label + (f" > {unit.faq_id}" if unit.faq_id else ""))
        prefix = " > ".join(prefix_parts)
        for position, group in enumerate(_pack_blocks(unit.blocks, max_words, overlap_words)):
            body_text = "\n".join(block.text for block in group)
            pages = [block.page for block in group]
            if position == 0:
                pages.append(unit.page_start)
            add(f"{prefix}\n{body_text}", unit.section_id, unit.section_label, unit.faq_id, min(pages), max(pages))
    return chunks


def load_corpus(data_dir: Path, max_words: int, overlap_words: int) -> list[Chunk]:
    pdfs = sorted(data_dir.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {data_dir}. Place the knowledge-base PDFs there.")
    corpus: list[Chunk] = []
    for pdf in pdfs:
        prefix = pdf.relative_to(data_dir).with_suffix("").as_posix().replace("/", "__")
        chunks = build_chunks(pdf, max_words, overlap_words, id_prefix=prefix)
        logger.info("Ingested %s -> %d chunks", pdf.name, len(chunks))
        corpus.extend(chunks)
    return corpus


def corpus_fingerprint(paths: Iterable[Path], *parts: object) -> str:
    """Hash of file contents plus every parameter that changes the index."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(CHUNKER_VERSION.encode())
    for part in parts:
        digest.update(repr(part).encode())
    return digest.hexdigest()
