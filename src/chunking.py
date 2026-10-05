"""Split document sections into overlapping chunks for embedding.

Prose is split recursively on paragraph -> line -> sentence -> word boundaries so
chunks rarely cut a sentence in half. Table sections are split only on row
boundaries so a row is never broken across chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import settings
from .ingestion import LoadedDocument

_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", " "]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def _split_recursive(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:  # hard cut as a last resort
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = [p for p in re.split(sep, text) if p.strip()]
    if len(parts) == 1:
        return _split_recursive(text, size, rest)
    out: list[str] = []
    for p in parts:
        out.extend(_split_recursive(p, size, rest) if len(p) > size else [p])
    return out


def _merge_with_overlap(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # start the overlap at a word boundary
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = f"{tail} {piece}".strip()
            if len(current) > size:
                current = piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def split_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    return _merge_with_overlap(_split_recursive(text, size, _SEPARATORS), size, overlap)


def split_table_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    header, *rows = text.split("\n")
    chunks, current = [], header
    for row in rows:
        if len(current) + len(row) + 1 > max_chars and current != header:
            chunks.append(current)
            current = header + " (continued)"
        current += "\n" + row
    chunks.append(current)
    return chunks


_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.*\S)")


def _split_markdown_table(table: str, max_chars: int, context: str) -> list[str]:
    """Split a Markdown table on row boundaries, repeating the header row(s) in every piece."""
    lines = table.split("\n")
    sep = next((i for i, ln in enumerate(lines) if re.fullmatch(r"\|?[\s:|-]+\|?", ln.strip()) and "-" in ln), 0)
    end = sep + 1
    while end < len(lines) and re.match(r"^\s*\|\s*\|", lines[end]):  # sub-header rows (empty first cell)
        end += 1
    header, rows = lines[:end], lines[end:]
    prefix = (f"{context} (table)\n" if context else "") + "\n".join(header)
    pieces, current = [], []
    for row in rows:
        if current and len(prefix) + sum(len(r) + 1 for r in current) + len(row) > max_chars:
            pieces.append(prefix + "\n" + "\n".join(current))
            current = []
        current.append(row)
    if current or not pieces:
        pieces.append(prefix + ("\n" + "\n".join(current) if current else ""))
    return pieces


def split_prose_with_tables(text: str) -> list[str]:
    """Split page text; Markdown tables (from PDFs) are kept whole or split by rows with their header."""
    blocks: list[tuple[str, str, str]] = []  # (kind, text, nearest heading)
    heading, buf, kind = "", [], "text"

    def flush():
        if buf and "\n".join(buf).strip():
            blocks.append((kind, "\n".join(buf), heading))

    for line in text.split("\n"):
        is_table = line.lstrip().startswith("|")
        new_kind = "table" if is_table else "text"
        if new_kind != kind:
            flush()
            buf, kind = [], new_kind
        buf.append(line)
        m = _HEADING_RE.match(line)
        if m and not is_table:
            heading = m.group(1).strip("* ")
    flush()

    out: list[str] = []
    for kind, block, ctx in blocks:
        if kind == "table":
            out.extend(_split_markdown_table(block, settings.chunk_size * 2, ctx))
        else:
            out.extend(split_text(block))
    return out


def chunk_document(doc: LoadedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s_idx, section in enumerate(doc.sections):
        if "table" in section.metadata:
            pieces = split_table_text(section.text, settings.chunk_size * 2)
        else:
            pieces = split_prose_with_tables(section.text)
        for p_idx, piece in enumerate(pieces):
            meta = dict(section.metadata, file_hash=doc.file_hash, chunk_index=len(chunks))
            chunks.append(Chunk(f"{doc.file_hash[:16]}-{s_idx}-{p_idx}", piece, meta))
    return chunks
