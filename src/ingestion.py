"""Document ingestion: validate uploads and extract text from PDF, TXT, CSV and Excel.

Each loader returns a list of `Section` objects (a piece of text plus metadata
such as file name and page number). Tabular files additionally return their
DataFrames so the table-analysis tool can compute exact numbers.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised when a file cannot be accepted or parsed. Message is user-facing."""


@dataclass
class Section:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class LoadedDocument:
    name: str
    file_hash: str
    doc_type: str
    sections: list[Section]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # table_name -> DataFrame
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_file(path: Path) -> None:
    if not path.exists():
        raise IngestionError(f"File not found: {path.name}")
    ext = path.suffix.lower()
    if ext not in settings.allowed_extensions:
        raise IngestionError(
            f"'{path.name}': unsupported type '{ext}'. Allowed: {', '.join(settings.allowed_extensions)}"
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        raise IngestionError(f"'{path.name}' is empty.")
    if size_mb > settings.max_file_mb:
        raise IngestionError(f"'{path.name}' is {size_mb:.1f} MB; the limit is {settings.max_file_mb} MB.")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    lines = [ln.rstrip() for ln in text.splitlines()]
    # collapse 3+ blank lines into one
    out, blank = [], 0
    for ln in lines:
        blank = blank + 1 if not ln.strip() else 0
        if blank <= 1:
            out.append(ln)
    return "\n".join(out).strip()


_RUPEE_FIXES = [  # many Indian reports map the Rupee symbol to a backtick in their font
    (re.compile(r"`\s?(?=\d)"), "₹"),
    (re.compile(r"`\s?(?=(?:\*\*)?\s?(?:crore|lakh|lakhs|million|billion)\b)", re.IGNORECASE), "₹ "),
    (re.compile(r"\(\s?in\s?`"), "(in ₹"),
]


def _fix_symbols(text: str) -> str:
    # strip Markdown/HTML styling noise from PDF conversion (keeps the table pipes)
    text = text.replace("**", "").replace("<br>", " ")
    text = re.sub(r"</?(?:sup|sub|span|b|i)[^>]*>", "", text)
    text = re.sub(r"`(?=\s?\))", "₹", text)
    for rx, rep in _RUPEE_FIXES:
        text = rx.sub(rep, text)
    return text


def _pdf_pages_layout(path: Path) -> list[tuple[int, str]] | None:
    """Layout-aware extraction (multi-column reading order, tables -> Markdown tables,
    OCR for scanned pages when Tesseract is installed). Returns None if unavailable."""
    try:
        import pymupdf4llm
    except ImportError:
        return None
    try:
        try:
            chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False,
                                             header=False, footer=False)
        except TypeError:  # older pymupdf4llm without header/footer options
            chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
        return [(c.get("metadata", {}).get("page_number") or i + 1, c.get("text", "")) for i, c in enumerate(chunks)]
    except Exception as e:
        logger.warning("Layout-aware PDF parsing failed for %s (%s); using basic extraction.", path.name, e)
        return None


def load_pdf(path: Path) -> tuple[list[Section], list[str]]:
    import pymupdf as fitz  # PyMuPDF

    warnings: list[str] = []
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise IngestionError(f"'{path.name}' could not be opened as a PDF ({e}).") from e
    if doc.needs_pass:
        raise IngestionError(f"'{path.name}' is password-protected. Please upload an unlocked copy.")
    n_pages = doc.page_count

    pages = _pdf_pages_layout(path)
    if pages is None:  # basic fallback: plain text in reading order
        pages = [(i, page.get_text("text")) for i, page in enumerate(doc, start=1)]
    doc.close()

    sections, empty_pages = [], 0
    for page_no, raw in pages:
        text = _clean(_fix_symbols(raw))
        if len(re.sub(r"[\W_]+", "", text)) < 20:
            empty_pages += 1
            continue
        sections.append(Section(text, {"page": int(page_no)}))

    if not sections:
        raise IngestionError(
            f"'{path.name}' has no extractable text (it is probably a scanned image and OCR is not available)."
        )
    if empty_pages:
        warnings.append(f"{empty_pages} of {n_pages} pages had no extractable text and were skipped.")
    return sections, warnings


def load_txt(path: Path) -> tuple[list[Section], list[str]]:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    text = _clean(text)
    if not text:
        raise IngestionError(f"'{path.name}' contains no text.")
    return [Section(text, {})], []


def _read_csv(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")  # sniff delimiter
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            raise IngestionError(f"'{path.name}' could not be parsed as CSV ({e}).") from e
    raise IngestionError(f"'{path.name}' has an unknown text encoding ({last_err}).")


def _tidy_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() if not str(c).startswith("Unnamed") else f"column_{i+1}"
                  for i, c in enumerate(df.columns)]
    return df.reset_index(drop=True)


def table_to_sections(df: pd.DataFrame, table_name: str) -> list[Section]:
    """Turn a DataFrame into (1) a summary section and (2) row-group sections.

    Rows are written as `column: value` pairs so each chunk is self-describing
    and embeds well, e.g. "Row 12 | Region: North | Revenue: 45000".
    """
    sections: list[Section] = []

    # (1) Summary section — lets the LLM answer "what columns / how many rows" questions.
    lines = [f"Table '{table_name}' summary: {len(df)} rows, {len(df.columns)} columns.",
             "Columns: " + ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)]
    num = df.select_dtypes("number")
    for c in num.columns[:20]:
        s = num[c].dropna()
        if len(s):
            lines.append(f"{c}: sum={s.sum():,.2f}, mean={s.mean():,.2f}, min={s.min():,.2f}, max={s.max():,.2f}")
    for c in df.select_dtypes(exclude="number").columns[:10]:
        uniq = df[c].dropna().astype(str).unique()
        if 0 < len(uniq) <= 15:
            lines.append(f"{c} values: {', '.join(uniq)}")
    sections.append(Section("\n".join(lines), {"rows": "summary", "table": table_name}))

    # (2) Row groups
    n = settings.table_rows_per_chunk
    for start in range(0, len(df), n):
        block = df.iloc[start:start + n]
        rows = []
        for idx, row in block.iterrows():
            cells = " | ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v) and str(v).strip())
            rows.append(f"Row {idx + 1} | {cells}")
        sections.append(Section(
            f"Table '{table_name}' rows {start + 1}-{start + len(block)}:\n" + "\n".join(rows),
            {"rows": f"{start + 1}-{start + len(block)}", "table": table_name},
        ))
    return sections


def load_csv(path: Path) -> tuple[list[Section], dict[str, pd.DataFrame], list[str]]:
    df = _tidy_df(_read_csv(path))
    if df.empty:
        raise IngestionError(f"'{path.name}' has no data rows.")
    return table_to_sections(df, path.name), {path.name: df}, []


def load_excel(path: Path) -> tuple[list[Section], dict[str, pd.DataFrame], list[str]]:
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        raise IngestionError(f"'{path.name}' could not be read as Excel ({e}).") from e
    sections, tables, warnings = [], {}, []
    for sheet, df in sheets.items():
        df = _tidy_df(df)
        if df.empty:
            warnings.append(f"Sheet '{sheet}' is empty and was skipped.")
            continue
        name = f"{path.name}::{sheet}"
        tables[name] = df
        for s in table_to_sections(df, name):
            s.metadata["sheet"] = sheet
            sections.append(s)
    if not sections:
        raise IngestionError(f"'{path.name}' has no data in any sheet.")
    return sections, tables, warnings


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_document(path: str | Path, display_name: str | None = None) -> LoadedDocument:
    path = Path(path)
    validate_file(path)
    name = display_name or path.name
    ext = path.suffix.lower()
    tables: dict[str, pd.DataFrame] = {}

    if ext == ".pdf":
        sections, warnings = load_pdf(path)
    elif ext == ".txt":
        sections, warnings = load_txt(path)
    elif ext == ".csv":
        sections, tables, warnings = load_csv(path)
    else:  # .xlsx / .xls
        sections, tables, warnings = load_excel(path)

    # Tables were keyed by the on-disk name; re-key by the display name.
    if display_name and display_name != path.name:
        tables = {k.replace(path.name, display_name, 1): v for k, v in tables.items()}
        for s in sections:
            if "table" in s.metadata:
                s.metadata["table"] = s.metadata["table"].replace(path.name, display_name, 1)
                s.text = s.text.replace(f"'{path.name}", f"'{display_name}")

    for s in sections:
        s.metadata.update({"source": name, "doc_type": ext.lstrip(".")})

    logger.info("Loaded %s: %d sections, %d tables", name, len(sections), len(tables))
    return LoadedDocument(name, file_sha256(path), ext.lstrip("."), sections, tables, warnings)
