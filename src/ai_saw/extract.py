from __future__ import annotations

import re
from pathlib import Path

import fitz

from ai_saw.models import ExtractedDocument, PageBoundary


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def extract_pdf(pdf_path: str | Path) -> ExtractedDocument:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)
    parts: list[str] = []
    boundaries: list[PageBoundary] = []
    offset = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_text = page.get_text("text")
        if page_index > 0 and parts:
            parts.append("\n")
            offset += 1

        start = offset
        parts.append(page_text)
        offset += len(page_text)
        boundaries.append(
            PageBoundary(
                page_number=page_index + 1,
                start_char=start,
                end_char=offset,
            )
        )

    doc.close()
    text = _normalize_text("".join(parts))
    return ExtractedDocument(
        source_path=str(path.resolve()),
        text=text,
        word_count=_count_words(text),
        page_count=len(boundaries),
        page_boundaries=boundaries,
    )
