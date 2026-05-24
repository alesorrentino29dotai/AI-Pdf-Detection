from __future__ import annotations

import re
from dataclasses import dataclass

from ai_saw.models import Section, TextChunk

SECTION_PATTERN = re.compile(
    r"(?m)^(?:"
    r"[IVXLC]+\.\s+[A-Z][A-Z\s\-,]+|"
    r"\d+\.\s+[A-Z][A-Z\s\-,]+|"
    r"REFERENCES|"
    r"ACKNOWLEDGMENT(?:S)?|"
    r"CONCLUSION|"
    r"Index Terms"
    r")\s*$"
)

INLINE_ABSTRACT_PATTERN = re.compile(r"\bAbstract(?:—|-|–|:)\s*", re.IGNORECASE)

DEFAULT_CHUNK_WORDS = 400
DEFAULT_OVERLAP_WORDS = 50
MIN_CHUNK_WORDS = 50


@dataclass
class SectionMatch:
    name: str
    start: int


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _normalize_section_name(raw: str) -> str:
    name = raw.strip()
    name = re.sub(r"\s+", " ", name)
    return name.title() if name.isupper() else name


def split_sections(text: str) -> list[Section]:
    matches: list[SectionMatch] = []

    abstract_match = INLINE_ABSTRACT_PATTERN.search(text)
    if abstract_match:
        matches.append(SectionMatch(name="Abstract", start=abstract_match.start()))

    for match in SECTION_PATTERN.finditer(text):
        heading = match.group(0).strip()
        if heading.lower().startswith("index terms"):
            continue
        matches.append(SectionMatch(name=_normalize_section_name(heading), start=match.start()))

    matches.sort(key=lambda item: item.start)
    deduped: list[SectionMatch] = []
    for match in matches:
        if deduped and match.start == deduped[-1].start:
            continue
        deduped.append(match)
    matches = deduped

    if not matches:
        return [
            Section(
                name="Document",
                text=text,
                start_char=0,
                end_char=len(text),
                word_count=_count_words(text),
            )
        ]

    sections: list[Section] = []
    for index, match in enumerate(matches):
        start = match.start
        end = matches[index + 1].start if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        sections.append(
            Section(
                name=match.name,
                text=section_text,
                start_char=start,
                end_char=end,
                word_count=_count_words(section_text),
            )
        )

    return sections


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\b\w+\b", text)]


def chunk_section(
    section: Section,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> list[TextChunk]:
    spans = _word_spans(section.text)
    if not spans:
        return []

    if len(spans) <= chunk_words:
        if len(spans) < min_chunk_words:
            return []
        return [
            TextChunk(
                section_name=section.name,
                chunk_index=0,
                text=section.text.strip(),
                word_count=len(spans),
                start_char=section.start_char,
                end_char=section.end_char,
            )
        ]

    chunks: list[TextChunk] = []
    step = max(chunk_words - overlap_words, 1)
    chunk_index = 0
    word_start = 0

    while word_start < len(spans):
        word_end = min(word_start + chunk_words, len(spans))
        if len(spans) - word_end < min_chunk_words and word_end < len(spans):
            word_end = len(spans)

        char_start = spans[word_start][0]
        char_end = spans[word_end - 1][1]
        chunk_text = section.text[char_start:char_end].strip()
        word_count = word_end - word_start

        if word_count >= min_chunk_words:
            chunks.append(
                TextChunk(
                    section_name=section.name,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    word_count=word_count,
                    start_char=section.start_char + char_start,
                    end_char=section.start_char + char_end,
                )
            )
            chunk_index += 1

        if word_end >= len(spans):
            break
        word_start += step

    return chunks


def segment_document(
    text: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Section]:
    sections = split_sections(text)
    for section in sections:
        section.chunks = chunk_section(
            section,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )
    return sections
