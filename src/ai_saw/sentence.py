from __future__ import annotations

import re

from ai_saw.models import Section, SentenceSpan

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"(\[]|\u2022)')
MIN_SENTENCE_WORDS = 8
SKIP_LINE = re.compile(
    r"^(?:"
    r"[IVXLC]+\.\s+[A-Z]|"
    r"\d+\.\s+[A-Z]|"
    r"REFERENCES|"
    r"Abstract(?:—|-|–|:)?|"
    r"Index Terms|"
    r"Fig\.|"
    r"TABLE|"
    r"Algorithm"
    r")",
    re.IGNORECASE,
)


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _normalize_sentence(text: str) -> str:
    return " ".join(text.split())


def split_sentences(text: str) -> list[SentenceSpan]:
    sentences: list[SentenceSpan] = []
    cursor = 0
    for match in SENTENCE_SPLIT.finditer(text):
        end = match.start()
        raw = text[cursor:end]
        sentence = _normalize_sentence(raw)
        if sentence and _count_words(sentence) >= MIN_SENTENCE_WORDS and not SKIP_LINE.match(sentence):
            sentences.append(
                SentenceSpan(
                    text=sentence,
                    word_count=_count_words(sentence),
                    start_char=cursor,
                    end_char=end,
                )
            )
        cursor = match.end()

    tail = _normalize_sentence(text[cursor:])
    if tail and _count_words(tail) >= MIN_SENTENCE_WORDS and not SKIP_LINE.match(tail):
        sentences.append(
            SentenceSpan(
                text=tail,
                word_count=_count_words(tail),
                start_char=cursor,
                end_char=len(text),
            )
        )

    return sentences


def assign_sections(sentences: list[SentenceSpan], sections: list[Section]) -> list[SentenceSpan]:
    if not sections:
        for sentence in sentences:
            sentence.section_name = "Document"
        return sentences

    for sentence in sentences:
        for section in sections:
            if section.start_char <= sentence.start_char < section.end_char:
                sentence.section_name = section.name
                break
        else:
            sentence.section_name = sections[-1].name
    return sentences
