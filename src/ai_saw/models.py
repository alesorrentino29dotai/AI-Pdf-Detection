from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PageBoundary:
    page_number: int
    start_char: int
    end_char: int


@dataclass
class ExtractedDocument:
    source_path: str
    text: str
    word_count: int
    page_count: int
    page_boundaries: list[PageBoundary] = field(default_factory=list)


@dataclass
class TextChunk:
    section_name: str
    chunk_index: int
    text: str
    word_count: int
    start_char: int
    end_char: int


@dataclass
class Section:
    name: str
    text: str
    start_char: int
    end_char: int
    word_count: int
    chunks: list[TextChunk] = field(default_factory=list)


@dataclass
class SentenceSpan:
    text: str
    word_count: int
    start_char: int
    end_char: int
    section_name: str = ""


@dataclass
class SentenceScore:
    section_name: str
    sentence_index: int
    text: str
    word_count: int
    ai_score: float
    human_score: float
    start_char: int
    end_char: int


@dataclass
class ChunkScore:
    section_name: str
    chunk_index: int
    text: str
    word_count: int
    ai_score: float
    human_score: float
    start_char: int
    end_char: int


@dataclass
class SectionReport:
    name: str
    word_count: int
    ai_pct: float
    human_pct: float
    chunk_count: int
    chunks: list[ChunkScore] = field(default_factory=list)


@dataclass
class DetectionReport:
    source_path: str
    total_words: int
    ai_pct: float
    human_pct: float
    confidence: str
    sections: list[SectionReport] = field(default_factory=list)
    chunks: list[ChunkScore] = field(default_factory=list)
    sentences: list[SentenceScore] = field(default_factory=list)
    ai_sentence_threshold: float = 0.7
    profile: str = "general"
