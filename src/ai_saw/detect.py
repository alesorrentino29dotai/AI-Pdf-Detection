from __future__ import annotations

from collections import defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ai_saw.models import (
    ChunkScore,
    DetectionReport,
    Section,
    SectionReport,
    SentenceScore,
    SentenceSpan,
    TextChunk,
)

DEFAULT_MODEL_ID = "yuchuantian/AIGC_detector_env3"
BATCH_SIZE = 8
DEFAULT_AI_SENTENCE_THRESHOLD = 0.7


def _resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ai_label_index(id2label: dict[int, str]) -> int:
    for index, label in id2label.items():
        normalized = label.lower()
        if any(token in normalized for token in ("ai", "fake", "machine", "generated", "chatgpt", "gpt")):
            return int(index)
    return max(id2label.keys())


def _confidence_label(ai_pct: float) -> str:
    distance = abs(ai_pct - 50.0)
    if distance >= 30:
        return "high"
    if distance >= 15:
        return "medium"
    return "low"


class AIDetector:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.ai_label_index = _ai_label_index(self.model.config.id2label)

    def score_texts(self, texts: list[str]) -> list[float]:
        if not texts:
            return []

        scores: list[float] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)
                batch_scores = probabilities[:, self.ai_label_index].tolist()
            scores.extend(float(score) for score in batch_scores)
        return scores


def _aggregate_percentages(chunks: list[ChunkScore]) -> tuple[float, float]:
    total_words = sum(chunk.word_count for chunk in chunks)
    if total_words == 0:
        return 0.0, 0.0
    ai_pct = sum(chunk.word_count * chunk.ai_score for chunk in chunks) / total_words * 100
    human_pct = 100.0 - ai_pct
    return round(ai_pct, 2), round(human_pct, 2)


def detect_sentences(
    sentences: list[SentenceSpan],
    detector: AIDetector,
) -> list[SentenceScore]:
    if not sentences:
        return []

    texts = [item.text for item in sentences]
    ai_scores = detector.score_texts(texts)
    scored: list[SentenceScore] = []
    for index, (sentence, ai_score) in enumerate(zip(sentences, ai_scores, strict=True)):
        scored.append(
            SentenceScore(
                section_name=sentence.section_name,
                sentence_index=index,
                text=sentence.text,
                word_count=sentence.word_count,
                ai_score=ai_score,
                human_score=1.0 - ai_score,
                start_char=sentence.start_char,
                end_char=sentence.end_char,
            )
        )
    return scored


def detect_sections(
    sections: list[Section],
    detector: AIDetector,
    source_path: str,
    sentence_spans: list[SentenceSpan] | None = None,
    ai_sentence_threshold: float = DEFAULT_AI_SENTENCE_THRESHOLD,
) -> DetectionReport:
    all_chunks: list[TextChunk] = []
    for section in sections:
        all_chunks.extend(section.chunks)

    ai_scores = detector.score_texts([chunk.text for chunk in all_chunks])
    scored_chunks: list[ChunkScore] = []
    for chunk, ai_score in zip(all_chunks, ai_scores, strict=True):
        human_score = 1.0 - ai_score
        scored_chunks.append(
            ChunkScore(
                section_name=chunk.section_name,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                word_count=chunk.word_count,
                ai_score=ai_score,
                human_score=human_score,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
        )

    by_section: dict[str, list[ChunkScore]] = defaultdict(list)
    for chunk in scored_chunks:
        by_section[chunk.section_name].append(chunk)

    section_reports: list[SectionReport] = []
    for section in sections:
        section_chunks = by_section.get(section.name, [])
        if not section_chunks:
            continue
        ai_pct, human_pct = _aggregate_percentages(section_chunks)
        section_reports.append(
            SectionReport(
                name=section.name,
                word_count=section.word_count,
                ai_pct=ai_pct,
                human_pct=human_pct,
                chunk_count=len(section_chunks),
                chunks=section_chunks,
            )
        )

    overall_ai_pct, overall_human_pct = _aggregate_percentages(scored_chunks)

    scored_sentences: list[SentenceScore] = []
    if sentence_spans:
        scored_sentences = detect_sentences(sentence_spans, detector)

    return DetectionReport(
        source_path=source_path,
        total_words=sum(section.word_count for section in sections),
        ai_pct=overall_ai_pct,
        human_pct=overall_human_pct,
        confidence=_confidence_label(overall_ai_pct),
        sections=section_reports,
        chunks=scored_chunks,
        sentences=scored_sentences,
        ai_sentence_threshold=ai_sentence_threshold,
    )
