from __future__ import annotations

from pathlib import Path

import typer

from ai_saw.calibrate import load_calibrator
from ai_saw.detect import (
    ACADEMIC_AI_SENTENCE_THRESHOLD,
    AIDetector,
    DEFAULT_AI_SENTENCE_THRESHOLD,
    DEFAULT_MODEL_ID,
    detect_sections,
)
from ai_saw.extract import extract_pdf
from ai_saw.report import write_reports
from ai_saw.segment import segment_document
from ai_saw.sentence import assign_sections, split_sentences

app = typer.Typer(help="Detect AI-generated text in PDF documents.")


def _parse_formats(formats: str) -> list[str]:
    return [item.strip().lower() for item in formats.split(",") if item.strip()]


def _print_summary(report) -> None:
    typer.echo(f"Source: {report.source_path}")
    typer.echo(f"Profile: {report.profile}")
    typer.echo(f"Words: {report.total_words:,}")
    typer.echo(
        f"Overall: {report.human_pct:.1f}% human / {report.ai_pct:.1f}% AI "
        f"(confidence: {report.confidence})"
    )
    typer.echo("")
    typer.echo("Sections:")
    for section in report.sections:
        typer.echo(
            f"  {section.name:<24} — {section.human_pct:.1f}% human / "
            f"{section.ai_pct:.1f}% AI ({section.word_count} words)"
        )

    top_ai = sorted(report.chunks, key=lambda item: item.ai_score, reverse=True)[:5]
    if top_ai:
        typer.echo("")
        typer.echo("Top AI-like chunks:")
        for chunk in top_ai:
            typer.echo(
                f"  [{chunk.section_name}, chunk {chunk.chunk_index + 1}] "
                f"{chunk.ai_score * 100:.1f}% AI"
            )

    flagged = [s for s in report.sentences if s.ai_score >= report.ai_sentence_threshold]
    if flagged:
        typer.echo("")
        typer.echo(
            f"AI-flagged sentences ({len(flagged)} of {len(report.sentences)}, "
            f"threshold {report.ai_sentence_threshold * 100:.0f}%):"
        )
        for sentence in sorted(flagged, key=lambda item: item.ai_score, reverse=True)[:10]:
            preview = sentence.text if len(sentence.text) <= 120 else sentence.text[:117] + "..."
            typer.echo(
                f"  [{sentence.section_name}] {sentence.ai_score * 100:.1f}% — {preview}"
            )
        if len(flagged) > 10:
            typer.echo(f"  ... and {len(flagged) - 10} more in ai_sentences.txt")


@app.command()
def main(
    pdf_path: Path = typer.Argument(..., help="Path to the PDF file to analyze."),
    output: Path = typer.Option(
        Path("reports"),
        "--output",
        "-o",
        help="Directory where reports will be written.",
    ),
    formats: str = typer.Option(
        "html,json,txt",
        "--format",
        "-f",
        help="Comma-separated report formats: html, json, txt.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Torch device to use, e.g. cpu or cuda.",
    ),
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model",
        help="HuggingFace model ID for AI detection.",
    ),
    chunk_words: int = typer.Option(400, help="Target words per chunk."),
    overlap_words: int = typer.Option(50, help="Word overlap between chunks."),
    sentence_threshold: float | None = typer.Option(
        None,
        "--sentence-threshold",
        min=0.0,
        max=1.0,
        help="Flag sentences with AI probability above this value (0-1).",
    ),
    profile: str = typer.Option(
        "academic",
        "--profile",
        help="Detection profile: academic (recommended for papers) or general.",
    ),
) -> None:
    """Analyze a PDF and estimate human vs AI-generated content."""
    if not pdf_path.exists():
        raise typer.BadParameter(f"PDF not found: {pdf_path}")

    normalized_profile = profile.strip().lower()
    if normalized_profile not in {"academic", "general"}:
        raise typer.BadParameter("Profile must be 'academic' or 'general'.")

    calibrator = load_calibrator() if normalized_profile == "academic" else None
    if normalized_profile == "academic" and calibrator is None:
        typer.echo(
            "Warning: academic calibrator not found; run "
            "'python scripts/academic_tools.py build-calibrator' first. "
            "Falling back to general scoring.",
            err=True,
        )
        normalized_profile = "general"

    effective_threshold = sentence_threshold
    if effective_threshold is None:
        effective_threshold = (
            ACADEMIC_AI_SENTENCE_THRESHOLD
            if normalized_profile == "academic"
            else DEFAULT_AI_SENTENCE_THRESHOLD
        )

    typer.echo(f"Extracting text from {pdf_path}...")
    document = extract_pdf(pdf_path)
    typer.echo(f"Extracted {document.word_count:,} words from {document.page_count} pages.")

    typer.echo("Segmenting document...")
    sections = segment_document(
        document.text,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )
    chunk_count = sum(len(section.chunks) for section in sections)
    typer.echo(f"Found {len(sections)} sections and {chunk_count} chunks.")

    typer.echo(f"Loading model {model_id}...")
    detector = AIDetector(model_id=model_id, device=device)

    typer.echo("Scoring chunks and sentences...")
    sentence_spans = assign_sections(split_sentences(document.text), sections)
    typer.echo(f"Found {len(sentence_spans)} scorable sentences.")

    report = detect_sections(
        sections,
        detector,
        document.source_path,
        sentence_spans=sentence_spans,
        ai_sentence_threshold=effective_threshold,
        calibrator=calibrator,
    )

    report_dir = output / pdf_path.stem
    written = write_reports(report, report_dir, _parse_formats(formats))

    typer.echo("")
    _print_summary(report)
    typer.echo("")
    typer.echo("Reports written:")
    for name, path in written.items():
        typer.echo(f"  {name}: {path}")


if __name__ == "__main__":
    app()
