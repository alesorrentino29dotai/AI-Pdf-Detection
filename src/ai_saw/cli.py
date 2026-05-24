from __future__ import annotations

from pathlib import Path

import typer

from ai_saw.detect import AIDetector, DEFAULT_MODEL_ID, detect_sections
from ai_saw.extract import extract_pdf
from ai_saw.report import write_reports
from ai_saw.segment import segment_document

app = typer.Typer(help="Detect AI-generated text in PDF documents.")


def _parse_formats(formats: str) -> list[str]:
    return [item.strip().lower() for item in formats.split(",") if item.strip()]


def _print_summary(report) -> None:
    typer.echo(f"Source: {report.source_path}")
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
        "html,json",
        "--format",
        "-f",
        help="Comma-separated report formats: html, json.",
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
) -> None:
    """Analyze a PDF and estimate human vs AI-generated content."""
    if not pdf_path.exists():
        raise typer.BadParameter(f"PDF not found: {pdf_path}")

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

    typer.echo("Scoring chunks...")
    report = detect_sections(sections, detector, document.source_path)

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
