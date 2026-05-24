from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

from ai_saw.models import DetectionReport


def _ai_sentences(report: DetectionReport) -> list:
    threshold = report.ai_sentence_threshold
    return sorted(
        [sentence for sentence in report.sentences if sentence.ai_score >= threshold],
        key=lambda item: item.ai_score,
        reverse=True,
    )


DISCLAIMER = (
    "Scores are probabilistic estimates, not proof of authorship. "
    "Formal academic writing can resemble AI-generated text and may produce false positives. "
    "Sentences flagged below exceeded the AI probability threshold but may still be human-written."
)


def _chunk_color(ai_score: float) -> str:
    red = int(220 * ai_score + 120 * (1 - ai_score))
    green = int(120 * ai_score + 200 * (1 - ai_score))
    return f"rgb({red}, {green}, 120)"


def _preview(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def report_to_dict(report: DetectionReport) -> dict:
    return asdict(report)


def write_json_report(report: DetectionReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")


def write_ai_sentences_report(report: DetectionReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flagged = _ai_sentences(report)
    lines = [
        f"Source: {report.source_path}",
        f"Overall: {report.human_pct:.1f}% human / {report.ai_pct:.1f}% AI",
        f"Threshold: {report.ai_sentence_threshold * 100:.0f}% AI probability",
        f"Flagged sentences: {len(flagged)} of {len(report.sentences)}",
        "",
    ]
    for index, sentence in enumerate(flagged, start=1):
        lines.append(
            f"{index}. [{sentence.section_name}] {sentence.ai_score * 100:.1f}% AI — {sentence.text}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_report(report: DetectionReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    section_rows = []
    for section in report.sections:
        section_rows.append(
            "<tr>"
            f"<td>{html.escape(section.name)}</td>"
            f"<td>{section.word_count}</td>"
            f"<td>{section.human_pct:.1f}%</td>"
            f"<td>{section.ai_pct:.1f}%</td>"
            f"<td>{section.chunk_count}</td>"
            "</tr>"
        )

    chunk_cards = []
    for chunk in sorted(report.chunks, key=lambda item: item.ai_score, reverse=True):
        color = _chunk_color(chunk.ai_score)
        chunk_cards.append(
            "<article class='chunk' style='border-left-color: "
            f"{color}'>"
            "<header>"
            f"<strong>{html.escape(chunk.section_name)}</strong> "
            f"<span>chunk {chunk.chunk_index + 1}</span>"
            "</header>"
            f"<div class='scores'>Human {chunk.human_score * 100:.1f}% · "
            f"AI {chunk.ai_score * 100:.1f}% · {chunk.word_count} words</div>"
            f"<p>{html.escape(_preview(chunk.text))}</p>"
            "</article>"
        )

    top_ai = sorted(report.chunks, key=lambda item: item.ai_score, reverse=True)[:5]
    top_ai_lines = []
    for chunk in top_ai:
        top_ai_lines.append(
            "<li>"
            f"<strong>{html.escape(chunk.section_name)}</strong>, chunk {chunk.chunk_index + 1}: "
            f"{chunk.ai_score * 100:.1f}% AI"
            "</li>"
        )

    flagged_sentences = _ai_sentences(report)
    sentence_rows = []
    for sentence in flagged_sentences:
        color = _chunk_color(sentence.ai_score)
        sentence_rows.append(
            "<article class='sentence' style='border-left-color: "
            f"{color}'>"
            f"<div class='scores'><strong>{html.escape(sentence.section_name)}</strong> · "
            f"{sentence.ai_score * 100:.1f}% AI · {sentence.word_count} words</div>"
            f"<p>{html.escape(sentence.text)}</p>"
            "</article>"
        )

    human_width = max(min(report.human_pct, 100.0), 0.0)
    ai_width = max(min(report.ai_pct, 100.0), 0.0)

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AI Detection Report</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      margin: 2rem auto;
      max-width: 960px;
      line-height: 1.5;
      color: #1f2933;
      background: #f8fafc;
    }}
    h1, h2 {{ margin-bottom: 0.4rem; }}
    .meta {{ color: #52606d; margin-bottom: 1.5rem; }}
    .summary-bar {{
      display: flex;
      height: 28px;
      border-radius: 999px;
      overflow: hidden;
      margin: 1rem 0 0.5rem;
      border: 1px solid #cbd2d9;
    }}
    .human-bar {{ background: #2f855a; color: white; display: flex; align-items: center; justify-content: center; }}
    .ai-bar {{ background: #c53030; color: white; display: flex; align-items: center; justify-content: center; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      margin: 1rem 0 2rem;
    }}
    th, td {{
      border-bottom: 1px solid #e4e7eb;
      padding: 0.65rem 0.75rem;
      text-align: left;
    }}
    th {{ background: #eef2f6; }}
    .chunk {{
      background: white;
      border: 1px solid #d9e2ec;
      border-left-width: 6px;
      border-radius: 8px;
      padding: 0.9rem 1rem;
      margin-bottom: 0.85rem;
    }}
    .chunk header {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.35rem;
    }}
    .scores {{ color: #52606d; font-size: 0.95rem; margin-bottom: 0.35rem; }}
    .sentence {{
      background: white;
      border: 1px solid #d9e2ec;
      border-left-width: 6px;
      border-radius: 8px;
      padding: 0.9rem 1rem;
      margin-bottom: 0.85rem;
    }}
    .disclaimer {{
      background: #fffbea;
      border: 1px solid #f6e05e;
      padding: 0.85rem 1rem;
      border-radius: 8px;
      margin-top: 2rem;
    }}
    ul {{ padding-left: 1.2rem; }}
  </style>
</head>
<body>
  <h1>AI Detection Report</h1>
  <p class="meta">Source: {html.escape(report.source_path)}</p>
  <p class="meta">Profile: {html.escape(report.profile)} · Total words: {report.total_words:,} · Confidence: {html.escape(report.confidence)}</p>

  <h2>Overall</h2>
  <div class="summary-bar">
    <div class="human-bar" style="width: {human_width:.1f}%">{report.human_pct:.1f}% human</div>
    <div class="ai-bar" style="width: {ai_width:.1f}%">{report.ai_pct:.1f}% AI</div>
  </div>

  <h2>Sections</h2>
  <table>
    <thead>
      <tr>
        <th>Section</th>
        <th>Words</th>
        <th>Human</th>
        <th>AI</th>
        <th>Chunks</th>
      </tr>
    </thead>
    <tbody>
      {"".join(section_rows)}
    </tbody>
  </table>

  <h2>AI-flagged sentences ({len(flagged_sentences)} of {len(report.sentences)}, threshold {report.ai_sentence_threshold * 100:.0f}%)</h2>
  {"".join(sentence_rows) if sentence_rows else "<p>No sentences exceeded the threshold.</p>"}

  <h2>Top AI-like chunks</h2>
  <ul>
    {"".join(top_ai_lines)}
  </ul>

  <h2>Chunk map</h2>
  {"".join(chunk_cards)}

  <p class="disclaimer">{html.escape(DISCLAIMER)}</p>
</body>
</html>
"""

    output_path.write_text(document, encoding="utf-8")


def write_reports(
    report: DetectionReport,
    output_dir: Path,
    formats: list[str],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    normalized_formats = {item.strip().lower() for item in formats}

    if "json" in normalized_formats:
        json_path = output_dir / "report.json"
        write_json_report(report, json_path)
        written["json"] = json_path

    if "txt" in normalized_formats or "sentences" in normalized_formats:
        txt_path = output_dir / "ai_sentences.txt"
        write_ai_sentences_report(report, txt_path)
        written["txt"] = txt_path

    if "html" in normalized_formats:
        html_path = output_dir / "report.html"
        write_html_report(report, html_path)
        written["html"] = html_path

    return written
