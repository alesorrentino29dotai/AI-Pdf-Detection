# AI-Pdf-Detection

Local CLI tool to estimate how much text in a PDF document is likely human-written versus AI-generated. Analysis runs entirely on your machine using an open-source Hugging Face model.

## Features

- **PDF text extraction** via PyMuPDF
- **Section-aware analysis** for academic papers (Abstract, numbered sections, References)
- **Chunk-level scoring** with word-weighted document percentages
- **Sentence-level flagging** with configurable AI probability threshold
- **Multi-format reports**: HTML (interactive), JSON (machine-readable), TXT (flagged sentences list)

## Requirements

- Python 3.10+
- ~500 MB disk space for the detection model (downloaded once on first run)
- CPU or CUDA-capable GPU

## Installation

```bash
git clone https://github.com/alesorrentino29dotai/AI-Pdf-Detection.git
cd AI-Pdf-Detection

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On first analysis, the model `yuchuantian/AIGC_detector_env3` is downloaded from Hugging Face and cached locally.

## Usage

Analyze a PDF:

```bash
ai-saw path/to/document.pdf
```

Common options:

```bash
ai-saw document.pdf --device cpu
ai-saw document.pdf --output reports --format html,json,txt
ai-saw document.pdf --sentence-threshold 0.85
ai-saw document.pdf --chunk-words 400 --overlap-words 50
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `reports` | Output directory for generated reports |
| `--format`, `-f` | `html,json,txt` | Report formats: `html`, `json`, `txt` |
| `--device` | auto | Torch device (`cpu` or `cuda`) |
| `--model` | `yuchuantian/AIGC_detector_env3` | Hugging Face model ID |
| `--sentence-threshold` | `0.7` | Flag sentences with AI probability above this value (0–1) |
| `--chunk-words` | `400` | Target words per analysis chunk |
| `--overlap-words` | `50` | Overlap between consecutive chunks |

## Output

Reports are written to `reports/<pdf-name>/`:

| File | Description |
|------|-------------|
| `report.html` | Summary, section breakdown, AI-flagged sentences, chunk map |
| `report.json` | Full structured results including all sentence scores |
| `ai_sentences.txt` | Plain-text list of sentences above the AI threshold |

Example terminal summary:

```
Overall: 68.0% human / 32.0% AI (confidence: medium)

Sections:
  Abstract                 — 55.0% human / 45.0% AI (253 words)
  I. Introduction          — 72.0% human / 28.0% AI (343 words)

AI-flagged sentences (12 of 196, threshold 70%):
  [Introduction] 82.3% — Example sentence text...
```

## How it works

1. Extract plain text from the PDF
2. Split into sections using heading heuristics
3. Split each section into overlapping word chunks (~400 words)
4. Split the full document into sentences for fine-grained scoring
5. Score chunks and sentences with a transformer classifier
6. Aggregate percentages weighted by word count
7. Flag sentences above the configured AI probability threshold

The underlying model ([AIGC_detector_env3](https://huggingface.co/yuchuantian/AIGC_detector_env3)) is trained primarily on Q&A-style human vs. ChatGPT text. It is **not** calibrated for high-stakes authorship decisions.

## Limitations

- **Not forensic proof.** Scores are probabilistic estimates, not evidence of authorship.
- **Academic false positives.** Formal scientific and technical writing often scores as AI-like even when human-written.
- **English-focused.** Best results on English prose; other languages may produce unreliable scores.
- **PDF quality.** Scanned PDFs without a text layer are not supported (OCR is out of scope).
- **Edited AI text.** Heavily rewritten AI output is harder to detect.

Use this tool for exploratory analysis only. Do not rely on it as the sole basis for academic integrity or publishing decisions.

## Project structure

```
src/ai_saw/
├── cli.py       # Command-line interface
├── extract.py   # PDF text extraction
├── segment.py   # Section and chunk splitting
├── sentence.py  # Sentence splitting and section assignment
├── detect.py    # Model inference and aggregation
├── report.py    # HTML, JSON, and TXT report generation
└── models.py    # Data classes
```

## References

- [AIGC Text Detector (GitHub)](https://github.com/YuchuanTian/AIGC_text_detector)
- [Multiscale Positive-Unlabeled Detection of AI-Generated Texts (ICLR 2024)](https://arxiv.org/abs/2305.18149)
- [HC3 dataset](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)

## License

This project is provided as-is for research and personal use. The detection model weights are subject to their own license on Hugging Face.
