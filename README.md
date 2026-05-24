# AI-Pdf-Detection

Local CLI tool to estimate how much text in a PDF document is likely human-written versus AI-generated. Analysis runs entirely on your machine using an open-source Hugging Face model.

## Features

- **PDF text extraction** via PyMuPDF
- **Section-aware analysis** for academic papers (Abstract, numbered sections, References)
- **Academic calibration profile** to reduce false positives on IEEE-style papers
- **Chunk-level scoring** with word-weighted document percentages
- **Sentence-level flagging** with configurable AI probability threshold
- **Fine-tuning pipeline** for domain-specific models with labeled ground truth
- **Multi-format reports**: HTML (interactive), JSON (machine-readable), TXT (flagged sentences list)

## Requirements

- Python 3.10+
- ~500 MB disk space for the detection model (downloaded once on first run)
- CPU or CUDA-capable GPU (GPU recommended for fine-tuning)

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

Analyze a PDF (academic profile is the default for papers):

```bash
ai-saw path/to/document.pdf
ai-saw paper.pdf --profile academic --device cpu
ai-saw paper.pdf --profile general
```

Common options:

```bash
ai-saw document.pdf --output reports --format html,json,txt
ai-saw document.pdf --sentence-threshold 0.85
ai-saw document.pdf --model ./models/academic
```

| Option | Default | Description |
|--------|---------|-------------|
| `--profile` | `academic` | `academic` for papers, `general` for Q&A/blog-style text |
| `--output`, `-o` | `reports` | Output directory for generated reports |
| `--format`, `-f` | `html,json,txt` | Report formats: `html`, `json`, `txt` |
| `--device` | auto | Torch device (`cpu` or `cuda`) |
| `--model` | `yuchuantian/AIGC_detector_env3` | Hugging Face model ID or local fine-tuned path |
| `--sentence-threshold` | `0.85` academic / `0.7` general | Flag sentences above this AI probability |
| `--chunk-words` | `400` | Target words per analysis chunk |
| `--overlap-words` | `50` | Overlap between consecutive chunks |

## Academic profile (recommended for IEEE papers)

The base model is trained on Q&A text (HC3), so formal academic prose often triggers false positives. The **academic profile** applies calibration against a reference corpus of pre-2022 arXiv abstracts in CS domains (`cs.AR`, `cs.CR`, `cs.DC`, `cs.PL`).

Example improvement on a human-written IEEE conference paper:

| Profile | Overall AI | Flagged sentences |
|---------|------------|-------------------|
| `general` | ~90% | ~119 / 196 |
| `academic` | ~28% | ~5 / 196 |

Rebuild the reference calibrator:

```bash
python scripts/academic_tools.py build-calibrator --device cpu
```

## Fine-tuning on labeled academic data

For the best accuracy on your domain, fine-tune with verified ground truth:

### 1. Build a human reference dataset

```bash
python scripts/academic_tools.py build-dataset --max-human 500
```

This creates:

- `data/academic/human_arxiv.csv` — pre-2022 arXiv abstracts labeled `0` (human)
- `data/academic/ai_labeled.csv` — template for AI-labeled examples

### 2. Add AI ground truth

Add verified AI-generated IEEE-style sentences or paragraphs to `data/academic/ai_labeled.csv` with label `1`. Good sources:

- Sentences you know were written by ChatGPT/GPT-4 for your topic
- Public academic AI-detection datasets (e.g. [M4GT-Bench](https://github.com/mbzuai-nlp/M4GT-Bench) arXiv domain)

Merge human and AI rows into one training file:

```bash
cat data/academic/human_arxiv.csv > data/academic/train.csv
tail -n +2 data/academic/ai_labeled.csv >> data/academic/train.csv
```

### 3. Fine-tune

```bash
pip install -e ".[train]"
python scripts/academic_tools.py finetune \
  --train-csv data/academic/train.csv \
  --output-dir models/academic \
  --device cuda \
  --epochs 2
```

### 4. Use the fine-tuned model

```bash
ai-saw paper.pdf --model models/academic --profile academic
```

## Output

Reports are written to `reports/<pdf-name>/`:

| File | Description |
|------|-------------|
| `report.html` | Summary, section breakdown, AI-flagged sentences, chunk map |
| `report.json` | Full structured results including all sentence scores |
| `ai_sentences.txt` | Plain-text list of sentences above the AI threshold |

## How it works

1. Extract plain text from the PDF
2. Split into sections using heading heuristics
3. Split each section into overlapping word chunks (~400 words)
4. Split the full document into sentences for fine-grained scoring
5. Score chunks and sentences with a transformer classifier
6. Apply academic calibration (if `--profile academic`)
7. Aggregate percentages weighted by word count
8. Flag sentences above the configured AI probability threshold

The underlying model ([AIGC_detector_env3](https://huggingface.co/yuchuantian/AIGC_detector_env3)) is trained primarily on Q&A-style human vs. ChatGPT text. Use the academic profile or a fine-tuned model for papers.

## Limitations

- **Not forensic proof.** Scores are probabilistic estimates, not evidence of authorship.
- **Calibration is not perfect.** Academic profile reduces false positives but cannot guarantee correctness.
- **Fine-tuning needs ground truth.** Without labeled AI examples from your domain, fine-tuning will not help.
- **English-focused.** Best results on English prose; other languages may produce unreliable scores.
- **PDF quality.** Scanned PDFs without a text layer are not supported (OCR is out of scope).

Use this tool for exploratory analysis only. Do not rely on it as the sole basis for academic integrity or publishing decisions.

## Project structure

```
src/ai_saw/
├── cli.py         # Command-line interface
├── extract.py     # PDF text extraction
├── segment.py     # Section and chunk splitting
├── sentence.py    # Sentence splitting and section assignment
├── detect.py      # Model inference and aggregation
├── calibrate.py   # Academic score calibration
├── report.py      # HTML, JSON, and TXT report generation
├── models.py      # Data classes
└── data/
    └── academic_reference.json
scripts/
└── academic_tools.py  # Calibrator build, dataset prep, fine-tuning
```

## References

- [AIGC Text Detector (GitHub)](https://github.com/YuchuanTian/AIGC_text_detector)
- [M4GT-Bench (arXiv domain benchmark)](https://github.com/mbzuai-nlp/M4GT-Bench)
- [Multiscale Positive-Unlabeled Detection of AI-Generated Texts (ICLR 2024)](https://arxiv.org/abs/2305.18149)
- [HC3 dataset](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)

## License

This project is provided as-is for research and personal use. The detection model weights are subject to their own license on Hugging Face.
