from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ai_saw.calibrate import (
    AcademicCalibrator,
    build_calibrator_from_scores,
    fetch_arxiv_abstracts,
    save_calibrator,
)
from ai_saw.detect import AIDetector, DEFAULT_MODEL_ID


def build_reference_scores(
    model_id: str,
    device: str | None,
    max_results: int,
) -> list[float]:
    detector = AIDetector(model_id=model_id, device=device)
    abstracts = fetch_arxiv_abstracts(max_results=max_results)
    if not abstracts:
        raise RuntimeError("No arXiv abstracts fetched for academic calibration.")
    return detector.score_texts(abstracts)


def build_dataset(output_dir: Path, max_human: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    human_path = output_dir / "human_arxiv.csv"
    template_path = output_dir / "ai_labeled.csv"

    abstracts = fetch_arxiv_abstracts(max_results=max_human)
    with human_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        for text in abstracts:
            writer.writerow([text, 0])

    if not template_path.exists():
        template_path.write_text(
            "text,label\n"
            '"Paste AI-generated IEEE-style sentences here",1\n',
            encoding="utf-8",
        )

    return human_path, template_path


def finetune(
    train_csv: Path,
    output_dir: Path,
    model_id: str,
    device: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    rows: list[tuple[str, int]] = []
    with train_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row["text"].strip()
            label = int(row["label"])
            if text and label in (0, 1):
                rows.append((text, label))

    if not rows:
        raise RuntimeError(f"No training rows found in {train_csv}")

    labels = [label for _, label in rows]
    if len(set(labels)) < 2:
        raise RuntimeError(
            "Fine-tuning needs both human (0) and AI (1) labeled rows. "
            "Add AI examples to data/academic/ai_labeled.csv."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
        ignore_mismatched_sizes=True,
    )

    class TextDataset(Dataset):
        def __init__(self, samples: list[tuple[str, int]]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> dict:
            text, label = self.samples[index]
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            encoded["labels"] = label
            return encoded

    split_index = max(int(len(rows) * 0.9), 1)
    train_rows = rows[:split_index]
    eval_rows = rows[split_index:] or rows[:1]

    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=TextDataset(train_rows),
        eval_dataset=TextDataset(eval_rows),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "base_model": model_id,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "human_rows": sum(1 for _, label in rows if label == 0),
        "ai_rows": sum(1 for _, label in rows if label == 1),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Academic calibration and fine-tuning utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser(
        "build-calibrator",
        help="Build academic calibration reference from pre-2022 arXiv abstracts.",
    )
    calibrate_parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    calibrate_parser.add_argument("--device", default=None)
    calibrate_parser.add_argument("--max-results", type=int, default=200)
    calibrate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/ai_saw/data/academic_reference.json"),
    )

    dataset_parser = subparsers.add_parser(
        "build-dataset",
        help="Create human arXiv CSV and an AI label template for fine-tuning.",
    )
    dataset_parser.add_argument("--output-dir", type=Path, default=Path("data/academic"))
    dataset_parser.add_argument("--max-human", type=int, default=500)

    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune the detector on a labeled academic CSV.",
    )
    finetune_parser.add_argument("--train-csv", type=Path, required=True)
    finetune_parser.add_argument("--output-dir", type=Path, default=Path("models/academic"))
    finetune_parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    finetune_parser.add_argument("--device", default=None)
    finetune_parser.add_argument("--epochs", type=int, default=2)
    finetune_parser.add_argument("--batch-size", type=int, default=8)
    finetune_parser.add_argument("--learning-rate", type=float, default=2e-5)

    args = parser.parse_args()

    if args.command == "build-calibrator":
        scores = build_reference_scores(args.model, args.device, args.max_results)
        calibrator = build_calibrator_from_scores(scores)
        output = save_calibrator(calibrator, args.output)
        print(f"Saved academic calibrator with {len(scores)} reference scores to {output}")
        return

    if args.command == "build-dataset":
        human_path, ai_path = build_dataset(args.output_dir, args.max_human)
        print(f"Human dataset: {human_path}")
        print(f"AI label template: {ai_path}")
        print("Add verified AI-generated academic sentences to ai_labeled.csv, then merge for training.")
        return

    if args.command == "finetune":
        finetune(
            args.train_csv,
            args.output_dir,
            args.model,
            args.device,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
        print(f"Fine-tuned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
