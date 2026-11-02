"""Fine-tune, evaluate, and optionally save BioClinicalBERT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .data import SPECIALTIES, TARGET_COLUMN, TEXT_COLUMN, load_clinical_notes, split_frame
from .evaluate import classification_report_frame, error_analysis
from .utils import set_seed

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 256
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def label_maps() -> tuple[dict[str, int], dict[int, str]]:
    """Build stable six-class label mappings."""
    label2id = {label: index for index, label in enumerate(sorted(SPECIALTIES))}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def tokenize_batch(batch, tokenizer, max_length: int = MAX_LENGTH):
    """Tokenize with the experiment's fixed-length truncation policy."""
    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def create_huggingface_dataset(frame: pd.DataFrame, tokenizer, max_length: int = 256):
    """Create a tokenized Hugging Face Dataset from a labeled DataFrame."""
    from datasets import Dataset

    label2id, _ = label_maps()
    prepared = frame[[TEXT_COLUMN, TARGET_COLUMN]].copy()
    prepared["label"] = prepared[TARGET_COLUMN].map(label2id)
    if prepared["label"].isna().any():
        raise ValueError("Found a label outside the configured six specialties.")
    dataset = Dataset.from_pandas(
        prepared[[TEXT_COLUMN, "label"]], preserve_index=False
    )
    return dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, max_length),
        batched=True,
    )


def compute_metrics(eval_prediction) -> dict[str, float]:
    """Compute accuracy and macro-F1 for Hugging Face Trainer."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def token_lengths(texts, tokenizer) -> list[int]:
    """Count untruncated tokenizer input IDs for each note."""
    return [len(tokenizer(text, truncation=False)["input_ids"]) for text in texts]


def build_training_arguments(
    output_dir: str | Path,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    random_state: int = 42,
    device: str = "auto",
):
    """Build the original two-epoch configuration with portable device selection."""
    from transformers import TrainingArguments

    if device == "cpu":
        use_cpu = True
    else:
        use_cpu = False
        if device in {"mps", "cuda"}:
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            if device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available.")

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=random_state,
        data_seed=random_state,
        use_cpu=use_cpu,
    )


def train_bioclinicalbert(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: str | Path = "models/bioclinicalbert",
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    random_state: int = 42,
    device: str = "auto",
):
    """Fine-tune BioClinicalBERT and return trainer, test data, and tokenizer."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
    )

    set_seed(random_state)
    label2id, id2label = label_maps()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_train = create_huggingface_dataset(
        train_frame, tokenizer, max_length=max_length
    )
    tokenized_test = create_huggingface_dataset(
        test_frame, tokenizer, max_length=max_length
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    training_args = build_training_arguments(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        random_state=random_state,
        device=device,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    print(f"Training device: {training_args.device}")
    trainer.train()
    return trainer, tokenized_test, tokenizer


def evaluate_trainer(
    trainer,
    tokenized_test,
    test_frame: pd.DataFrame,
    output_dir: str | Path = "results/bert",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict the test set and save metrics, report, predictions, and errors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_output = trainer.predict(tokenized_test)
    predicted_ids = np.argmax(prediction_output.predictions, axis=1)
    true_ids = prediction_output.label_ids
    _, id2label = label_maps()
    predicted_labels = [id2label[int(index)] for index in predicted_ids]
    true_labels = [id2label[int(index)] for index in true_ids]

    metrics = pd.DataFrame(
        [
            {
                "Model": "BioClinicalBERT",
                "Input": "Full clinical text",
                "Accuracy": accuracy_score(true_labels, predicted_labels),
                "Macro_F1": f1_score(true_labels, predicted_labels, average="macro"),
            }
        ]
    )
    report = classification_report_frame(
        true_labels, predicted_labels, "BioClinicalBERT"
    )
    predictions = test_frame[[TARGET_COLUMN, TEXT_COLUMN]].copy()
    predictions["predicted_specialty"] = predicted_labels
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "model_results.csv", index=False)
    report.to_csv(output_dir / "classification_report.csv", index=False)
    error_analysis(
        true_labels,
        predicted_labels,
        texts=test_frame[TEXT_COLUMN],
    ).to_csv(output_dir / "errors.csv", index=False)
    return metrics, report


def save_final_model(trainer, tokenizer, output_dir: str | Path) -> Path:
    """Optionally save final weights and tokenizer to an ignored local directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def predict_text(model_path: str | Path, text: str, max_length: int = 256) -> dict:
    """Classify one note with a locally saved fine-tuned model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=1)[0]
    predicted_id = int(probabilities.argmax())
    return {
        "specialty": model.config.id2label[predicted_id],
        "confidence": float(probabilities[predicted_id]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mtsamples.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/bioclinicalbert"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/bert"))
    parser.add_argument("--final-model-dir", type=Path, default=Path("models/final_bioclinicalbert"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--save-final-model", action="store_true")
    args = parser.parse_args()

    frame = load_clinical_notes(args.data)
    train_frame, test_frame = split_frame(frame, random_state=args.random_state)
    trainer, tokenized_test, tokenizer = train_bioclinicalbert(
        train_frame,
        test_frame,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        random_state=args.random_state,
        device=args.device,
    )
    metrics, _ = evaluate_trainer(
        trainer, tokenized_test, test_frame, output_dir=args.results_dir
    )
    if args.save_final_model:
        save_final_model(trainer, tokenizer, args.final_model_dir)
        print(f"Saved final model to {args.final_model_dir}")

    run_config = vars(args).copy()
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in run_config.items()}
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
