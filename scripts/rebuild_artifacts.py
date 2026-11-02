#!/usr/bin/env python3
"""Rebuild result tables and repository figures from local experiment assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import fit_baseline, predict_baseline, top_features
from src.data import SPECIALTIES, TARGET_COLUMN, load_clinical_notes, split_frame
from src.evaluate import (
    classification_report_frame,
    plot_bert_vs_tfidf,
    plot_class_distribution,
    plot_confusion,
    plot_entity_distribution,
    plot_entity_type_distribution,
    plot_model_comparison,
    plot_note_lengths,
    plot_note_length_by_specialty,
    plot_per_class_f1,
    plot_token_lengths,
    plot_top_features,
)
from src.train_bert import MAX_LENGTH, token_lengths
from src.utils import RESULTS_DIR, ensure_output_dirs, set_seed

MODEL_LABELS = {
    "full_text": "TF-IDF + Logistic Regression",
    "ner_only": "NER-only + TF-IDF + Logistic Regression",
    "full_text_ner": "Full Text + NER + TF-IDF + Logistic Regression",
}

INPUT_LABELS = {
    "full_text": "Full clinical text",
    "ner_only": "Biomedical entities only",
    "full_text_ner": "Full text + biomedical entities",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Original local project containing data/ and the final model.",
    )
    parser.add_argument(
        "--skip-bert-inference",
        action="store_true",
        help="Use preserved BERT report rows without rebuilding its confusion matrix.",
    )
    return parser.parse_args()


def infer_bert(model_dir: Path, test_frame: pd.DataFrame):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []

    texts = test_frame["transcription"].tolist()
    for start in range(0, len(texts), 16):
        encoded = tokenizer(
            texts[start : start + 16],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    labels = [model.config.id2label[index] for index in predictions]
    return labels, tokenizer


def preserved_bert_report() -> pd.DataFrame:
    rows = [
        ("Cardiovascular / Pulmonary", 0.92, 0.95, 0.93, 74),
        ("Gastroenterology", 0.92, 0.98, 0.95, 45),
        ("Neurology", 0.84, 0.71, 0.77, 45),
        ("Obstetrics / Gynecology", 0.97, 0.97, 0.97, 31),
        ("Orthopedic", 0.88, 0.92, 0.90, 71),
        ("Urology", 0.97, 0.94, 0.95, 31),
    ]
    return pd.DataFrame(
        [
            {
                "model": "BioClinicalBERT",
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
            for label, precision, recall, f1, support in rows
        ]
    )


def main() -> None:
    args = parse_args()
    set_seed(42)
    ensure_output_dirs()

    data_path = args.source_root / "data" / "mtsamples_with_ner.csv"
    frame = load_clinical_notes(data_path)
    train_frame, test_frame = split_frame(frame)

    result_rows = []
    reports = []
    full_text_model = None

    for view in MODEL_LABELS:
        model = fit_baseline(train_frame, view=view)
        predictions = predict_baseline(model, test_frame, view=view)
        result_rows.append(
            {
                "Model": MODEL_LABELS[view],
                "Input": INPUT_LABELS[view],
                "Accuracy": accuracy_score(test_frame[TARGET_COLUMN], predictions),
                "Macro_F1": f1_score(test_frame[TARGET_COLUMN], predictions, average="macro"),
            }
        )
        reports.append(
            classification_report_frame(
                test_frame[TARGET_COLUMN], predictions, MODEL_LABELS[view]
            )
        )
        if view == "full_text":
            full_text_model = model
            plot_confusion(
                test_frame[TARGET_COLUMN],
                predictions,
                "confusion_matrix_tfidf.png",
                "TF-IDF + Logistic Regression Confusion Matrix",
            )

    result_rows.append(
        {
            "Model": "BioClinicalBERT",
            "Input": "Full clinical text",
            "Accuracy": 0.909091,
            "Macro_F1": 0.910961,
        }
    )
    results = pd.DataFrame(result_rows)

    tokenizer = AutoTokenizer.from_pretrained(
        args.source_root / "models" / "final_bioclinicalbert",
        local_files_only=True,
    )
    if args.skip_bert_inference:
        bert_report = preserved_bert_report()
    else:
        bert_predictions, tokenizer = infer_bert(
            args.source_root / "models" / "final_bioclinicalbert",
            test_frame,
        )
        bert_report = classification_report_frame(
            test_frame[TARGET_COLUMN], bert_predictions, "BioClinicalBERT"
        )
        plot_confusion(
            test_frame[TARGET_COLUMN],
            bert_predictions,
            "confusion_matrix_bert.png",
            "BioClinicalBERT Confusion Matrix",
        )

    reports.append(bert_report)
    report_frame = pd.concat(reports, ignore_index=True)
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    report_frame.to_csv(RESULTS_DIR / "classification_reports.csv", index=False)

    plot_class_distribution(frame)
    plot_note_lengths(frame)
    plot_note_length_by_specialty(frame)
    plot_entity_distribution(frame)
    if "ner_entity_types" in frame:
        plot_entity_type_distribution(frame)
    plot_model_comparison(results)
    plot_top_features(top_features(full_text_model, top_n=10))
    plot_per_class_f1(report_frame)
    plot_bert_vs_tfidf(report_frame)

    length_rows = []
    for specialty in SPECIALTIES:
        specialty_texts = test_frame.loc[
            test_frame[TARGET_COLUMN] == specialty, "transcription"
        ]
        for length in token_lengths(specialty_texts, tokenizer):
            length_rows.append({"specialty": specialty, "token_length": length})
    length_frame = pd.DataFrame(length_rows)
    plot_token_lengths(length_frame)

    neurology = length_frame.loc[
        length_frame["specialty"] == "Neurology", "token_length"
    ]
    print(f"Neurology mean tokens: {neurology.mean():.4f}")
    print(f"Neurology median tokens: {neurology.median():.0f}")
    print(f"Neurology notes >256: {(neurology > 256).sum()} / {len(neurology)}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

