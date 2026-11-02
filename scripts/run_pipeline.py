#!/usr/bin/env python3
"""Run data preparation, NER, baselines, and optional BERT fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import train_baseline_experiments
from src.data import load_clinical_notes, save_splits, split_frame
from src.ner import enrich_with_entities, load_ner_pipeline
from src.train_bert import evaluate_trainer, save_final_model, train_bioclinicalbert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, default=Path("data/mtsamples.csv"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--skip-ner", action="store_true")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--save-final-bert", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"
    models_dir = PROJECT_ROOT / "models"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    frame = load_clinical_notes(args.raw_data)
    cleaned_path = data_dir / "mtsamples_six_specialties.csv"
    frame.to_csv(cleaned_path, index=False)
    save_splits(frame, data_dir)

    ner_path = data_dir / "mtsamples_with_ner.csv"
    if args.skip_ner:
        if not ner_path.exists():
            raise FileNotFoundError(
                f"{ner_path} does not exist. Remove --skip-ner or provide the cache."
            )
        enriched = load_clinical_notes(ner_path)
    else:
        ner_pipeline = load_ner_pipeline(device=args.device)
        enriched = enrich_with_entities(frame, ner_path, ner_pipeline)

    baseline_metrics, baseline_reports = train_baseline_experiments(
        enriched,
        output_dir=results_dir / "baselines",
        model_dir=models_dir / "baselines",
    )
    all_metrics = [baseline_metrics]
    all_reports = [baseline_reports]

    if not args.skip_bert:
        train_frame, test_frame = split_frame(frame)
        trainer, tokenized_test, tokenizer = train_bioclinicalbert(
            train_frame,
            test_frame,
            output_dir=models_dir / "bioclinicalbert",
            device=args.device,
        )
        bert_metrics, bert_report = evaluate_trainer(
            trainer,
            tokenized_test,
            test_frame,
            output_dir=results_dir / "bert",
        )
        all_metrics.append(bert_metrics)
        all_reports.append(bert_report)
        if args.save_final_bert:
            save_final_model(trainer, tokenizer, models_dir / "final_bioclinicalbert")

    pd.concat(all_metrics, ignore_index=True).to_csv(
        results_dir / "model_results_reproduced.csv", index=False
    )
    pd.concat(all_reports, ignore_index=True).to_csv(
        results_dir / "classification_reports_reproduced.csv", index=False
    )
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
