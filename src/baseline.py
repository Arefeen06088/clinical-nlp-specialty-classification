"""Train and evaluate all TF-IDF + logistic-regression experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .data import TARGET_COLUMN, build_text_view, load_clinical_notes, split_frame
from .evaluate import classification_report_frame, error_analysis
from .utils import set_seed

EXPERIMENTS = {
    "full_text": {
        "model": "TF-IDF + Logistic Regression",
        "input": "Full clinical text",
        "artifact": "tfidf_full_text.joblib",
    },
    "ner_only": {
        "model": "NER-only + TF-IDF + Logistic Regression",
        "input": "Biomedical entities only",
        "artifact": "tfidf_ner_only.joblib",
    },
    "full_text_ner": {
        "model": "Full Text + NER + TF-IDF + Logistic Regression",
        "input": "Full text + biomedical entities",
        "artifact": "tfidf_full_text_ner.joblib",
    },
}


def build_baseline(max_features: int = 10_000, random_state: int = 42) -> Pipeline:
    """Construct the baseline classifier shared by all three text views."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(stop_words="english", max_features=max_features),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=random_state),
            ),
        ]
    )


def fit_baseline(
    train_frame: pd.DataFrame,
    view: str = "full_text",
    max_features: int = 10_000,
    random_state: int = 42,
) -> Pipeline:
    """Fit a baseline against one input-text view."""
    model = build_baseline(max_features=max_features, random_state=random_state)
    model.fit(build_text_view(train_frame, view), train_frame[TARGET_COLUMN])
    return model


def predict_baseline(
    model: Pipeline,
    test_frame: pd.DataFrame,
    view: str = "full_text",
) -> np.ndarray:
    """Predict medical specialties for a held-out frame."""
    return model.predict(build_text_view(test_frame, view))


def top_features(model: Pipeline, top_n: int = 10) -> pd.DataFrame:
    """Return the strongest positive TF-IDF features per specialty."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    rows = []

    for class_index, specialty in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        for rank, feature_index in enumerate(
            np.argsort(coefficients)[-top_n:][::-1], start=1
        ):
            rows.append(
                {
                    "specialty": specialty,
                    "feature": feature_names[feature_index],
                    "coefficient": coefficients[feature_index],
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows)


def train_baseline_experiments(
    frame: pd.DataFrame,
    output_dir: str | Path = "results/baselines",
    model_dir: str | Path = "models/baselines",
    random_state: int = 42,
    max_features: int = 10_000,
    save_models: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train, evaluate, and persist all three baseline experiments."""
    set_seed(random_state)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    train_frame, test_frame = split_frame(frame, random_state=random_state)
    metric_rows = []
    report_frames = []

    for view, metadata in EXPERIMENTS.items():
        model = fit_baseline(
            train_frame,
            view=view,
            max_features=max_features,
            random_state=random_state,
        )
        predictions = predict_baseline(model, test_frame, view=view)
        metric_rows.append(
            {
                "Model": metadata["model"],
                "Input": metadata["input"],
                "Accuracy": accuracy_score(test_frame[TARGET_COLUMN], predictions),
                "Macro_F1": f1_score(
                    test_frame[TARGET_COLUMN], predictions, average="macro"
                ),
            }
        )
        report_frames.append(
            classification_report_frame(
                test_frame[TARGET_COLUMN], predictions, metadata["model"]
            )
        )
        prediction_frame = test_frame[[TARGET_COLUMN, "transcription"]].copy()
        prediction_frame["predicted_specialty"] = predictions
        prediction_frame.to_csv(
            output_dir / f"predictions_{view}.csv", index=False
        )
        error_analysis(
            test_frame[TARGET_COLUMN],
            predictions,
            texts=test_frame["transcription"],
        ).to_csv(output_dir / f"errors_{view}.csv", index=False)

        if view == "full_text":
            top_features(model, top_n=15).to_csv(
                output_dir / "top_tfidf_features.csv", index=False
            )
        if save_models:
            joblib.dump(model, model_dir / metadata["artifact"])

    metrics = pd.DataFrame(metric_rows)
    reports = pd.concat(report_frames, ignore_index=True)
    metrics.to_csv(output_dir / "model_results.csv", index=False)
    reports.to_csv(output_dir / "classification_reports.csv", index=False)
    return metrics, reports


def predict_text(model_path: str | Path, text: str) -> str:
    """Load a saved baseline pipeline and classify one note."""
    model = joblib.load(model_path)
    return str(model.predict([text])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/mtsamples_with_ner.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/baselines"))
    parser.add_argument("--max-features", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()

    frame = load_clinical_notes(args.data)
    metrics, _ = train_baseline_experiments(
        frame,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        random_state=args.random_state,
        max_features=args.max_features,
        save_models=not args.no_save_models,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
