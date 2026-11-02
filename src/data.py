"""Dataset loading, cleaning, and deterministic splitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPECIALTIES = [
    "Cardiovascular / Pulmonary",
    "Gastroenterology",
    "Neurology",
    "Obstetrics / Gynecology",
    "Orthopedic",
    "Urology",
]

TEXT_COLUMN = "transcription"
TARGET_COLUMN = "medical_specialty"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20


def load_clinical_notes(path: str | Path) -> pd.DataFrame:
    """Load MT Samples, clean required fields, and retain six specialties."""
    frame = pd.read_csv(path)
    frame = frame.drop(columns=["Unnamed: 0"], errors="ignore")
    frame = frame.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).str.strip()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str).str.strip()
    frame = frame[
        frame[TARGET_COLUMN].isin(SPECIALTIES) & frame[TEXT_COLUMN].ne("")
    ].copy()
    frame["word_count"] = frame[TEXT_COLUMN].str.split().str.len()

    for column in ("ner_entities", "ner_entity_types", "ner_error"):
        if column in frame:
            frame[column] = frame[column].fillna("").astype(str)

    return frame.reset_index(drop=True)


def split_frame(
    frame: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the shared reproducible stratified train/test split."""
    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET_COLUMN],
    )
    return train_frame.copy(), test_frame.copy()


def save_splits(
    frame: pd.DataFrame,
    output_dir: str | Path,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Path, Path]:
    """Create and save the canonical train and test CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, test_frame = split_frame(frame, test_size, random_state)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)
    return train_path, test_path


def build_text_view(frame: pd.DataFrame, view: str = "full_text") -> pd.Series:
    """Return full text, extracted entities, or their concatenation."""
    if view == "full_text":
        return frame[TEXT_COLUMN].fillna("")
    if "ner_entities" not in frame:
        raise ValueError(
            "NER views require a 'ner_entities' column. Run `python -m src.ner` first."
        )
    if view == "ner_only":
        return frame["ner_entities"].fillna("")
    if view == "full_text_ner":
        return (
            frame[TEXT_COLUMN].fillna("")
            + " "
            + frame["ner_entities"].fillna("")
        ).str.strip()
    raise ValueError(f"Unknown text view: {view}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/mtsamples.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    frame = load_clinical_notes(args.input)
    cleaned_path = args.output_dir / "mtsamples_six_specialties.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cleaned_path, index=False)
    train_path, test_path = save_splits(
        frame,
        args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Saved {len(frame):,} cleaned notes to {cleaned_path}")
    print(f"Saved train split to {train_path}")
    print(f"Saved test split to {test_path}")


if __name__ == "__main__":
    main()
