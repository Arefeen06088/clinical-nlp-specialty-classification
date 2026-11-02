"""Biomedical NER extraction with resumable CSV caching."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .data import TEXT_COLUMN, load_clinical_notes

DEFAULT_NER_MODEL = "d4data/biomedical-ner-all"
DEFAULT_MAX_CHARACTERS = 2_000


def resolve_pipeline_device(device: str) -> int | str:
    """Map a friendly device option to a Transformers pipeline device."""
    if device != "auto":
        return -1 if device == "cpu" else device

    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError):
        pass
    return -1


def load_ner_pipeline(
    model_name: str = DEFAULT_NER_MODEL,
    device: str = "auto",
):
    """Load the biomedical Hugging Face NER pipeline lazily."""
    from transformers import pipeline

    return pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
        device=resolve_pipeline_device(device),
    )


def extract_entity_record(
    text: object,
    ner_pipeline,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
) -> dict[str, str | int]:
    """Extract entity text and types while recording empty/failed cases."""
    if not isinstance(text, str) or not text.strip():
        return {
            "ner_entities": "",
            "ner_entity_types": "[]",
            "ner_entity_count": 0,
            "ner_error": "empty_input",
        }

    try:
        entities = ner_pipeline(text[:max_characters])
    except Exception as error:
        return {
            "ner_entities": "",
            "ner_entity_types": "[]",
            "ner_entity_count": 0,
            "ner_error": f"{type(error).__name__}: {error}",
        }

    words = [str(entity.get("word", "")).strip() for entity in entities]
    words = [word for word in words if word]
    entity_types = [
        str(entity.get("entity_group") or entity.get("entity") or "UNKNOWN")
        for entity in entities
    ]
    return {
        "ner_entities": " ".join(words),
        "ner_entity_types": json.dumps(entity_types),
        "ner_entity_count": len(words),
        "ner_error": "" if words else "no_entities",
    }


def extract_entities(text: object, ner_pipeline, max_characters: int = 2_000) -> str:
    """Return only the space-delimited entity text for one note."""
    return str(
        extract_entity_record(text, ner_pipeline, max_characters)["ner_entities"]
    )


def extract_entity_series(
    texts: Iterable[object],
    ner_pipeline,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
) -> pd.Series:
    """Extract entity text for a sequence while preserving its length."""
    return pd.Series(
        [extract_entities(text, ner_pipeline, max_characters) for text in texts],
        dtype="string",
    )


def enrich_with_entities(
    frame: pd.DataFrame,
    output_path: str | Path,
    ner_pipeline,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    checkpoint_every: int = 25,
    resume: bool = True,
) -> pd.DataFrame:
    """Extract entities, checkpoint progress, and resume an interrupted run."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = frame.reset_index(drop=True).copy()

    completed = 0
    if resume and output_path.exists():
        cached = pd.read_csv(output_path).fillna("")
        if len(cached) > len(enriched):
            raise ValueError("Cached NER file has more rows than the input dataset.")
        required_columns = {
            "ner_entities",
            "ner_entity_types",
            "ner_entity_count",
            "ner_error",
        }
        if not required_columns.issubset(cached.columns):
            cached = cached.iloc[0:0]
        for column in (
            "ner_entities",
            "ner_entity_types",
            "ner_entity_count",
            "ner_error",
        ):
            if column in cached:
                enriched.loc[: len(cached) - 1, column] = cached[column].values
        completed = len(cached)

    for row_index in range(completed, len(enriched)):
        record = extract_entity_record(
            enriched.at[row_index, TEXT_COLUMN],
            ner_pipeline,
            max_characters=max_characters,
        )
        for column, value in record.items():
            enriched.at[row_index, column] = value

        processed = row_index + 1
        if processed % checkpoint_every == 0 or processed == len(enriched):
            enriched.iloc[:processed].to_csv(output_path, index=False)
            print(f"NER progress: {processed:,}/{len(enriched):,}")

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/mtsamples.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/mtsamples_with_ner.csv")
    )
    parser.add_argument("--model", default=DEFAULT_NER_MODEL)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--max-characters", type=int, default=DEFAULT_MAX_CHARACTERS)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    frame = load_clinical_notes(args.input)
    ner_pipeline = load_ner_pipeline(args.model, args.device)
    enriched = enrich_with_entities(
        frame,
        args.output,
        ner_pipeline,
        max_characters=args.max_characters,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    missing = enriched["ner_entities"].fillna("").eq("").sum()
    print(f"Saved {len(enriched):,} rows to {args.output}")
    print(f"Rows without extracted entities: {missing:,}")


if __name__ == "__main__":
    main()
