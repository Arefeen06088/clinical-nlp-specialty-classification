"""Run inference with a saved TF-IDF or BioClinicalBERT model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .baseline import predict_text as predict_baseline_text
from .train_bert import predict_text as predict_bert_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=["tfidf", "bert"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text")
    text_group.add_argument("--text-file", type=Path)
    args = parser.parse_args()

    text = args.text if args.text is not None else args.text_file.read_text()
    if args.model_type == "tfidf":
        result = {"specialty": predict_baseline_text(args.model_path, text)}
    else:
        result = predict_bert_text(args.model_path, text)
    print(result)


if __name__ == "__main__":
    main()

