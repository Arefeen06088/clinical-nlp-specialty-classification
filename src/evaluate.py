"""Evaluation tables and Seaborn visualizations."""

from __future__ import annotations

import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .data import SPECIALTIES, TARGET_COLUMN
from .utils import configure_plots, save_figure


def classification_report_frame(y_true, y_pred, model_name: str) -> pd.DataFrame:
    """Return a tidy per-class classification report."""
    report = classification_report(
        y_true,
        y_pred,
        labels=SPECIALTIES,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "model": model_name,
                "class": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": metrics["support"],
            }
        )
    return pd.DataFrame(rows)


def summary_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the two primary experiment metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def error_analysis(y_true, y_pred, texts=None) -> pd.DataFrame:
    """Return misclassified examples for qualitative analysis."""
    errors = pd.DataFrame(
        {"true_specialty": list(y_true), "predicted_specialty": list(y_pred)}
    )
    if texts is not None:
        errors["text"] = list(texts)
    return errors[
        errors["true_specialty"] != errors["predicted_specialty"]
    ].reset_index(drop=True)


def plot_class_distribution(frame: pd.DataFrame):
    configure_plots()
    counts = frame[TARGET_COLUMN].value_counts().reindex(SPECIALTIES)
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, legend=False, ax=axis)
    axis.set(title="Clinical Notes by Specialty", xlabel="Number of notes", ylabel="")
    axis.bar_label(axis.containers[0], padding=4)
    sns.despine()
    save_figure(figure, "class_distribution.png")
    return figure


def plot_note_lengths(frame: pd.DataFrame):
    configure_plots()
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(frame, x="word_count", bins=40, kde=True, color="#3B82F6", ax=axis)
    axis.axvline(frame["word_count"].median(), color="#D97706", linestyle="--", label="Median")
    axis.set(title="Clinical Note Length Distribution", xlabel="Words per note", ylabel="Notes")
    axis.legend(frameon=False)
    sns.despine()
    save_figure(figure, "note_length_distribution.png")
    return figure


def plot_note_length_by_specialty(frame: pd.DataFrame):
    """Compare note word-count distributions across specialties."""
    configure_plots()
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.boxenplot(
        data=frame,
        x="word_count",
        y=TARGET_COLUMN,
        order=SPECIALTIES,
        color="#93C5FD",
        ax=axis,
    )
    axis.set(
        title="Clinical Note Length by Specialty",
        xlabel="Words per note",
        ylabel="",
    )
    sns.despine()
    save_figure(figure, "note_length_by_specialty.png")
    return figure


def plot_top_features(features: pd.DataFrame):
    configure_plots()
    data = features[features["rank"] <= 8].copy()
    grid = sns.catplot(
        data=data,
        x="coefficient",
        y="feature",
        col="specialty",
        col_wrap=2,
        kind="bar",
        sharey=False,
        height=3.4,
        aspect=1.35,
        color="#2563EB",
    )
    grid.set_axis_labels("Logistic-regression coefficient", "")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Most Predictive TF-IDF Features by Specialty", y=1.02, fontsize=16)
    save_figure(grid.figure, "top_tfidf_features.png")
    return grid.figure


def plot_entity_distribution(frame: pd.DataFrame, top_n: int = 20):
    configure_plots()
    tokens = " ".join(frame["ner_entities"].fillna("")).lower().split()
    counts = pd.DataFrame(Counter(tokens).most_common(top_n), columns=["entity_token", "count"])
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.barplot(data=counts, x="count", y="entity_token", color="#0F766E", ax=axis)
    axis.set(title="Most Frequent Extracted Biomedical Entity Tokens", xlabel="Occurrences", ylabel="")
    sns.despine()
    save_figure(figure, "ner_entity_distribution.png")
    return figure


def plot_entity_type_distribution(frame: pd.DataFrame, top_n: int = 20):
    """Plot counts of NER model entity groups saved during extraction."""
    if "ner_entity_types" not in frame:
        raise ValueError(
            "Entity-type plotting requires 'ner_entity_types'. Re-run `python -m src.ner`."
        )

    entity_types = []
    for value in frame["ner_entity_types"].fillna("[]"):
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            parsed = [item for item in str(value).split("|") if item]
        entity_types.extend(parsed)

    counts = pd.DataFrame(
        Counter(entity_types).most_common(top_n), columns=["entity_type", "count"]
    )
    counts["entity_type"] = counts["entity_type"].str.replace("_", " ")
    configure_plots()
    figure, axis = plt.subplots(figsize=(11, 7))
    sns.barplot(data=counts, x="count", y="entity_type", color="#0F766E", ax=axis)
    axis.set(
        title=f"Top {top_n} Biomedical NER Entity Types",
        xlabel="Extracted entities",
        ylabel="",
    )
    axis.bar_label(axis.containers[0], padding=4, fmt="{:,.0f}")
    sns.despine()
    save_figure(figure, "ner_entity_type_distribution.png")
    return figure


def plot_model_comparison(results: pd.DataFrame):
    configure_plots()
    data = results.melt(
        id_vars=["Model", "Input"],
        value_vars=["Accuracy", "Macro_F1"],
        var_name="Metric",
        value_name="Score",
    )
    figure, axis = plt.subplots(figsize=(12, 6.5))
    sns.barplot(data=data, x="Score", y="Model", hue="Metric", ax=axis)
    axis.set(xlim=(0.78, 0.93), title="Model Performance", xlabel="Score", ylabel="")
    axis.legend(title="", loc="lower right")
    sns.despine()
    save_figure(figure, "model_comparison.png")
    return figure


def plot_confusion(y_true, y_pred, filename: str, title: str):
    configure_plots()
    matrix = confusion_matrix(y_true, y_pred, labels=SPECIALTIES)
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=SPECIALTIES,
        yticklabels=SPECIALTIES,
        ax=axis,
    )
    axis.set(title=title, xlabel="Predicted specialty", ylabel="True specialty")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    save_figure(figure, filename)
    return figure


def plot_per_class_f1(reports: pd.DataFrame):
    configure_plots()
    data = reports[reports["class"].isin(SPECIALTIES)].copy()
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.barplot(data=data, x="f1_score", y="class", hue="model", ax=axis)
    axis.set(xlim=(0.65, 1.0), title="Per-Specialty F1 Scores", xlabel="F1 score", ylabel="")
    axis.legend(title="", loc="lower right")
    sns.despine()
    save_figure(figure, "per_class_f1.png")
    return figure


def plot_bert_vs_tfidf(reports: pd.DataFrame):
    configure_plots()
    selected = reports[
        reports["model"].isin(["TF-IDF + Logistic Regression", "BioClinicalBERT"])
        & reports["class"].isin(SPECIALTIES)
    ].copy()
    pivot = selected.pivot(index="class", columns="model", values="f1_score").reindex(SPECIALTIES)
    pivot["change"] = pivot["BioClinicalBERT"] - pivot["TF-IDF + Logistic Regression"]
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = np.where(pivot["change"] >= 0, "#0F766E", "#C2410C")
    axis.barh(pivot.index, pivot["change"], color=colors)
    axis.axvline(0, color="#334155", linewidth=1)
    axis.set(title="BioClinicalBERT F1 Change vs. TF-IDF", xlabel="Change in F1 score", ylabel="")
    sns.despine()
    save_figure(figure, "bert_vs_tfidf_f1.png")
    return figure


def plot_token_lengths(length_frame: pd.DataFrame, max_length: int = 256):
    configure_plots()
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.boxenplot(data=length_frame, x="token_length", y="specialty", color="#60A5FA", ax=axis)
    axis.axvline(max_length, color="#DC2626", linestyle="--", label=f"{max_length}-token limit")
    axis.set(title="BioClinicalBERT Token Length by Specialty", xlabel="Untruncated tokens", ylabel="")
    axis.legend(frameon=False)
    sns.despine()
    save_figure(figure, "token_length_by_specialty.png")
    return figure

