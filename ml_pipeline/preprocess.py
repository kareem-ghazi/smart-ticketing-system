"""Prepare multilingual review data for urgency classification."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import DatasetDict, load_dataset
from datasets.exceptions import DataFilesNotFoundError
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

SEED = 42
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TEXT_COLUMNS = (
    "text",
    "review",
    "reviewtext",
    "sentence",
    "content",
    "comment",
    "query",
)
LABEL_COLUMNS = (
    "label",
    "sentiment",
    "rating",
    "overall",
    "score",
    "stars",
    "polarity",
)


def _records_from_dataset(dataset: Any) -> list[dict[str, Any]]:
    """Convert a Hugging Face dataset or split collection into records."""
    if isinstance(dataset, DatasetDict):
        records: list[dict[str, Any]] = []
        for split in dataset.values():
            records.extend(split.to_list())
        return records
    if hasattr(dataset, "to_list"):
        return dataset.to_list()
    if isinstance(dataset, Mapping):
        return [dict(dataset)]
    if isinstance(dataset, Iterable):
        return [dict(record) for record in dataset]
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def _find_column(records: list[dict[str, Any]], candidates: tuple[str, ...]) -> str:
    available = {str(key).lower(): str(key) for key in records[0]}
    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    raise ValueError(
        f"Could not find one of {candidates}; available columns are "
        f"{sorted(available.values())}"
    )


def normalize_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def _numeric_label(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number in (0, 1):
        return int(number)
    return int(number <= 2) if number >= 0 else None


def normalize_label(value: Any) -> int:
    numeric = _numeric_label(value)
    if numeric is not None:
        return numeric
    label = str(value).strip().lower()
    urgent_terms = ("negative", "angry", "urgent", "bad", "complaint", "dissatisfied")
    normal_terms = ("positive", "neutral", "normal", "good", "satisfied")
    if any(term in label for term in urgent_terms):
        return 1
    if any(term in label for term in normal_terms):
        return 0
    raise ValueError(f"Unrecognized sentiment label: {value!r}")


def records_to_frame(dataset: Any, language: str) -> pd.DataFrame:
    records = _records_from_dataset(dataset)
    if not records:
        raise ValueError(f"The {language} dataset is empty.")
    text_column = _find_column(records, TEXT_COLUMNS)
    label_column = _find_column(records, LABEL_COLUMNS)
    frame = pd.DataFrame(
        {
            "text": [normalize_text(record.get(text_column)) for record in records],
            "urgency": [normalize_label(record.get(label_column)) for record in records],
            "language": language,
        }
    )
    return frame[frame["text"].str.len() > 0].drop_duplicates(subset=["text"])


def _stratified_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, remainder = train_test_split(
        frame,
        test_size=0.2,
        random_state=SEED,
        stratify=frame[["urgency", "language"]].astype(str).agg("_".join, axis=1),
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        random_state=SEED,
        stratify=remainder[["urgency", "language"]].astype(str).agg("_".join, axis=1),
    )
    return train, validation, test


def _load_source_dataset(dataset_id: str) -> Any:
    try:
        return load_dataset(dataset_id)
    except DataFilesNotFoundError as error:
        raise RuntimeError(
            f"Hugging Face dataset '{dataset_id}' has no supported data files. "
            "Choose an updated dataset ID or provide local data before rerunning."
        ) from error


def prepare_data(
    output_dir: Path,
    max_rows_per_class: int | None = None,
    english_dataset: str = "hugginglearners/amazon-reviews-sentiment-analysis",
    arabic_dataset: str = "Ruqiya/Arabic_Reviews_of_SHEIN",
) -> dict[str, Path]:
    """Download, normalize, balance, split, and save the source datasets."""
    random.seed(SEED)
    english = records_to_frame(
        _load_source_dataset(english_dataset), "en"
    )
    arabic = records_to_frame(_load_source_dataset(arabic_dataset), "ar")
    frame = pd.concat([english, arabic], ignore_index=True)

    if max_rows_per_class is not None:
        sampled_groups = [
            group.sample(n=min(len(group), max_rows_per_class), random_state=SEED)
            for _, group in frame.groupby(["language", "urgency"], sort=True)
        ]
        frame = pd.concat(sampled_groups, ignore_index=True)

    if frame["urgency"].value_counts().min() < 3:
        raise ValueError("Each urgency class needs at least three records.")

    train, validation, test = _stratified_split(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        path = output_dir / f"{name}.csv"
        split.sample(frac=1, random_state=SEED).to_csv(path, index=False)
        outputs[name] = path

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "rows": {name: int(len(split)) for name, split in (
                    ("train", train), ("validation", validation), ("test", test)
                )},
                "class_mapping": {"0": "Normal", "1": "Urgent"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs["metadata"] = metadata_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-rows-per-class", type=int)
    parser.add_argument(
        "--english-dataset",
        default="hugginglearners/amazon-reviews-sentiment-analysis",
    )
    parser.add_argument(
        "--arabic-dataset",
        default="Ruqiya/Arabic_Reviews_of_SHEIN",
    )
    args = parser.parse_args()
    for path in prepare_data(
        args.output_dir,
        args.max_rows_per_class,
        args.english_dataset,
        args.arabic_dataset,
    ).values():
        print(path)


if __name__ == "__main__":
    main()
