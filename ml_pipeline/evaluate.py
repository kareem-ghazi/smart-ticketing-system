"""Evaluate an exported triage model overall and by language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .model import UrgencyClassifier, Vocabulary
from .train_classifier import TicketDataset


def evaluate(data_path: Path, model_path: Path, vocab_path: Path) -> None:
    frame = pd.read_csv(data_path)
    vocabulary = Vocabulary()
    vocabulary.stoi = {
        str(token): int(index)
        for token, index in json.loads(vocab_path.read_text(encoding="utf-8")).items()
    }
    vocabulary.itos = [token for token, _ in sorted(vocabulary.stoi.items(), key=lambda item: item[1])]
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = UrgencyClassifier(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(TicketDataset(frame, vocabulary, checkpoint["max_length"]), batch_size=64)
    predictions: list[int] = []
    labels: list[int] = []
    languages: list[str] = []
    with torch.no_grad():
        for tokens, batch_labels, batch_languages in loader:
            predictions.extend(model(tokens).argmax(dim=1).tolist())
            labels.extend(batch_labels.tolist())
            languages.extend(batch_languages)
    print(classification_report(labels, predictions, target_names=["Normal", "Urgent"], zero_division=0))
    print("confusion_matrix=")
    print(confusion_matrix(labels, predictions))
    for language in sorted(set(languages)):
        indices = [index for index, value in enumerate(languages) if value == language]
        print(f"{language}=")
        print(classification_report(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
            target_names=["Normal", "Urgent"],
            zero_division=0,
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--model", type=Path, default=Path("backend/saved_models/triage_model.pth"))
    parser.add_argument("--vocab", type=Path, default=Path("backend/saved_models/triage_vocab.json"))
    args = parser.parse_args()
    evaluate(args.data, args.model, args.vocab)


if __name__ == "__main__":
    main()
