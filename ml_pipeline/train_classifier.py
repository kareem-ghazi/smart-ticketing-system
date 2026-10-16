"""Train and export the multilingual urgency classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import UrgencyClassifier, Vocabulary

SEED = 42


class TicketDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, frame: pd.DataFrame, vocabulary: Vocabulary, max_length: int) -> None:
        self.tokens = torch.tensor(
            [vocabulary.encode(text, max_length) for text in frame["text"]],
            dtype=torch.long,
        )
        self.labels = torch.tensor(frame["urgency"].to_numpy(), dtype=torch.long)
        self.languages = frame["language"].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.tokens[index], self.labels[index], self.languages[index]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: UrgencyClassifier,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, list[str]]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []
    with torch.set_grad_enabled(training):
        for tokens, batch_labels, _ in loader:
            tokens, batch_labels = tokens.to(device), batch_labels.to(device)
            logits = model(tokens)
            loss = criterion(logits, batch_labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def train(
    processed_dir: Path,
    output_dir: Path,
    epochs: int = 8,
    batch_size: int = 64,
    max_length: int = 128,
) -> dict[str, float]:
    set_seed()
    train_frame = pd.read_csv(processed_dir / "train.csv")
    validation_frame = pd.read_csv(processed_dir / "validation.csv")
    vocabulary = Vocabulary.build(train_frame["text"])
    train_data = TicketDataset(train_frame, vocabulary, max_length)
    validation_data = TicketDataset(validation_frame, vocabulary, max_length)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UrgencyClassifier(len(vocabulary.itos)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        print(f"epoch={epoch} train={train_metrics} validation={validation_metrics}")
        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            metrics = {f"validation_{key}": value for key, value in validation_metrics.items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "architecture": {"vocabulary_size": len(vocabulary.itos)},
            "class_mapping": {"0": "Normal", "1": "Urgent"},
            "max_length": max_length,
            "preprocessing_version": 1,
        },
        output_dir / "triage_model.pth",
    )
    (output_dir / "triage_vocab.json").write_text(
        json.dumps(vocabulary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "triage_config.json").write_text(
        json.dumps(
            {
                "architecture": {"vocabulary_size": len(vocabulary.itos)},
                "max_length": max_length,
                "class_mapping": {"0": "Normal", "1": "Urgent"},
                "device": str(device),
                "seed": SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("backend/saved_models"))
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    train(args.processed_dir, args.output_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
