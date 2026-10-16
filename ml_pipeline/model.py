"""Model and vocabulary definitions shared by training and inference."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

import torch
from torch import nn


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


class Vocabulary:
    def __init__(self, tokens: Iterable[str] = ()) -> None:
        self.itos = ["<pad>", "<unk>"] + sorted(set(tokens) - {"<pad>", "<unk>"})
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: Iterable[str], min_frequency: int = 1) -> "Vocabulary":
        counts = Counter(token for text in texts for token in tokenize(text))
        return cls(token for token, count in counts.items() if count >= min_frequency)

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [self.stoi.get(token, 1) for token in tokenize(text)[:max_length]]
        return ids + [0] * (max_length - len(ids))

    def to_dict(self) -> dict[str, int]:
        return self.stoi


class UrgencyClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 96,
        hidden_dim: int = 96,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        encoded, _ = self.encoder(embedded)
        mask = tokens.ne(0).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(self.dropout(pooled))
