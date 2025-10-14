# -*- coding: utf-8 -*-
"""
Definitionen für PyTorch-basierte KI-Modelle wie Autoencoder.
"""
from __future__ import annotations

import torch
from torch import nn


class ClientAutoencoder(nn.Module):
    """
    Ein einfacher Autoencoder, um Client-Features in einen dichten latenten Raum (Embedding) zu komprimieren.
    Das Ziel ist, die wichtigsten Muster in den Daten zu lernen.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 16):
        super().__init__()

        # Der Encoder komprimiert die Daten
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),  # Komprimiert auf die finale Embedding-Größe
        )

        # Der Decoder versucht, die Originaldaten aus der Kompression wiederherzustellen
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Expandiert zurück auf die originale Dimension
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Führt einen vollen Encoder-Decoder-Durchlauf aus (für das Training)."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Führt nur den Encoder-Teil aus, um das Embedding zu erhalten (für das Clustering)."""
        return self.encoder(x)
