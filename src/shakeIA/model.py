#!python3

import torch
import torch.nn

from .dataset import ALPHABET


class TransformationBlock(torch.nn.Module):

    def __init__(
        self,
        device: torch.device,
        embedding_dim: int = 4,
        head_num: int = 4,
        forward_expansion: int = 3,
    ):
        # Transformer block
        self.multi_head = torch.nn.MultiheadAttention(
            embedding_dim, head_num, device=device
        )

        # Feed forward
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(
                embedding_dim, forward_expansion * embedding_dim, device=device
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                forward_expansion * embedding_dim, embedding_dim, device=device
            ),
        )

        self.normalisation1 = torch.nn.LayerNorm(embedding_dim, device=device)
        self.normalisation2 = torch.nn.LayerNorm(embedding_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalisation1(x)
        x = self.multi_head(x, x, x, need_weights=False) + x

        x = self.normalisation2(x)
        x = self.feed_forward(x) + x
        return x


class ShakeModel(torch.nn.Module):

    def __init__(
        self,
        device: torch.device,
        embedding_dim: int = 4,
        head_num: int = 4,
        forward_expansion: int = 3,
        block_number: int = 1,
    ):
        super().__init__()

        # Embedding layer
        self.embedding = torch.nn.Embedding(len(ALPHABET), embedding_dim, device=device)
        self.pos_embedding = torch.nn.Embedding(4, embedding_dim, device=device)

        # Tranformation block
        self.blocks = torch.nn.ModuleList(
            TransformationBlock(device, embedding_dim, head_num, forward_expansion)
            for _ in range(block_number)
        )

        # Normalisation
        self.Final_LayerNorm = torch.nn.LayerNorm(embedding_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.pos_embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.Final_LayerNorm(x)
        return x
