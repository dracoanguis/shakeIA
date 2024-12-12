#!python3

import torch
import torch.nn

from shakeIA import ALPHABET


class TransformationBlock(torch.nn.Module):

    def __init__(
        self,
        device: torch.device,
        embedding_dim: int,
        head_num: int,
        forward_expansion: int,
    ):
        super().__init__()

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
        x = self.multi_head(x, x, x, need_weights=False)[0] + x

        x = self.normalisation2(x)
        x = self.feed_forward(x) + x
        return x


class ShakeModel(torch.nn.Module):

    def __init__(
        self,
        device: torch.device,
        vector_len: int = 4,
        embedding_dim: int = 4,
        head_num: int = 4,
        forward_expansion: int = 3,
        block_number: int = 1,
    ):
        super().__init__()

        self.config = {
            "vector_len": vector_len,
            "embedding_dim": embedding_dim,
            "head_num": head_num,
            "forward_expansion": forward_expansion,
            "block_number": block_number,
        }

        self.vector_len = vector_len

        # Embedding layer
        self.embedding = torch.nn.Embedding(len(ALPHABET), embedding_dim, device=device)

        # Tranformation block
        self.blocks = torch.nn.ModuleList(
<<<<<<< Updated upstream
            TransformationBlock(device, embedding_dim, head_num, forward_expansion)
            for _ in block_number
=======
            [
                TransformationBlock(device, embedding_dim, head_num, forward_expansion)
                for _ in range(block_number)
            ]
>>>>>>> Stashed changes
        )

        # Normalisation
        self.Final_LayerNorm = torch.nn.LayerNorm(embedding_dim, device=device)

        # Getting back the output
        self.linear = torch.nn.Linear(embedding_dim, len(ALPHABET))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.embedding(torch.arange(0, self.vector_len))

        for block in self.blocks:
            x = block(x)

        x = self.Final_LayerNorm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
