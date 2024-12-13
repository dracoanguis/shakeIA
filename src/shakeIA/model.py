#!python3

import torch
import torch.nn

from shakeIA import ALPHABET

from typing import TypedDict
from typing_extensions import ReadOnly

ModelConfig = TypedDict(
    "ModelConfig",
    {
        "vector_len": ReadOnly[int],
        "embedding_dim": ReadOnly[int],
        "head_num": ReadOnly[int],
        "forward_expansion": ReadOnly[int],
        "block_number": ReadOnly[int],
    },
)

default_config: ModelConfig = {
    "vector_len": 4,
    "embedding_dim": 4,
    "head_num": 4,
    "forward_expansion": 6,
    "block_number": 2,
}


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

    def forward(self, x: torch.Tensor, vector_len: int) -> torch.Tensor:
        x = self.normalisation1(x)

        causal_mask = torch.triu(torch.ones(vector_len, vector_len), diagonal=1).to(
            x.device
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))

        x = self.multi_head(x, x, x, need_weights=False, attn_mask=causal_mask)[0] + x

        x = self.normalisation2(x)
        x = self.feed_forward(x) + x
        return x


class ShakeModel(torch.nn.Module):

    def __init__(
        self,
        device: torch.device,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = torch.nn.Embedding(
            len(ALPHABET), self.config["embedding_dim"], device=device
        )

        # Tranformation block
        self.blocks = torch.nn.ModuleList(
            [
                TransformationBlock(
                    device,
                    self.config["embedding_dim"],
                    self.config["head_num"],
                    self.config["forward_expansion"],
                )
                for _ in range(self.config["block_number"])
            ]
        )

        # Normalisation
        self.Final_LayerNorm = torch.nn.LayerNorm(
            self.config["embedding_dim"], device=device
        )

        # Getting back the output
        self.linear = torch.nn.Linear(self.config["embedding_dim"], len(ALPHABET))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.embedding(
            torch.arange(0, self.config["vector_len"])
        )

        for block in self.blocks:
            x = block(x, self.config["vector_len"])

        x = self.Final_LayerNorm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
