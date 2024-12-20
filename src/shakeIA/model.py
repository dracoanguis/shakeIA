#!python3

import torch
import torch.nn

from .dataset import ALPHABET, STOI, ITOS

from typing import TypedDict
from typing_extensions import ReadOnly

from functools import reduce

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
    "embedding_dim": 12,
    "head_num": 3,
    "forward_expansion": 6,
    "block_number": 2,
}


class TransformationBlock(torch.nn.Module):
    """
    A Transformer-based module that performs multi-head self-attention and feed-forward operations
    with layer normalization and residual connections.

    Attributes:
        multi_head (torch.nn.MultiheadAttention): Multi-head self-attention module.
        feed_forward (torch.nn.Sequential): Feed-forward neural network with a hidden layer
                                             expanded by `forward_expansion`.
        normalisation1 (torch.nn.LayerNorm): Layer normalization applied before the attention module.
        normalisation2 (torch.nn.LayerNorm): Layer normalization applied before the feed-forward module.

    Methods:
        forward(x: torch.Tensor, vector_len: int) -> torch.Tensor:
            Computes the forward pass of the transformation block, including multi-head self-attention,
            a causal mask for autoregressive behavior, and feed-forward operations.
    """

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

        # causal_mask = torch.triu(torch.ones(vector_len, vector_len), diagonal=1).to(
        # x.device
        # )
        # causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
        causal_mask = None

        x = self.multi_head(x, x, x, need_weights=False, attn_mask=causal_mask)[0] + x

        x = self.normalisation2(x)
        x = self.feed_forward(x) + x
        return x


class ShakeModel(torch.nn.Module):
    """
    A Transformer-based sequence model designed for autoregressive tasks over a predefined
    vocabulary. It employs embedding layers, multiple transformation blocks, and a final
    linear layer to generate predictions.

    Attributes:
        config (dict): Configuration of the model parameters, including:
            - vector_len (int): Input sequence length.
            - embedding_dim (int): Dimension of the embedding space.
            - head_num (int): Number of attention heads in the transformer block.
            - forward_expansion (int): Expansion factor for the hidden layer in the feed-forward module.
            - block_number (int): Number of transformation blocks in the model.
        vector_len (int): Length of the input sequence.
        embedding (torch.nn.Embedding): Embedding layer for converting tokens to vector representations.
        blocks (torch.nn.ModuleList): List of `TransformationBlock` modules.
        Final_LayerNorm (torch.nn.LayerNorm): Final layer normalization applied before output.
        linear (torch.nn.Linear): Linear layer mapping the embedding space to the output vocabulary.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Processes the input tensor through embedding, transformation blocks, and a final
            linear layer to generate logits over the vocabulary.

    """

    def __init__(
        self,
        device: torch.device,
        config: ModelConfig,
    ):
        super().__init__()
        self.device = device
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
        self.linear = torch.nn.Linear(
            self.config["embedding_dim"], len(ALPHABET), device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.embedding(
            torch.arange(self.config["vector_len"]).to(self.device)
        )

        for block in self.blocks:
            x = block(x, self.config["vector_len"])

        x = self.Final_LayerNorm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


def prompt(
    device: torch.device, prompt: str, model: ShakeModel, target_len: int = 400
) -> str:
    iprompt = [STOI[c] for c in prompt]

    p_vec = torch.tensor([iprompt[-model.config["vector_len"] :]]).to(device)
    res: list[int] = iprompt[-model.config["vector_len"] :]

    for j in range(model.config["vector_len"], target_len):
        with torch.no_grad():
            letter = torch.argmax(model(p_vec))
        res += [letter.item()]
        p_vec = torch.tensor([res[j - model.config["vector_len"] + 1 :]]).to(device)

    res_str = reduce(str.__add__, [ITOS[i] for i in res])
    return res_str
