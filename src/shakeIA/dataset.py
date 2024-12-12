#!Python3.12

import string
import torch
from torch.utils.data import IterableDataset
from collections.abc import Iterator

ALPHABET = string.ascii_letters + string.digits + string.punctuation + string.whitespace

STOI = {c: i for i, c in enumerate(ALPHABET)}
ITOS = {i: c for i, c in enumerate(ALPHABET)}


class CharacterDataset(IterableDataset):

    def __init__(self, data: str):
        """data: str
        The full dataset"""
        super().__init__()
        self.data = data
        self.idata_stream = [STOI[c] for c in data]

    def get_vocab_size(self) -> int:
        return len(ALPHABET)

    def __iter__(self) -> Iterator[torch.Tensor]:
        work_info = torch.utils.data.get_worker_info()
        if work_info is None:
            start = 0
            end = len(self.data) - 3
        else:
            per_worker = (len(self.data) - 3) // work_info.num_workers
            start = work_info.id * per_worker
            end = start + per_worker

        G = (
            torch.tensor(
                [
                    self.idata_stream[i],
                    self.idata_stream[i + 1],
                    self.idata_stream[i + 2],
                    self.idata_stream[i + 3],
                ],
                dtype=torch.int16
            )
            for i in range(start, end)
        )

        return G