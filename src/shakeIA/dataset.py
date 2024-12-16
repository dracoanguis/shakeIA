#!Python3.12

import string
import torch
from torch.utils.data import IterableDataset
from collections.abc import Iterator

ALPHABET = string.ascii_letters + string.digits + string.punctuation + string.whitespace

STOI = {c: i for i, c in enumerate(ALPHABET)}
ITOS = {i: c for i, c in enumerate(ALPHABET)}


class CharacterDataset(IterableDataset[torch.Tensor]):

    def __init__(self, data: str, vector_len: int = 4):
        """data: str
        The full dataset
        """
        super().__init__()
        self.data = data
        self.idata_stream = [STOI[c] for c in data]
        self.vector_len = vector_len

    def get_vocab_size(self) -> int:
        return len(ALPHABET)

    def __iter__(self) -> Iterator[torch.Tensor]:
        work_info = torch.utils.data.get_worker_info()
        if work_info is None:
            start = 0
            end = len(self.data) - self.vector_len
        else:
            per_worker = (len(self.data) - self.vector_len) // work_info.num_workers
            start = work_info.id * per_worker
            end = start + per_worker

        return (
            (
                torch.tensor(
                    [self.idata_stream[i + j] for j in range(self.vector_len)],
                ),
                torch.tensor(
                    [
                        self.idata_stream[i + self.vector_len],
                    ]
                ),
            )
            for i in range(start, end)
        )
