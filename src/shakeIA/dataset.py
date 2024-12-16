#!Python3.12

import string
import torch
from torch.utils.data import IterableDataset
from collections.abc import Iterator

ALPHABET = string.ascii_letters + string.digits + string.punctuation + string.whitespace

STOI = {c: i for i, c in enumerate(ALPHABET)}
ITOS = {i: c for i, c in enumerate(ALPHABET)}


class CharacterDataset(IterableDataset[torch.Tensor]):
    """
    A PyTorch IterableDataset for character-level modeling that processes an input string and 
    produces tensors of character indices.

    This dataset transforms a string into a stream of character indices using a predefined 
    vocabulary. It then generates training samples consisting of a fixed-length input vector 
    and a corresponding target character.

    Attributes:
        data (str): The input string to process.
        vector_len (int): The length of the input sequences (default is 4).
        idata_stream (List[int]): The numerical representation of the input string based on the 
                                  `STOI` vocabulary mapping.

    Methods:
        get_vocab_size() -> int:
            Returns the size of the vocabulary.

        __iter__() -> Iterator[torch.Tensor]:
            Creates an iterator that yields tuples of input and target tensors. Each input tensor 
            has a length of `vector_len`, and the target tensor contains the next character index 
            in the sequence.
    """
    def __init__(self, data: str, vector_len: int = 4):

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
