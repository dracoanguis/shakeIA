import torch

from torch import nn
from torch.utils.data import DataLoader
from functools import reduce
from operator import add
from shakeIA import *
from tqdm import tqdm
from .dataset import ALPHABET


def train(
    device: torch.device,
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module
) -> None:
    model.train()
    for input, target in tqdm(dataloader):
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = criterion(output.view(-1, len(ALPHABET), target.view(-1)))
        loss.backward()


if __name__ == '__main__':

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Working on device {device}")

    with open("../input.txt",) as f:
        cd = CharacterDataset(reduce(add,f.readlines()))

    model = ShakeModel(device)


    dl = DataLoader(cd, 50)

    train(device,model,dl,nn.CrossEntropyLoss().to(device))
