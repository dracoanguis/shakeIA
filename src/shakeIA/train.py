import torch
import os
import json

from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from shakeIA import *

from functools import reduce
from operator import add

from tqdm import tqdm


def train(
    device: torch.device,
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    save_file: os.PathLike | None,
) -> None:

    model.train()
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = criterion(output, target.view(-1))
        loss.backward()

    if save_file:
        torch.save(model, save_file)


def dump(
    model: ShakeModel,
    folder_path: os.PathLike,
    model_name: str,
) -> None:

    model_dir_path = os.path.join(folder_path, model_name)
    config_path = os.path.join(model_dir_path, "config.json")
    model_path = os.path.join(model_dir_path, "shake.model")

    torch.save(model, model_path)

    with open(config_path) as f:
        json.dump(model.config, f)


if __name__ == "__main__":

    device = torch.device(
        "cpu"
    )  # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Working on device {device}")

    with open(
        "dataset.txt",
    ) as f:
        cd = CharacterDataset(reduce(add, f.readlines()))

    model = ShakeModel(device, embedding_dim=12, head_num=3)
    optimizer = optim.Adam(model.parameters())
    # loss =

    for epoch in tqdm(range(100)):

        dl = DataLoader(cd, 50, shuffle=True)

        train(
            device,
            model,
            dl,
            nn.CrossEntropyLoss().to(device),
            save_file=f"adam_ep{epoch}.model" if epoch % 10 else None,
        )
