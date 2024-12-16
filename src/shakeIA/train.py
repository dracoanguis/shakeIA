import torch
import json

from pathlib import Path

from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from shakeIA.model import ShakeModel, ModelConfig, default_config
from shakeIA.dataset import CharacterDataset

from functools import reduce

from tqdm import tqdm, trange

from typing import TypedDict, Optional
from typing_extensions import ReadOnly, NotRequired

MetaConfig = TypedDict(
    "MetaConfig",
    {
        "train_epoch": ReadOnly[int],
        "optimizer": ReadOnly[str],
        "learning_rate": ReadOnly[float],
        "batch_size": ReadOnly[int],
        "Note": NotRequired[str],
    },
)


def train_full(
    config: MetaConfig,
    data_set: CharacterDataset,
    model_config: ModelConfig,
    model_name: Optional[str] = None,
    save_folder: Optional[Path] = None,
    save_frequency: Optional[int] = None,  # epoch diviser at witch we save
) -> tuple[ShakeModel, list[float]]:

    if save_frequency <= 0:
        raise ValueError("Modulo opération needs at least a striclty positive integer")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loss_list: list[float] = list()
    criterion = nn.CrossEntropyLoss().to(device)
    lr = config["learning_rate"]

    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        data_set, config["batch_size"]
    )
    model = ShakeModel(device, model_config)

    match config["optimizer"]:
        case "adam":
            optimizer = optim.Adam(model.parameters(), lr)
        case _:
            raise ValueError("Non valid optimizer request")

    for epoch in trange(config["train_epoch"], desc="Training epochs", position=0):

        for input, target in tqdm(data_loader, leave=False, desc="batch", position=1):

            optimizer.zero_grad()

            output = model(input.to(device))
            loss = criterion(output, target.to(device).view(-1))
            loss.backward()

            optimizer.step()

            loss_list.append(loss.item())

        if (
            model_name is not None
            and save_frequency is not None
            and epoch % save_frequency == 0
        ):

            if save_folder is None:
                raise ValueError("if save precised, you need to precise a save folder")

            dump(model, save_folder, model_name, config, loss_list, epoch=epoch)

    if model_name is not None and save_folder is not None:

        dump(model, save_folder, model_name, config, loss_list, final=True)

    return model, loss_list


def dump(
    model: ShakeModel,
    folder_path: Path,
    model_name: str,
    meta_config: MetaConfig,
    loss_list: Optional[list[float]],
    final: bool = False,
    epoch: Optional[int] = None,
) -> None:
    if not final:
        model_name += str(epoch)
    model_dir_path = folder_path.joinpath(model_name)
    model_config_path = model_dir_path.joinpath("model_config.json")
    meta_config_path = model_dir_path.joinpath("meta_config.json")
    model_path = model_dir_path.joinpath("shake.model")
    loss_path = model_dir_path.joinpath("loss.dat")

    if not folder_path.exists():
        folder_path.mkdir()

    if not model_dir_path.exists():
        model_dir_path.mkdir()

    if not model_config_path.exists():
        model_config_path.touch()

    if not meta_config_path.exists():
        meta_config_path.touch()

    torch.save(model, model_path)

    with open(model_config_path, "r+") as f:
        json.dump(model.config, f)

    with open(meta_config_path, "r+") as f:
        json.dump(meta_config, f)

    if loss_list is not None:
        torch.save(loss_list, loss_path)


if __name__ == "__main__":

    with open(
        "dataset.txt",
    ) as f:
        cd = CharacterDataset(reduce(str.__add__, f.readlines()), 100)

    test_config: MetaConfig = {
        "train_epoch": 15,
        "optimizer": "adam",
        "learning_rate": 0.1,
        "batch_size": 100,
    }

    model_config: ModelConfig = {
        "vector_len": 100,
        "embedding_dim": 10,
        "forward_expansion": 6,
        "block_number": 1,
        "head_num": 2,
    }

    train_full(test_config, cd, model_config, "first", Path("./models/"), 1)
