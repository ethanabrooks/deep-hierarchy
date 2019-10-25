from __future__ import print_function

import argparse
import csv
import itertools
import random
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from rl_utils import hierarchical_parse_args
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from four_rooms import FourRooms
from network import Net, eval_activation, eval_init

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def format_image(data, output, target):
    return torch.stack([data.sum(1)[0], target[0], output[0]], dim=0).unsqueeze(1)


def main(
    no_cuda: bool,
    seed: int,
    batch_size: int,
    lr: float,
    log_interval: int,
    save_interval: int,
    log_dir: Path,
    run_id: str,
    four_rooms_args: dict,
    network_args: dict,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    writer = SummaryWriter(str(log_dir))

    torch.manual_seed(seed)

    if use_cuda:
        nvidia_smi = subprocess.check_output(
            "nvidia-smi --format=csv --query-gpu=memory.free".split(),
            universal_newlines=True,
        )
        n_gpu = len(list(csv.reader(StringIO(nvidia_smi)))) - 1
        try:
            index = int(run_id[-1])
        except (ValueError, IndexError):
            index = random.randrange(0, n_gpu)
        print("Using GPU", index)
        device = torch.device("cuda", index=index % n_gpu)
    else:
        device = "cpu"

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        FourRooms(**four_rooms_args), batch_size=batch_size, **kwargs
    )
    network = Net(**network_args).to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    network.train()

    log_progress = None
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            log_progress = tqdm(total=log_interval, desc="next log")
            writer.add_scalar("train loss", loss.item(), global_step=i)
            img = format_image(data, output, target)
            writer.add_images("train image", img, dataformats="NCHW", global_step=i)
        if i % save_interval == 0:
            torch.save(network.state_dict(), str(Path(log_dir, "network.pt")))
        log_progress.update()


def cli():
    # Training settings
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate "
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed ")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--log-dir", default="/tmp/mnist", metavar="N", help="")
    parser.add_argument("--run-id", default="", metavar="N", help="")
    four_rooms_parser = parser.add_argument_group("four_rooms_args")
    four_rooms_parser.add_argument("--room-size", type=int, default=100)
    four_rooms_parser.add_argument("--distance", type=float, default=100, help="")
    four_rooms_parser.add_argument("--len-dataset", type=int, default=int(1e5), help="")
    network_parser = parser.add_argument_group("network_args")
    network_parser.add_argument("--activation", type=eval_activation, default=None)
    network_parser.add_argument("--init", type=eval_init, default=None)
    main(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    cli()
