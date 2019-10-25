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
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from four_rooms import FourRooms
from network import Net

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def train(
    log_interval: int,
    network: Net,
    device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    writer: SummaryWriter,
    iteration: int,
):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            writer.add_scalar(
                "train loss",
                loss.item(),
                global_step=batch_idx
                # global_step=iteration * len(train_loader) + batch_idx,
            )
            img = format_image(data, output, target)
            writer.add_images(
                "train image",
                img,
                dataformats="NCHW",
                global_step=batch_idx
                # global_step=iteration * len(train_loader) + batch_idx,
            )


def format_image(data, output, target):
    return torch.stack([data.sum(1)[0], target[0], output[0]], dim=0).unsqueeze(1)


def test(
    network: Net, device, test_loader: DataLoader, writer: SummaryWriter, iteration: int
):
    network.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    writer.add_scalar("test loss", test_loss, global_step=iteration)
    writer.add_images(
        "test image",
        format_image(data, output, target),
        dataformats="NCHW",
        global_step=iteration,
    )

    print("\nTest set: Average loss: {:.4f}, \n".format(test_loss))


def main(
    no_cuda: bool,
    seed: int,
    batch_size: int,
    test_batch_size: int,
    lr: float,
    log_interval: int,
    log_dir: Path,
    run_id: str,
    four_rooms_args: dict,
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
        except ValueError:
            index = random.randrange(0, n_gpu)
        print("Using GPU", index)
        device = torch.device("cuda", index=index % n_gpu)
    else:
        device = "cpu"

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        FourRooms(**four_rooms_args, room_size=100), batch_size=batch_size, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        FourRooms(**four_rooms_args, room_size=100),
        batch_size=test_batch_size,
        **kwargs
    )

    network = Net().to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)

    for epoch in itertools.count():
        train(
            log_interval=log_interval,
            network=network,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
            iteration=epoch,
        )
        test(
            network=network,
            device=device,
            test_loader=test_loader,
            writer=writer,
            iteration=epoch,
        )

    torch.save(network.state_dict(), "mnist_cnn.pt")
    torch.save(network.state_dict(), str(Path(log_dir, "classifier.pt")))


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
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate "
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed ")
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
    # four_rooms_parser.add_argument("--room-size", type=int, default=100)
    four_rooms_parser.add_argument("--distance", type=float, default=100, help="")
    four_rooms_parser.add_argument("--len-dataset", type=int, default=int(1e5), help="")
    main(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    cli()
