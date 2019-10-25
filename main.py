from __future__ import print_function

import argparse
import csv
import itertools
import random
import subprocess
from StringIO import StringIO
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl_utils import hierarchical_parse_args
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from four_rooms import FourRooms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1, padding=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=5, stride=5, return_indices=True)
        self.conv2 = nn.Conv2d(20, 50, 5, 1, padding=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=5, stride=5, return_indices=True)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 4 * 4 * 50)
        self.de_conv1 = nn.ConvTranspose2d(50, 20, 5, 1, padding=2)
        self.max_un_pool1 = nn.MaxUnpool2d(kernel_size=5, stride=5)
        self.de_conv2 = nn.ConvTranspose2d(20, 1, 5, 1, padding=2)
        self.max_un_pool2 = nn.MaxUnpool2d(kernel_size=5, stride=5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        size1 = x.size()
        x, indices1 = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        size2 = x.size()
        x, indices2 = self.max_pool2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 50, 4, 4)
        x = self.max_un_pool1(x, indices2, size2)
        x = F.relu(self.de_conv1(x))
        x = self.max_un_pool2(x, indices1, size1)
        x = self.de_conv2(x)
        return x.view(-1, 101, 101)


def train(
    log_interval: int,
    network: Net,
    device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    writer: SummaryWriter,
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
            writer.add_scalar("train loss", loss.item())
            img = format_image(data, output, target)
            writer.add_images("train image", img, dataformats="NCHW")
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def format_image(data, output, target):
    return torch.stack([data.sum(1)[0], target[0], output[0]], dim=0).unsqueeze(1)


def test(network: Net, device, test_loader: DataLoader, writer: SummaryWriter):
    network.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    writer.add_scalar("test loss", test_loss)
    writer.add_images(
        "test image", format_image(data, output, target), dataformats="NCHW"
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
        device = torch.device("cuda", index=index % n_gpu)
    else:
        device = "cpu"

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        FourRooms(**four_rooms_args, room_size=100),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        FourRooms(**four_rooms_args, room_size=100),
        batch_size=test_batch_size,
        shuffle=True,
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
        )
        test(network=network, device=device, test_loader=test_loader, writer=writer)

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
        default=1000,
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
    main(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    cli()
