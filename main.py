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
from network import DeConvNet, DeepHierarchicalNet

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
    baseline: bool,
    curriculum_threshold: float,
    four_rooms_args: dict,
    baseline_args: dict,
    deep_hierarchical_args: dict,
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
    dataset = FourRooms(**four_rooms_args, room_size=128)
    baseline_args.update(num_embeddings=dataset.size)
    if baseline:
        network = DeConvNet(**baseline_args)
    else:
        network = DeepHierarchicalNet(**deep_hierarchical_args, **baseline_args)
    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    network.train()
    start = 0

    for curriculum_level in itertools.count():
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, **kwargs
        )
        start = train(
            dataset=dataset,
            device=device,
            log_dir=log_dir,
            log_interval=log_interval,
            network=network,
            optimizer=optimizer,
            save_interval=save_interval,
            train_loader=train_loader,
            writer=writer,
            start=start,
            curriculum_threshold=curriculum_threshold,
            baseline=baseline,
        )
        dataset.increment_curriculum()
        network.increment_curriculum()
        writer.add_scalar("curriculum level", curriculum_level, global_step=start)


def train(
    dataset,
    device,
    log_dir,
    log_interval,
    network,
    optimizer,
    save_interval,
    train_loader,
    writer,
    start,
    curriculum_threshold,
    baseline,
):
    log_progress = None
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = raw_output = network(data)
        if not baseline:
            output = output.sum(0).sigmoid()
        loss = F.mse_loss(output, target, reduction="mean")
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            log_progress = tqdm(total=log_interval, desc="next log")
            array = dataset.empty_rooms.copy()
            data_img = torch.tensor(
                dataset.draw_points(data[0, :2].cpu(), data[0, 2:4].cpu(), array=array),
                device=device,
            )[: dataset.size, : dataset.size]
            img_output = raw_output[:, 0]
            if baseline:
                img_output.unsqueeze_(0)
            img = torch.cat(
                [data_img.unsqueeze(0), target[0].unsqueeze(0), img_output], dim=0
            ).unsqueeze(1)
            writer.add_images(
                "train image", img, dataformats="NCHW", global_step=i + start
            )
            writer.add_scalar("loss", loss, global_step=i + start)
        if i % save_interval == 0:
            torch.save(network.state_dict(), str(Path(log_dir, "network.pt")))
        log_progress.update()
        if loss < curriculum_threshold:
            return i


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
    parser.add_argument("--curriculum-threshold", type=float, default=0.01, help=" ")
    parser.add_argument("--log-dir", default="/tmp/mnist", metavar="N", help="")
    parser.add_argument("--run-id", default="", metavar="N", help="")
    parser.add_argument("--baseline", action="store_true")
    four_rooms_parser = parser.add_argument_group("four_rooms_args")
    # four_rooms_parser.add_argument("--room-size", type=int, default=128)
    four_rooms_parser.add_argument("--distance", type=float, default=100, help="")
    four_rooms_parser.add_argument("--len-dataset", type=int, default=int(1e5), help="")
    baseline_parser = parser.add_argument_group("baseline_args")
    baseline_parser.add_argument("--hidden-size", type=int, default=64)
    deep_hierarchical_parser = parser.add_argument_group("deep_hierarchical_args")
    deep_hierarchical_parser.add_argument("--arity", type=int, default=2)
    deep_hierarchical_parser.add_argument("--num-gru-layers", type=int, default=2)
    deep_hierarchical_parser.add_argument("--max-depth", type=int, default=5)
    main(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    cli()
