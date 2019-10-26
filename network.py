from torch import nn as nn
import torch
import torch.nn.functional as F


def eval_activation(string):
    return eval(f"nn.{string}")


def eval_init(string):
    return eval(f"nn.init.{string}")


class ConvDeConvNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, hidden_size: int):
        super(ConvDeConvNet, self).__init__()

        # Size of z latent vector (i.e. size of generator input)
        nz = 100

        self.encoder = nn.Sequential(
            # input is (input_channels) x 64 x 64
            nn.Conv2d(input_channels, hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size) x 32 x 32
            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*2) x 16 x 16
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*4) x 8 x 8
            nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*8) x 4 x 4
            # nn.Conv2d(hidden_size * 8, 1, 4, 1, 0, bias=False),
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(nz, hidden_size * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(hidden_size * 8),
            # nn.ReLU(True),
            # state size. (hidden_size*8) x 4 x 4
            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            # state size. (hidden_size*4) x 8 x 8
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            # state size. (hidden_size*2) x 16 x 16
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            # state size. (hidden_size) x 32 x 32
            nn.ConvTranspose2d(hidden_size, output_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (output_channels) x 64 x 64
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z).squeeze(1)


class DeepHierarchicalNetwork(nn.Module):
    def __init__(
        self, arity: int, hidden_size: int, num_gru_layers: int, max_depth: int
    ):
        super().__init__()
        self.max_depth = max_depth
        self.arity = arity
        self.task_splitter = nn.GRU(hidden_size, hidden_size, num_layers=num_gru_layers)
        self.task_gru = nn.GRU(
            hidden_size, hidden_size, num_layers=num_gru_layers, bidirectional=True
        )
        self.logits = nn.Linear(hidden_size, 2)

    def decompose(self, task_matrix):
        for task in task_matrix:
            gru_input = task.unsqueeze(0).expand((self.arity, 1, 1))
            _, subtasks = self.task_splitter(gru_input)
            yield subtasks

    def forward(self, x):
        task = self.convolve(x)  # type:torch.Tensor
        assert isinstance(task, torch.Tensor)
        not_done = torch.ones(x.size(0), 1)

        for _ in range(self.max_depth):

            # done
            _, task_encoding = self.task_gru(task)
            one_hot = F.gumbel_softmax(self.logits(task_encoding), hard=True)
            _, not_done = torch.split(not_done * one_hot, 2, dim=-1)  # done stays done
            done = 1 - not_done

            # decompose
            subtasks = torch.cat(list(self.decompose(task)), dim=-1)
            task = done * task + not_done * subtasks

        output = torch.zeros(self.output_size)
        for g in task:
            output += self.deconvolve(g)
        return output
