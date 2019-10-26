from torch import nn as nn
import torch
import torch.nn.functional as F


def eval_activation(string):
    return eval(f"nn.{string}")


def eval_init(string):
    return eval(f"nn.init.{string}")


class ConvDeConvNet(nn.Module):
    def __init__(self, activation, init, num_layers, kernel_size):
        super(ConvDeConvNet, self).__init__()
        activation = activation or nn.ReLU
        init = init or nn.init.xavier_uniform_
        assert num_layers % 2 == 0
        assert kernel_size % 2 == 1
        in_channels = 2
        out_channels = 16
        self.conv = nn.ModuleList()
        for i in range(num_layers // 2 - 1):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    activation(),
                    nn.MaxPool2d(kernel_size=kernel_size, return_indices=True),
                )
            )
            in_channels = out_channels
            out_channels *= 2
        self.conv.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                activation(),
            )
        )
        self.max_un_pool = nn.ModuleList()
        self.de_conv = nn.ModuleList()
        for i in range(num_layers // 2 - 1):
            in_channels = out_channels
            out_channels //= 2
            self.de_conv.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    activation(),
                )
            )
            self.max_un_pool.append(nn.MaxUnpool2d(kernel_size=kernel_size))
        self.de_conv.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                activation(),
            )
        )

        for module_list in [self.conv, self.de_conv]:
            for layer in module_list:
                for param in layer.parameters():
                    if len(param.shape) > 1:
                        init(param)

    def forward(self, x):
        x, indices, sizes = self.encode(x)
        return self.decode(x, indices, sizes)

    def decode(self, x, indices, sizes):
        # noinspection PyTypeChecker
        for de_conv, un_pool, i, size in zip(
            self.de_conv, self.max_un_pool, reversed(indices), reversed(sizes)
        ):
            x = de_conv(x)
            x = un_pool(x, i, size)
        x = self.de_conv[-1](x)
        return x.squeeze(1)

    def encode(self, x):
        sizes = []
        indices = []
        for conv in self.conv[:-1]:
            sizes.append(x.size())
            x, i = conv(x)
            indices.append(i)
        x = self.conv[-1](x)
        return x, indices, sizes


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
