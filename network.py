from torch import nn as nn
import torch
import torch.nn.functional as F


class ConvDeConvNet(nn.Module):
    def __init__(self, hidden_size: int, num_embeddings: int):
        super(ConvDeConvNet, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings + 1, embedding_dim=hidden_size * 2
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(nz, hidden_size * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(hidden_size * 8),
            # nn.ReLU(True),
            # state size. (hidden_size*8) x 4 x 4
            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 8, 4, 2, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 8, 4, 2, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 8, 4, 2, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (output_channels) x 64 x 64
        )

    def forward(self, x):
        z = self.embedding(x).view(x.size(0), -1, 1, 1)
        return self.decoder(z).squeeze(1)


class DeepHierarchicalNetwork(ConvDeConvNet):
    def __init__(
        self,
        arity: int,
        hidden_size: int,
        num_gru_layers: int,
        max_depth: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        task = self.encode(x)  # type:torch.Tensor
        assert isinstance(task, torch.Tensor)
        not_done = torch.ones(x.size(0), 1)

        # TODO depth first
        for _ in range(self.max_depth):
            # TODO: cast goals back into goal-space

            # done
            _, task_encoding = self.task_gru(task)
            one_hot = F.gumbel_softmax(
                self.logits(task_encoding), hard=True
            )  # TODO allow asymmetry
            _, not_done = torch.split(not_done * one_hot, 2, dim=-1)  # done stays done

            # decompose
            done = 1 - not_done
            subtasks = torch.cat(list(self.decompose(task)), dim=-1)
            task = done * task + not_done * subtasks

        # combine outputs
        output = torch.zeros(self.output_size)
        # TODO: batch this
        for g in task:
            output += self.decode(g)
        return torch.sigmoid(output)  # TODO: other kinds of combination
