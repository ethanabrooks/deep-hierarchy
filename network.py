from torch import nn as nn
import torch
import torch.nn.functional as F


class DeConvNet(nn.Module):
    def __init__(self, hidden_size: int, num_embeddings: int):
        super(DeConvNet, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings + 1, embedding_dim=hidden_size
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(nz, hidden_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            # state size. (hidden_size*8) x 4 x 4
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 8, 4, 2, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 2, hidden_size * 1, 8, 4, 2, bias=False),
            nn.BatchNorm2d(hidden_size * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, 1, 16, 8, 4, bias=False),
            # nn.BatchNorm2d(hidden_size * 2),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(hidden_size, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (output_channels) x 64 x 64
        )

    def forward(self, x):
        z = self.embedding(x).view(x.size(0), -1, 1, 1)
        return self.decoder(z).squeeze(1)


class DeepHierarchicalNet(DeConvNet):
    def __init__(
        self,
        arity: int,
        hidden_size: int,
        num_gru_layers: int,
        max_depth: int,
        **kwargs,
    ):
        super().__init__(**kwargs, hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.max_depth = max_depth
        self.arity = arity
        self.task_splitter = nn.GRU(hidden_size, hidden_size, num_layers=num_gru_layers)
        self.task_gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.embedding2 = nn.Sequential(
            nn.ReLU(True), nn.Linear(4 * self.embedding.embedding_dim, hidden_size)
        )
        self.logits = nn.Sequential(nn.ReLU(True), nn.Linear(2 * hidden_size, 2))
        self.pre_decode = nn.Sequential(
            nn.ReLU(True), nn.Linear(hidden_size, 8 * hidden_size), nn.ReLU(True)
        )

    """

    def decompose(self, task_matrix):
        for task in task_matrix:
            gru_input = task.unsqueeze(0).expand((self.arity, -1, -1))
            subtasks, _ = self.task_splitter(gru_input)
            yield subtasks

    def forward(self, x):
        task = self.embedding(x).view(x.size(0), -1)  # type:torch.Tensor
        # task = self.embedding2(task).unsqueeze(0)
        """
