from torch import nn as nn
import torch
import torch.nn.functional as F
import gc


class DeConvNet(nn.Module):
    def __init__(self, hidden_size: int, num_embeddings: int):
        super(DeConvNet, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings + 1, embedding_dim=hidden_size
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(nz, hidden_size * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(hidden_size * 8),
            # nn.ReLU(True),
            # state size. (hidden_size*8) x 4 x 4
            # nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 8, 4, 2, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 16, 8, 4, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 2, 1, 32, 16, 8, bias=False),
            # nn.BatchNorm2d(hidden_size * 2),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(hidden_size, 1, 4, 2, 1, bias=False),
            # state size. (output_channels) x 64 x 64
        )

    def forward(self, x):
        z = self.embedding(x).view(x.size(0), -1, 1, 1)
        return self.decoder(z).squeeze(1).sigmoid()


class DeepHierarchicalNet(DeConvNet):
    def __init__(
        self,
        arity: int,
        hidden_size: int,
        num_gru_layers: int,
        num_embeddings: int,
        **kwargs,
    ):
        super().__init__(
            **kwargs, hidden_size=hidden_size, num_embeddings=num_embeddings
        )
        self.de_conv_parameters = list(super().parameters())
        self.hidden_size = hidden_size
        self.tree_depth = 0
        self.arity = arity
        self.task_splitter = nn.GRU(hidden_size, hidden_size, num_layers=num_gru_layers)
        self.task_gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings + 1, embedding_dim=hidden_size // 4
        )
        gc.collect()
        self.logits = nn.Sequential(nn.ReLU(True), nn.Linear(2 * hidden_size, 2))
        self.pre_decode = nn.Sequential(
            nn.ReLU(True), nn.Linear(hidden_size, 4 * hidden_size), nn.ReLU(True)
        )
        self.hierarchy_modules = nn.ModuleList(
            [
                self.task_splitter,
                self.task_gru,
                self.embedding,
                self.logits,
                self.pre_decode,
            ]
        )

    def decompose(self, task_matrix):
        for task in task_matrix:
            gru_input = task[None].expand((self.arity, -1, -1))
            subtasks, _ = self.task_splitter(gru_input)
            yield subtasks

    def forward(self, x):
        task = self.embedding(x).view(1, x.size(0), -1)  # type:torch.Tensor
        assert isinstance(task, torch.Tensor)
        not_done = torch.ones(x.size(0), 1, device=x.device)

        # TODO depth first
        for i in range(self.tree_depth):
            # TODO: cast goals back into goal-space

            # done
            # _, task_encoding = self.task_gru(task)
            # logits_input = task_encoding.transpose(0, 1).reshape(x.size(0), -1)
            # one_hot = F.gumbel_softmax(self.logits(logits_input), hard=True)
            # # TODO allow asymmetric tree
            # _, not_done = torch.split(not_done * one_hot, 1, dim=-1)  # done stays done

            # decompose
            # done = 1 - not_done
            subtasks = F.relu(torch.cat(list(self.decompose(task)), dim=0))
            task = subtasks
            # relu ensures that padding is unique
            # task = F.pad(
            #     task, [0, 0, 0, 0, 0, subtasks.size(0) - task.size(0)], value=-1
            # )
            # task = done * task + not_done * subtasks

        # combine outputs
        # mask = (task != -1).any(dim=-1)
        # decoder_input = self.pre_decode(task[mask]).unsqueeze(-1).unsqueeze(-1)
        if task.size(0) > 1:
            aux_loss = -torch.norm(task[None] - task[:, None], dim=3).mean()
        else:
            aux_loss = 0
        output = self.decode(task)
        return output, aux_loss

    def decode(self, task):
        decoder_input = self.pre_decode(task.view(-1, self.hidden_size))
        decoded = self.decoder(decoder_input[:, :, None, None]).squeeze(1)
        # padded = nn.utils.rnn.pad_sequence(torch.split(decoded, tuple(mask.sum(0))))
        # return padded.sum(0).sigmoid()  # TODO: other kinds of combination
        output = decoded.view(
            task.size(0), task.size(1), decoded.size(1), decoded.size(2)
        ).sigmoid()
        return output

    def increment_curriculum(self):
        self.tree_depth += 1

    def level0_parameters(self):
        return iter(self.de_conv_parameters)

    def parameters(self, **kwargs):
        return self.hierarchy_modules.parameters(**kwargs)
