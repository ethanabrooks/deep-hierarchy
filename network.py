from torch import nn as nn


def eval_activation(string):
    return eval(f"nn.{string}")


def eval_init(string):
    return eval(f"nn.init.{string}")


class Net(nn.Module):
    def __init__(self, activation, init, num_layers, kernel_size):
        super(Net, self).__init__()
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
        sizes = []
        indices = []
        for conv in self.conv[:-1]:
            sizes.append(x.size())
            x, i = conv(x)
            indices.append(i)

        x = self.conv[-1](x)
        for de_conv, un_pool, i, size in zip(
            self.de_conv, self.max_un_pool, reversed(indices), reversed(sizes)
        ):
            x = de_conv(x)
            x = un_pool(x, i, size)
        x = self.de_conv[-1](x)
        return x.squeeze(1)
