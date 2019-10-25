from torch import nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolution 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=4, stride=1, padding=0
        )
        nn.init.xavier_uniform(self.conv1.weight)  # Xaviers Initialisation
        self.swish1 = nn.ReLU()

        # Max Pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        nn.init.xavier_uniform(self.conv2.weight)
        self.swish2 = nn.ReLU()

        # Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        # Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        nn.init.xavier_uniform(self.conv3.weight)
        self.swish3 = nn.ReLU()

        # De Convolution 1
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3
        )
        nn.init.xavier_uniform(self.deconv1.weight)
        self.swish4 = nn.ReLU()

        # Max UnPool 1
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2)

        # De Convolution 2
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=5
        )
        nn.init.xavier_uniform(self.deconv2.weight)
        self.swish5 = nn.ReLU()

        # Max UnPool 2
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2)

        # DeConvolution 3
        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4)
        nn.init.xavier_uniform(self.deconv3.weight)
        self.swish6 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.swish1(x)
        size1 = x.size()
        x, indices1 = self.maxpool1(x)
        x = self.conv2(x)
        x = self.swish2(x)
        size2 = x.size()
        x, indices2 = self.maxpool2(x)
        x = self.conv3(x)
        x = self.swish3(x)

        x = self.deconv1(x)
        x = self.swish4(x)
        x = self.maxunpool1(x, indices2, size2)
        x = self.deconv2(x)
        x = self.swish5(x)
        x = self.maxunpool2(x, indices1, size1)
        x = self.deconv3(x)
        return x.view(-1, 101, 101)
