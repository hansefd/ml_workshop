import torch.nn as nn


class VanillaNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        output = self.net(x)
        return output
