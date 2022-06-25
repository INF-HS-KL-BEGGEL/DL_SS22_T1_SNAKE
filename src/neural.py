from torch import nn
from enum import Enum
import copy


class NetMode(Enum):
    TARGET = 1
    TRAINING = 2


class SnakeCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Picture Shape Error, Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Picture Shape Error, Expecting input width: 84, got: {w}")

        self.trainingNet = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
        #create a target Net to train against
        self.targetNet = copy.deepcopy(self.trainingNet)

        for p in self.targetNet.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == NetMode.TRAINING:
            return self.trainingNet(input)
        elif model == NetMode.TARGET:
            return self.targetNet(input)
