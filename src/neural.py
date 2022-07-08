import torch
from torch import nn
from enum import Enum
import copy
from torchinfo import summary

class NetMode(Enum):
    TARGET = 'target'
    TRAINING = 'online'


class SnakeCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Picture Shape Error, Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Picture Shape Error, Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
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
        self.target= copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

        summary(
            self.target,
            (32, 4,84,84),
            dtypes=[torch.float],
            verbose=2,
            col_width=16,
            col_names=["input_size", "output_size", "num_params"],
            row_settings=["var_names"],
        )

    def forward(self, input, model):
        if model == NetMode.TRAINING:
            return self.online(input)
        elif model == NetMode.TARGET:
            return self.target(input)
