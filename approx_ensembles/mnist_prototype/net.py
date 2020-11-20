import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTnet(nn.Module):
    def __init__(self, params):
        super(MNISTnet, self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 1, kernel_size=5, padding=2))


    def forward(self, x):
        return self.convs(x)
