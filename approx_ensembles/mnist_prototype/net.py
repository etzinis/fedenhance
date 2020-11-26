import torch.nn as nn


class MNISTnet(nn.Module):
    def __init__(self, n_intermediate_layers=4):
        super(MNISTnet, self).__init__()
        layers_list = [nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.ReLU()] +\
            [nn.Conv2d(32, 32, kernel_size=5, padding=2),
             nn.ReLU()] * n_intermediate_layers + \
            [nn.Conv2d(32, 1, kernel_size=5, padding=2)]
        self.convs = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.convs(x)

if __name__ == "__main__":
    model = MNISTnet(n_intermediate_layers=0)
    print(model)