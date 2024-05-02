import torch
import torch.nn as nn
from torchinfo import summary


class EnhancedCNN(nn.Module):
    __name__ = "EnhancedCNN"

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=4,
            dilation=2,
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=8,
            dilation=4,
            bias=False,
        )
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=16,
            dilation=8,
            bias=False,
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.swish = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.swish(self.pool1(self.bn1(self.conv1(x))))
        out = self.swish(self.pool2(self.bn2(self.conv2(out))))
        out = self.swish(self.bn3(self.conv3(out)))
        out = self.swish(self.bn4(self.conv4(out)))

        out = self.flatten(self.gap(out))
        out = self.swish(self.fc1(out))
        out = self.swish(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out


def _test():
    model = EnhancedCNN()
    # model.load_state_dict(torch.load("trained_models/pickle/EnhancedCNN.pt"))
    data = torch.rand(1, 1, 256, 256)
    # summary(model, input_data=data)
    print(model.__name__)


# _test()
