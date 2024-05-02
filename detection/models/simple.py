import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchinfo import summary
from torchvision.transforms.v2 import Normalize


class SimpleCNN(nn.Module):
    __name__ = "SimpleCNN"

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.pool1(self.conv1(x)))
        x = self.relu(self.pool2(self.conv2(x)))
        x = self.sigmoid(self.fc(self.flatten(x)))

        return x


class SimplePipeline(nn.Module):
    __name__ = "SimplePipeline"
    n_ffts = 511
    hop_length = 345
    power = 2
    mean = -44.11405944824219
    std = 19.984331130981445

    def __init__(self):
        super().__init__()
        self.to_specgram = T.Spectrogram(
            n_fft=self.n_ffts, hop_length=self.hop_length, power=self.power
        )
        self.power_to_dB = T.AmplitudeToDB(stype="power")
        self.normalize = Normalize(mean=[self.mean], std=[self.std])

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.power_to_dB(self.to_specgram(x))
        x = self.normalize(x)

        x = self.relu(self.pool1(self.conv1(x)))
        x = self.relu(self.pool2(self.conv2(x)))
        x = self.sigmoid(self.fc(self.flatten(x)))

        return x


def _test():
    model = SimpleCNN()
    # model.load_state_dict(torch.load("trained_models/pickle/SimpleCNN.pt"))
    data = torch.rand(1, 1, 256, 256)
    # summary(model, input_data=data)
    print(model.__name__)


# _test()
