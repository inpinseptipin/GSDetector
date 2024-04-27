import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import torchaudio
import torchvision


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = self.flatten(self.gap(x))
        out = self.fc(x)
        
        return out
    

class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=4, dilation=2, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=8, dilation=4, bias=False)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=16, dilation=8, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        out = F.silu(self.pool1(self.bn1(self.conv1(x))))
        out = F.silu(self.pool2(self.bn2(self.conv2(out))))
        out = F.silu(self.bn3(self.conv3(out)))
        out = F.silu(self.bn4(self.conv4(out)))
        
        out = self.flatten(self.gap(out))
        out = F.silu(self.fc1(out))
        out = F.silu(self.fc2(out))
        out = self.fc3(out)
        
        return out


def build_resnet18():
    model = torchvision.models.resnet18(num_classes=1)
    og = model.conv1
    model.conv1 = nn.Conv2d(1, og.out_channels, kernel_size=og.kernel_size, stride=og.stride, padding=og.padding, bias=og.bias)
    return model

def build_resnet34():
    model = torchvision.models.resnet34(num_classes=1)
    og = model.conv1
    model.conv1 = nn.Conv2d(1, og.out_channels, kernel_size=og.kernel_size, stride=og.stride, padding=og.padding, bias=og.bias)
    return model

def build_resnet50():
    model = torchvision.models.resnet50(num_classes=1)
    og = model.conv1
    model.conv1 = nn.Conv2d(1, og.out_channels, kernel_size=og.kernel_size, stride=og.stride, padding=og.padding, bias=og.bias)
    return model

def build_resnet101():
    model = torchvision.models.resnet101(num_classes=1)
    og = model.conv1
    model.conv1 = nn.Conv2d(1, og.out_channels, kernel_size=og.kernel_size, stride=og.stride, padding=og.padding, bias=og.bias)
    return model


def _test():
    model = SimpleCNN()
    # model.load_state_dict(torch.load("trained_models/pickle/SimpleCNN.pt"))
    # model = EnhancedCNN()
    # model = build_resnet18()
    data = torch.rand(1, 1, 256, 256)
    # traced_model = torch.jit.trace(model, data)
    torchinfo.summary(model, input_data=data)

# _test()
