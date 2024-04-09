import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import torchaudio
import torchvision


class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
    
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))
        out = self.pool(F.relu(self.conv4(out)))
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

def build_resnet18():
    return torchvision.models.resnet18(in_channels=1, num_classes=1)

def build_resnet34():
    return torchvision.models.resnet34(in_channels=1, num_classes=1)

def build_resnet50():
    return torchvision.models.resnet50(in_channels=1, num_classes=1)

def build_resnet101():
    return torchvision.models.resnet101(in_channels=1, num_classes=1)

def build_densenet121():
    return torchvision.models.densenet121(in_channels=1, num_classes=1)

def build_densenet161():
    return torchvision.models.densenet161(in_channels=1, num_classes=1)

def build_densenet169():
    return torchvision.models.densenet169(in_channels=1, num_classes=1)

def build_densenet201():
    return torchvision.models.densenet201(in_channels=1, num_classes=1)

def build_convnext_tiny():
    return torchvision.models.convnext_tiny(in_channels=1, num_classes=1)

def build_convnext_small():
    return torchvision.models.convnext_small(in_channels=1, num_classes=1)

def build_convnext_base():
    return torchvision.models.convnext_base(in_channels=1, num_classes=1)

def build_convnext_large():
    return torchvision.models.convnext_large(in_channels=1, num_classes=1)

# model = CNN2D()
# model = torchaudio.models.Conformer(input_dim=80, num_heads=4, ffn_dim=128, num_layers=4, depthwise_conv_kernel_size=31)
# model = torchvision.models.resnet18(in_channels=1, num_classes=1)
# data = torch.rand(64, 1, 256, 256)
# torchinfo.summary(model, input_data=data)