import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary


def build_resnet18() -> nn.Module:
    model = models.resnet18(num_classes=1)
    model.__name__ = "ResNet18"
    og = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=og.out_channels,
        kernel_size=og.kernel_size,
        stride=og.stride,
        padding=og.padding,
        dilation=og.dilation,
        groups=og.groups,
        bias=og.bias,
        padding_mode=og.padding_mode,
    )
    return model


def build_resnet34():
    model = models.resnet34(num_classes=1)
    model.__name__ = "ResNet34"
    og = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=og.out_channels,
        kernel_size=og.kernel_size,
        stride=og.stride,
        padding=og.padding,
        dilation=og.dilation,
        groups=og.groups,
        bias=og.bias,
        padding_mode=og.padding_mode,
    )
    return model


def _test():
    model = build_resnet34()
    # model.load_state_dict(torch.load("trained_models/pickle/ResNet18.pt"))
    data = torch.rand(1, 1, 256, 256)
    summary(model, input_data=data)
    print(model.__name__)


# _test()
