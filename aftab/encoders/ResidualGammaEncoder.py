import torch
from ..common import LayerNorm2d, ModuleType


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        conv1: torch.nn.Module,
        norm1: torch.nn.Module,
        act1: torch.nn.Module,
        conv2: torch.nn.Module,
        norm2: torch.nn.Module,
    ):
        super().__init__()
        self.conv1 = conv1
        self.norm1 = norm1
        self.act1 = act1
        self.conv2 = conv2
        self.norm2 = norm2

        if in_channels != out_channels or stride != 1:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                LayerNorm2d(out_channels),
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if identity.shape[2:] != out.shape[2:]:
            diff_h = identity.shape[2] - out.shape[2]
            diff_w = identity.shape[3] - out.shape[3]
            top = diff_h // 2
            bottom = identity.shape[2] - (diff_h - top)
            left = diff_w // 2
            right = identity.shape[3] - (diff_w - left)
            identity = identity[:, :, top:bottom, left:right]

        out += identity
        return out


class GammaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.alef = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(32),
            activation(),
        )

        self.be = ResBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
            conv1=torch.nn.Conv2d(
                32, 48, kernel_size=3, stride=2, padding=1, bias=False
            ),
            norm1=LayerNorm2d(48),
            act1=activation(),
            conv2=torch.nn.Conv2d(
                48, 64, kernel_size=3, stride=1, padding=0, bias=False
            ),
            norm2=LayerNorm2d(64),
        )
        self.be_activation = activation()

        self.pe = ResBlock(
            in_channels=64,
            out_channels=64,
            stride=2,
            conv1=torch.nn.Conv2d(
                64, 64, kernel_size=3, stride=2, padding=0, bias=False
            ),
            norm1=LayerNorm2d(64),
            act1=activation(),
            conv2=torch.nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=0, bias=False
            ),
            norm2=LayerNorm2d(64),
        )
        self.pe_activation = activation()

    def forward(self, x):
        x = self.alef(x)
        x = self.be(x)
        x = self.be_activation(x)
        x = self.pe(x)
        x = self.pe_activation(x)
        return torch.flatten(x, 1)
