import torch
from . import LayerNorm2d
from ..constants import ModuleType


class HadamaxBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        pool_kernel: int,
        pool_stride: int,
        pool_padding: int = 0,
        chi: ModuleType = torch.nn.GELU,
        psi: ModuleType = torch.nn.GELU,
    ):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv_b = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm_a = LayerNorm2d(out_channels)
        self.norm_b = LayerNorm2d(out_channels)

        self.chi = chi()
        self.psi = psi()

        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        a = self.psi(self.norm_a(self.conv_a(x)))
        b = self.chi(self.norm_b(self.conv_b(x)))

        x = a * b

        return torch.nn.functional.max_pool2d(
            x,
            kernel_size=self.pool_kernel,
            stride=self.pool_stride,
            padding=self.pool_padding,
        )
