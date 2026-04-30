import torch
from ..constants import ModuleType
from ..modules import FusedLayerNorm2d


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

        self.fused = torch.nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.normalization = FusedLayerNorm2d(out_channels)

        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        self.same_acts = chi == psi
        if self.same_acts:
            self.act = chi()
        else:
            self.chi = chi()
            self.psi = psi()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fused(x)
        x = self.normalization(x)

        if self.same_acts:
            x = self.act(x)
            a, b = torch.chunk(x, 2, dim=1)
        else:
            a, b = torch.chunk(x, 2, dim=1)
            a = self.psi(a)
            b = self.chi(b)

        x = a * b

        return torch.nn.functional.max_pool2d(
            x,
            kernel_size=self.pool_kernel,
            stride=self.pool_stride,
            padding=self.pool_padding,
        )
