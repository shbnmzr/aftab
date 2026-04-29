import torch
from ..modules import LayerNorm2d
from ..constants import ModuleType


class ThriftyEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        filters: int,
        iterations: int,
        kernel_size: int,
        activation: ModuleType = torch.nn.ReLU,
    ):
        super().__init__()
        self.filters = filters
        self.iterations = iterations
        self.conv = torch.nn.Conv2d(
            filters, filters, kernel_size=kernel_size, padding=1, bias=False
        )
        self.activation = activation()
        self.normalizations = torch.nn.ModuleList(
            [LayerNorm2d(filters) for _ in range(iterations)]
        )
        self.pool_strategy = [iterations // 5 * i for i in range(1, 5)]
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = torch.nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.filters > x.size(1):
            x = torch.torch.nn.functional.pad(
                x, (0, 0, 0, 0, 0, self.filters - x.size(1))
            )

        for t in range(self.iterations):
            a_t = self.activation(self.conv(x))
            x = x + a_t
            x = self.normalizations[t](x)
            if t in self.pool_strategy:
                x = self.pool(x)

        x = self.global_pool(x)
        return x.flatten(1)
