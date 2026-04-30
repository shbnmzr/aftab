import torch


class FusedLayerNorm2d(torch.nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2 * num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(2 * num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_total, H, W = x.shape
        C = C_total // 2
        x_reshaped = x.view(N, 2, C, H, W)
        mean = x_reshaped.mean(dim=2, keepdim=True)
        var = (x_reshaped - mean).pow(2).mean(dim=2, keepdim=True)
        x_norm = (x_reshaped - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, 2, C, 1, 1)
        bias = self.bias.view(1, 2, C, 1, 1)
        x_norm = x_norm * weight + bias
        return x_norm.view(N, C_total, H, W)
