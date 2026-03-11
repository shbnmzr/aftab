import torch


class LayerNorm2d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.ln = torch.nn.LayerNorm(num_features, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)
