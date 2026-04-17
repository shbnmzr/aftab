import torch


class RandomShiftModule(torch.nn.Module):
    def __init__(self, padding: int = 4):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        n, c, h, w = x.shape
        padded = torch.nn.functional.pad(
            x, (self.pad, self.pad, self.pad, self.pad), mode="replicate"
        )
        cropped = torch.empty_like(x)
        w_starts = torch.randint(0, 2 * self.pad + 1, (n,))
        h_starts = torch.randint(0, 2 * self.pad + 1, (n,))
        for i in range(n):
            cropped[i] = padded[
                i, :, h_starts[i] : h_starts[i] + h, w_starts[i] : w_starts[i] + w
            ]
        return cropped
