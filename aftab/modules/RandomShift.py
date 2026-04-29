import torch


class RandomShift(torch.nn.Module):
    def __init__(self, *, padding: int):
        super().__init__()
        if padding < 0:
            raise ValueError("Expected `padding` to be non-negative.")

        self.padding = padding
        self._base_grid_key = None
        self.register_buffer("_base_grid", torch.empty(0), persistent=False)

    def _get_base_grid(self, x: torch.Tensor, h: int) -> torch.Tensor:
        key = (h, str(x.device), x.dtype)
        if self._base_grid_key == key:
            return self._base_grid

        eps = 1.0 / (h + 2 * self.padding)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.padding,
            device=x.device,
            dtype=x.dtype,
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2).unsqueeze(0)
        self._base_grid = base_grid
        self._base_grid_key = key
        return base_grid

    def _sample_shift(self, x: torch.Tensor, n: int, h: int) -> torch.Tensor:
        shift = torch.randint(
            0, 2 * self.padding + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.padding)
        return shift

    def _apply_shift(self, x: torch.Tensor, shift: torch.Tensor, h: int) -> torch.Tensor:
        padding = tuple([self.padding] * 4)
        x = torch.nn.functional.pad(x, padding, "replicate")
        base_grid = self._get_base_grid(x, h)
        grid = base_grid + shift
        return torch.nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )

    def _forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        assert h == w
        shift = self._sample_shift(x, n, h)
        return self._apply_shift(x, shift, h)

    def _forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        trajectory_length, n, c, h, w = x.size()
        assert h == w
        x = x.reshape(trajectory_length * n, c, h, w)
        shift = self._sample_shift(x, trajectory_length * n, h)
        x = self._apply_shift(x, shift, h)
        return x.reshape(trajectory_length, n, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self._forward_batch(x)
        if x.ndim == 5:
            return self._forward_sequence(x)
        raise ValueError("Expected a 4D batch or 5D trajectory tensor.")
