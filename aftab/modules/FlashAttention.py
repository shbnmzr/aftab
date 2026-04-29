import torch
from einops import rearrange
from .Attend import Attend


class FlashAttention(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int = 8,
        head_dimension: int = 64,
        use_flash: bool = True,
    ):
        super().__init__()
        inner_dimension = head_dimension * heads
        self.heads = heads
        self.scale = head_dimension**-0.5
        self.norm = torch.nn.LayerNorm(dimension)
        self.attend = Attend(use_flash=use_flash)
        self.to_qkv = torch.nn.Linear(dimension, inner_dimension * 3, bias=False)
        self.to_out = torch.nn.Linear(inner_dimension, dimension, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        out = self.attend(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
