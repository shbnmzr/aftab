import torch
from einops import rearrange
from .Attend import Attend


class FlashAttention(torch.nn.Module):
    def __init__(
        self, embedding_dimension: int, heads: int, head_dimension: int, use_flash: bool
    ):
        super().__init__()
        inner_dimension = head_dimension * heads
        self.heads = heads
        self.scale = head_dimension**-0.5
        self.attend = Attend(use_flash=use_flash)
        self.norm = torch.nn.LayerNorm(embedding_dimension)
        self.to_qkv = torch.nn.Linear(
            embedding_dimension, inner_dimension * 3, bias=False
        )
        self.to_out = torch.nn.Linear(inner_dimension, embedding_dimension, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        out = self.attend(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
