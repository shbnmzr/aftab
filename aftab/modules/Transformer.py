import torch
from .FlashAttention import FlashAttention
from .ViTStream import ViTStream


class Transformer(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        depth: int,
        heads: int,
        head_dimension: int,
        mlp_dimension: int,
        use_flash: bool = True,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        FlashAttention(
                            dimension,
                            heads=heads,
                            head_dimension=head_dimension,
                            use_flash=use_flash,
                        ),
                        ViTStream(dimension, mlp_dimension),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
