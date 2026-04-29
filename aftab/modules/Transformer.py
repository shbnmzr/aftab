import torch
from .FlashAttention import FlashAttention
from .ViTStream import ViTStream


class Transformer(torch.nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        depth: int,
        heads: int,
        head_dimension: int,
        mlp_dimension: int,
        use_flash: bool,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        FlashAttention(
                            embedding_dimension,
                            heads=heads,
                            head_dimension=head_dimension,
                            use_flash=use_flash,
                        ),
                        ViTStream(
                            embedding_dimension=embedding_dimension,
                            mlp_dimension=mlp_dimension,
                        ),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention, feedforward in self.layers:
            x = attention(x) + x
            x = feedforward(x) + x
        return x
