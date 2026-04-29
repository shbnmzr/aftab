import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from ..modules.Transformer import Transformer
from ..functions import positional_embedding_sincos_2d


class SimpleViTEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        embedding_dimension: int,
        depth: int,
        heads: int,
        mlp_dimension: int,
        channels: int,
        head_dimension: int,
        use_flash: bool = True,
    ):
        super().__init__()
        image_height, image_width = (image_size, image_size)
        patch_height, patch_width = (patch_size, patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = torch.nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b h w (p1 p2 c)", p1=patch_height, p2=patch_width
            ),
            torch.nn.LayerNorm(patch_dim),
            torch.nn.Linear(patch_dim, embedding_dimension),
            torch.nn.LayerNorm(embedding_dimension),
        )

        self.transformer = Transformer(
            embedding_dimension=embedding_dimension,
            depth=depth,
            heads=heads,
            head_dimension=head_dimension,
            mlp_dimension=mlp_dimension,
            use_flash=use_flash,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        *_, h, w, dtype = *images.shape, images.dtype
        x = self.to_patch_embedding(images)
        positional_embedding = positional_embedding_sincos_2d(x)
        x = rearrange(x, "b ... d -> b (...) d") + positional_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x
