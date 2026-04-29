from .SimpleViTEncoder import SimpleViTEncoder


class IotaEncoder(SimpleViTEncoder):
    def __init__(self):
        super().__init__(
            patch_size=7,
            depth=4,
            heads=4,
            head_dimension=16,
            embedding_dimension=64,
            mlp_dimension=128,
        )
