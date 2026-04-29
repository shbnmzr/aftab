from .SimpleViTEncoder import SimpleViTEncoder


class IotaEncoder(SimpleViTEncoder):
    def __init__(self):
        super().__init__(
            patch_size=7,
            dimension=64,
            depth=4,
            heads=4,
            head_dimension=64,
            mlp_dimension=64,
        )
