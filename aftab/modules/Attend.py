import torch
from packaging import version
from ..constants import FlashAttentionConfig


class Attend(torch.nn.Module):
    def __init__(self, use_flash=False):
        super().__init__()
        self.use_flash = use_flash
        assert not (
            use_flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        config = self.cuda_config if q.is_cuda else self.cpu_config
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        n, device, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5
        if self.use_flash:
            return self.flash_attn(q, k, v)
        similarity = torch.einsum("b h i d, b j d -> b h i j", q, k) * scale
        attention = similarity.softmax(dim=-1)
        return torch.einsum("b h i j, b j d -> b h i d", attention, v)
