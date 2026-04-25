import torch


def random_shifts(
    *,
    observation: torch.Tensor,
    padding: int,
    height_shifts: torch.Tensor = None,
    width_shifts: torch.Tensor = None,
):
    if not observation.is_floating_point():
        observation = observation.float()

    batch_size, _, height, width = observation.shape
    if height != width:
        raise ValueError("DrQ-style random shifts expect square observations.")

    padded_size = height + 2 * padding
    observation = torch.nn.functional.pad(
        observation,
        (padding, padding, padding, padding),
        mode="replicate",
    )

    eps = 1.0 / padded_size
    arange = torch.linspace(
        -1.0 + eps,
        1.0 - eps,
        padded_size,
        device=observation.device,
        dtype=observation.dtype,
    )[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    if height_shifts is None or width_shifts is None:
        shift = torch.randint(
            0,
            2 * padding + 1,
            size=(batch_size, 1, 1, 2),
            device=observation.device,
            dtype=observation.dtype,
        )
    else:
        shift = torch.stack([width_shifts, height_shifts], dim=-1).to(
            device=observation.device,
            dtype=observation.dtype,
        )
        shift = shift.view(batch_size, 1, 1, 2)

    grid = base_grid + shift * (2.0 / padded_size)
    return torch.nn.functional.grid_sample(
        observation,
        grid,
        padding_mode="zeros",
        align_corners=False,
    )
