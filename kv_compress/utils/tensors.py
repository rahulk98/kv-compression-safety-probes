from __future__ import annotations

import torch


def stack_layer_group(
    layers: dict[int, dict[str, torch.Tensor]],
    idxs: list[int],
    key: str,
) -> torch.Tensor:
    tensors = [layers[i][key] for i in idxs]
    stacked = torch.stack(tensors, dim=0)
    return stacked.to(torch.float32)
