from __future__ import annotations

from typing import Literal

import torch

from .base import GroupingStrategy

AlignmentSource = Literal["k_only", "v_only", "joint"]


class AdaptiveGrouping(GroupingStrategy):
    def __init__(
        self,
        alignment_source: AlignmentSource = "k_only",
        threshold: float | None = None,
        k_top: int = 8,
        min_size: int = 2,
        max_size: int = 8,
    ):
        if alignment_source not in ("k_only", "v_only", "joint"):
            raise ValueError(f"unknown alignment_source: {alignment_source}")
        self.alignment_source = alignment_source
        self.threshold = threshold
        self.k_top = k_top
        self.min_size = min_size
        self.max_size = max_size

    def _token_mode_basis(self, X: torch.Tensor) -> torch.Tensor:
        # X: (n_h, T, d_h). Token mode is dim 1; mode-1 unfolding is T x (n_h*d_h).
        T = X.shape[1]
        unf = X.permute(1, 0, 2).reshape(T, -1).to(torch.float32)
        k = min(self.k_top, T, unf.shape[1])
        U, _, _ = torch.linalg.svd(unf, full_matrices=False)
        return U[:, :k]

    def _basis_for_layer(self, layer: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.alignment_source == "k_only":
            return self._token_mode_basis(layer["k"])
        if self.alignment_source == "v_only":
            return self._token_mode_basis(layer["v"])
        Uk = self._token_mode_basis(layer["k"])
        Uv = self._token_mode_basis(layer["v"])
        return torch.cat([Uk, Uv], dim=1)

    def assign(
        self,
        layers: dict[int, dict[str, torch.Tensor]],
    ) -> list[list[int]]:
        idxs = sorted(layers.keys())
        if len(idxs) <= 1:
            return [idxs]

        bases = [self._basis_for_layer(layers[i]) for i in idxs]
        # consecutive Grassmann-style alignments
        scores = []
        for a, b in zip(bases[:-1], bases[1:]):
            k = min(a.shape[1], b.shape[1])
            M = a[:, :k].T @ b[:, :k]
            scores.append((torch.linalg.norm(M) ** 2).item() / k)

        tau = self.threshold if self.threshold is not None else float(torch.tensor(scores).median())

        groups: list[list[int]] = [[idxs[0]]]
        for layer_pos in range(1, len(idxs)):
            cur = groups[-1]
            score = scores[layer_pos - 1]
            start_new = (score < tau and len(cur) >= self.min_size) or len(cur) >= self.max_size
            if start_new:
                groups.append([idxs[layer_pos]])
            else:
                cur.append(idxs[layer_pos])

        # merge any trailing group below min_size into the previous
        if len(groups) >= 2 and len(groups[-1]) < self.min_size:
            groups[-2].extend(groups[-1])
            groups.pop()
        return groups


def parse_grouping_id(
    spec: str | GroupingStrategy,
    *,
    alignment_source: str = "k_only",
) -> GroupingStrategy:
    if isinstance(spec, GroupingStrategy):
        return spec
    if spec == "adaptive":
        return AdaptiveGrouping(alignment_source=alignment_source)
    if spec.startswith("fixed:"):
        from .fixed import FixedSizeGrouping
        return FixedSizeGrouping(int(spec.split(":", 1)[1]))
    raise ValueError(f"unknown grouping spec: {spec}")
