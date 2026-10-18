from typing import Callable

from .base import BudgetSpec, RankAllocator
from .lagrangian import LagrangianRankAllocator
from .lagrangian_joint import JointTuckerJLAllocator
from .memory import mem_original, mem_tucker4d_compressed
from .tucker2_grid import Tucker2GridAllocator

# 4D Tucker rank indices: layer, head, token, feature.
_TUCKER2_MODES = (2, 3)
_TOKEN_MODES = (2,)

REGISTRY: dict[str, Callable[[], RankAllocator]] = {
    "lagrangian": LagrangianRankAllocator,
    "lagrangian_tucker2": lambda: LagrangianRankAllocator(active_modes=_TUCKER2_MODES),
    "lagrangian_token": lambda: LagrangianRankAllocator(active_modes=_TOKEN_MODES),
    "tucker2_grid": Tucker2GridAllocator,
    "tucker+jl_joint": lambda: JointTuckerJLAllocator(active_modes=_TUCKER2_MODES),
}

__all__ = [
    "BudgetSpec",
    "RankAllocator",
    "LagrangianRankAllocator",
    "JointTuckerJLAllocator",
    "Tucker2GridAllocator",
    "mem_original",
    "mem_tucker4d_compressed",
    "REGISTRY",
]
