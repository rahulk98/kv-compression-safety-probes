from .base import Approximation, Decomposition
from .tucker2_partial import Tucker2Partial
from .tucker2_partial_lowrank import Tucker2PartialMode2LowRank, resolve_q_cap
from .tucker4d import Tucker4D, Tucker4DApproximation

REGISTRY: dict[str, type[Decomposition]] = {
    "tucker4d": Tucker4D,
    "tucker2_partial": Tucker2Partial,
    "tucker2_partial_lowrank": Tucker2PartialMode2LowRank,
}

__all__ = [
    "Approximation",
    "Decomposition",
    "Tucker4D",
    "Tucker4DApproximation",
    "Tucker2Partial",
    "Tucker2PartialMode2LowRank",
    "resolve_q_cap",
    "REGISTRY",
]
