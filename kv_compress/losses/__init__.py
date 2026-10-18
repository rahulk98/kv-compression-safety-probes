from .base import LossFormulation
from .factor_space import FactorSpaceLoss

REGISTRY: dict[str, type[LossFormulation]] = {
    "factor_space": FactorSpaceLoss,
}

__all__ = ["LossFormulation", "FactorSpaceLoss", "REGISTRY"]
