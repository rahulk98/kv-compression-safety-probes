from .base import EncodedResidual, ResidualEncoder
from .jl_projection import JLProjectionResidual
from .jl_rotated import JLRotatedResidual
from .none import NoResidual

REGISTRY: dict[str, type[ResidualEncoder]] = {
    "none": NoResidual,
    "jl_rotated": JLRotatedResidual,
    # EXPERIMENTAL, and deliberately NOT reachable through CompressionConfig:
    # `config.py.__post_init__` accepts only "none" / "jl_rotated" and rejects
    # this id with an explanatory error. That asymmetry is intentional. The
    # project decided rename-not-adopt on 2026-07-17: the full-dimension
    # rotation encoder ships, and this true dimension-reducing projection is
    # retained as the negative-result artifact behind results.md 11.7 (it wins
    # only in the 2 bits/elem cell, and loses 24/24 paired cells at 3, 4 and 8
    # bits/elem). Registered so experiments can instantiate it directly. Do not
    # "fix" the config check by widening its tuple; see the matching comment
    # there and tests/test_residual_config_exclusion.py.
    "jl_projection": JLProjectionResidual,
}

__all__ = [
    "EncodedResidual",
    "ResidualEncoder",
    "NoResidual",
    "JLProjectionResidual",
    "JLRotatedResidual",
    "REGISTRY",
]
