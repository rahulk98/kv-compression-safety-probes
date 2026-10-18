from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import CompressionConfig


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


@dataclass
class CompressionResult:
    compression_ratio: float
    original_bytes: int
    original_bytes_k: int
    original_bytes_v: int
    compressed_bytes_k: int
    compressed_bytes_v: int
    core_bytes: int
    factor_bytes: int
    residual_bytes: int
    error_k: float
    error_v: float
    groups: list[list[int]]
    ranks_k: dict[int, tuple[int, ...]]
    ranks_v: dict[int, tuple[int, ...]]
    config: "CompressionConfig"
    k_target_used: str

    @property
    def compressed_bytes(self) -> int:
        return self.core_bytes + self.factor_bytes + self.residual_bytes

    @property
    def ratio_k(self) -> float:
        return self.original_bytes_k / self.compressed_bytes_k if self.compressed_bytes_k > 0 else float("inf")

    @property
    def ratio_v(self) -> float:
        return self.original_bytes_v / self.compressed_bytes_v if self.compressed_bytes_v > 0 else float("inf")

    def _format_ranks(self, label: str, ranks: dict[int, tuple[int, ...]]) -> list[str]:
        out = [f"Per-group ranks ({label}):"]
        for g, r in ranks.items():
            out.append(f"  group {g}: {tuple(r)}")
        return out

    def summary(self) -> str:
        cfg = self.config
        lines = [
            "=" * 72,
            "KV Compression Result",
            "=" * 72,
            f"Compression: {self.compression_ratio:.2f}x  "
            f"({_fmt_bytes(self.original_bytes)} -> {_fmt_bytes(self.compressed_bytes)})",
            f"  K: {self.ratio_k:.2f}x  ({_fmt_bytes(self.original_bytes_k)} -> {_fmt_bytes(self.compressed_bytes_k)})",
            f"  V: {self.ratio_v:.2f}x  ({_fmt_bytes(self.original_bytes_v)} -> {_fmt_bytes(self.compressed_bytes_v)})",
            f"  core:     {_fmt_bytes(self.core_bytes)}",
            f"  factors:  {_fmt_bytes(self.factor_bytes)}",
            f"  residual: {_fmt_bytes(self.residual_bytes)}",
            f"Recon error (relative Frobenius): K={self.error_k:.4f}  V={self.error_v:.4f}",
            f"K target compressed: {self.k_target_used}",
            f"Groups ({len(self.groups)}): "
            + ", ".join("[" + ",".join(str(i) for i in g) + "]" for g in self.groups),
            f"Grouping: {cfg.grouping}   Allocation: {cfg.allocation}   Decomposition: {cfg.decomposition}",
            f"Loss: {cfg.loss}   Residual: {cfg.residual} "
            f"(policy={cfg.residual_bits_policy}, f={cfg.residual_byte_fraction:.2f})",
        ]
        lines.extend(self._format_ranks("K", self.ranks_k))
        lines.extend(self._format_ranks("V", self.ranks_v))
        lines.extend([
            "-" * 72,
            "NOTE: ERRORS ARE OFFLINE FROBENIUS, NOT PERPLEXITY-PREDICTIVE.",
            "  error_k/error_v measure ||X - X_hat||_F / ||X||_F on the cached tensors.",
            "  They are what the factor-space loss (L1) optimizes; they do NOT measure",
            "  attention-output error, KL between attention distributions, or downstream",
            "  task accuracy. A low Frobenius error does not yet imply low perplexity hit.",
            "=" * 72,
        ])
        return "\n".join(lines)
