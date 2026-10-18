from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import CompressionConfig
from .reporting.metrics import CompressionResult


_CALIBRATION_SCHEMA = 1

# Above this ratio, uniform residual placement exceeds the Tucker reservation.
AUTO_RESIDUAL_RATIO_THRESHOLD = 3.76


def _layer_fingerprint(layers_kv: dict[int, dict[str, torch.Tensor]]) -> dict:
    idx0 = next(iter(layers_kv))
    k0 = layers_kv[idx0]["k"]
    v0 = layers_kv[idx0]["v"]
    return {
        "num_layers": len(layers_kv),
        "k_shape": list(k0.shape),
        "v_shape": list(v0.shape),
    }


def _load_calibration(path: Path, fingerprint: dict, cfg: CompressionConfig) -> list[list[int]]:
    with path.open() as f:
        blob = json.load(f)
    if blob.get("schema") != _CALIBRATION_SCHEMA:
        raise ValueError(f"calibration schema mismatch in {path}: expected {_CALIBRATION_SCHEMA}, got {blob.get('schema')}")
    if blob["fingerprint"] != fingerprint:
        raise ValueError(
            f"calibration fingerprint mismatch in {path}: cached {blob['fingerprint']}, current {fingerprint}"
        )
    if blob["strategy"] != cfg.grouping or blob["alignment_source"] != cfg.alignment_source:
        raise ValueError(
            f"calibration strategy mismatch in {path}: cached "
            f"({blob['strategy']}, {blob['alignment_source']}), current ({cfg.grouping}, {cfg.alignment_source})"
        )
    return [list(g) for g in blob["groups"]]


def _save_calibration(
    path: Path,
    fingerprint: dict,
    cfg: CompressionConfig,
    groups: list[list[int]],
) -> None:
    blob = {
        "schema": _CALIBRATION_SCHEMA,
        "model_name": cfg.model_name,
        "fingerprint": fingerprint,
        "strategy": cfg.grouping,
        "alignment_source": cfg.alignment_source,
        "groups": [list(g) for g in groups],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(blob, f, indent=2)


def _resolve_k_target(layers: dict[int, dict[str, torch.Tensor]], k_target: str) -> str:
    if k_target in ("pre_rope", "post_rope"):
        return k_target
    if k_target == "auto":
        all_have_post = all(layers[i].get("k_post_rope") is not None for i in layers)
        return "post_rope" if all_have_post else "pre_rope"
    if k_target == "both":
        return "both"
    raise ValueError(f"unknown k_target: {k_target}")


def _normalize_layers(
    layers: dict[int, dict[str, torch.Tensor]],
    k_target_resolved: str,
) -> dict[int, dict[str, torch.Tensor]]:
    out: dict[int, dict[str, torch.Tensor]] = {}
    k_key = "k_post_rope" if k_target_resolved == "post_rope" else "k_pre_rope"
    for idx, t in layers.items():
        k = t[k_key]
        if k is None:
            raise ValueError(f"layer {idx}: requested {k_key} is None")
        v = t["v"]
        out[idx] = {
            "k": k.detach().to(torch.float32),
            "v": v.detach().to(torch.float32),
        }
    return out



def _stack_groups(
    layers_kv: dict[int, dict[str, torch.Tensor]],
    groups: list[list[int]],
    kind: str,
):
    from .utils.tensors import stack_layer_group
    return [stack_layer_group(layers_kv, g, kind) for g in groups]


def _make_allocator(name: str):
    from .allocation import REGISTRY as ALLOC_REG  # type: ignore
    if name not in ALLOC_REG:
        raise ValueError(f"unknown allocation '{name}'; available: {list(ALLOC_REG)}")
    return ALLOC_REG[name]()


def _make_decomposition(name: str):
    from .decompositions import REGISTRY as DEC_REG
    if name not in DEC_REG:
        raise ValueError(f"unknown decomposition '{name}'; available: {list(DEC_REG)}")
    return DEC_REG[name]()


def _select_residual_groups(
    groups: list[list[int]],
    per_kind_orig: dict[str, dict[int, torch.Tensor]],
    per_kind_recon: dict[str, dict[int, torch.Tensor]],
    residual_enc,
    byte_budget: int,
) -> set[int]:
    """Greedy per-group bits: select the groups with the highest residual energy
    per byte until the byte budget is exhausted.

    Energy is the sum of squared Frobenius norms of E = X - X_hat across all
    layers in the group, summed across K and V. Cost is the encoder's expected
    footprint at the same K and V shapes. Cheap: one pass per group, no SVDs.
    """
    candidates: list[tuple[float, int, int]] = []
    for g, group_idxs in enumerate(groups):
        energy = 0.0
        cost = 0
        for kind in ("k", "v"):
            for idx in group_idxs:
                X = per_kind_orig[kind][idx]
                Xh = per_kind_recon[kind][idx]
                diff = (X - Xh).to(torch.float32)
                energy += float(torch.sum(diff * diff).item())
                cost += residual_enc.expected_footprint_bytes(tuple(X.shape))
        if cost > 0:
            candidates.append((energy / cost, g, cost))
    candidates.sort(reverse=True)

    selected: set[int] = set()
    spent = 0
    for _epb, g, cost in candidates:
        if spent + cost > byte_budget:
            continue
        selected.add(g)
        spent += cost
    return selected


def _make_residual_encoder(cfg: CompressionConfig):
    if cfg.residual == "none":
        return None
    from .residuals import REGISTRY as RES_REG  # type: ignore
    if cfg.residual not in RES_REG:
        raise ValueError(f"unknown residual '{cfg.residual}'; available: {list(RES_REG)}")
    cls = RES_REG[cfg.residual]
    return cls(seed=cfg.residual_seed)


def _compress_one(
    layers: dict[int, dict[str, torch.Tensor]],
    cfg: CompressionConfig,
    k_target_resolved: str,
) -> CompressionResult:
    from .allocation.base import BudgetSpec
    from .allocation.memory import mem_original, mem_tucker4d_breakdown
    from .grouping import parse_grouping_id

    layers_kv = _normalize_layers(layers, k_target_resolved)

    def kind_original_bytes(kind: str) -> int:
        total = 0
        for idx in layers_kv:
            dims = tuple(layers_kv[idx][kind].shape)
            total += mem_original(dims, dtype_bytes=cfg.dtype_bytes)
        return total

    orig_k = kind_original_bytes("k")
    orig_v = kind_original_bytes("v")
    original_bytes = orig_k + orig_v

    fingerprint = _layer_fingerprint(layers_kv)
    cache_path = Path(cfg.calibration_cache) if cfg.calibration_cache else None
    if cache_path is not None and cache_path.exists():
        groups = _load_calibration(cache_path, fingerprint, cfg)
    else:
        grouper = parse_grouping_id(cfg.grouping, alignment_source=cfg.alignment_source)
        groups = grouper.assign(layers_kv)
        if cache_path is not None:
            _save_calibration(cache_path, fingerprint, cfg, groups)

    decomp = _make_decomposition(cfg.decomposition)
    allocator = _make_allocator(cfg.allocation)
    residual_enc = _make_residual_encoder(cfg)
    joint_alloc = cfg.allocation == "tucker+jl_joint"

    ranks_k: dict[int, tuple[int, ...]] = {}
    ranks_v: dict[int, tuple[int, ...]] = {}
    core_bytes_total = 0
    factor_bytes_total = 0
    residual_bytes_total = 0
    compressed_bytes_k = 0
    compressed_bytes_v = 0
    per_kind_recon: dict[str, dict[int, torch.Tensor]] = {"k": {}, "v": {}}
    per_kind_orig: dict[str, dict[int, torch.Tensor]] = {"k": {}, "v": {}}

    stacked_k = _stack_groups(layers_kv, groups, "k")
    stacked_v = _stack_groups(layers_kv, groups, "v")

    mode = cfg.budget_mode()
    if mode == "ratio":
        bvalue = float(cfg.target_compression_ratio)
    elif mode == "bytes":
        bvalue = float(cfg.memory_budget_bytes)
    else:
        bvalue = float(cfg.max_reconstruction_error)

    # Reserve residual space unless the joint allocator prices it directly.
    policy = cfg.residual_bits_policy
    adaptive_residual = False
    if not joint_alloc:
        if policy == "auto":
            if mode == "ratio":
                implied_ratio = bvalue
            elif mode == "bytes":
                implied_ratio = original_bytes / max(bvalue, 1.0)
            else:
                implied_ratio = 0.0
            policy = "adaptive" if implied_ratio >= AUTO_RESIDUAL_RATIO_THRESHOLD else "fixed"
        adaptive_residual = residual_enc is not None and policy == "adaptive"
        if residual_enc is not None and mode in ("ratio", "bytes"):
            if adaptive_residual:
                expected_residual = int(cfg.residual_byte_fraction * original_bytes)
            else:
                expected_residual = sum(
                    residual_enc.expected_footprint_bytes(tuple(layers_kv[idx]["k"].shape))
                    + residual_enc.expected_footprint_bytes(tuple(layers_kv[idx]["v"].shape))
                    for idx in layers_kv
                )
            total_target = original_bytes / bvalue if mode == "ratio" else bvalue
            tucker_bytes = max(total_target - expected_residual, 1.0)
            mode = "bytes"
            bvalue = tucker_bytes

    budget = BudgetSpec(mode=mode, value=bvalue, dtype_bytes=cfg.dtype_bytes)

    if joint_alloc:
        allocated_ranks_k, allocated_ranks_v, joint_bits_k, joint_bits_v = (
            allocator.allocate_joint_with_bits(stacked_k, stacked_v, budget, decomp)
        )
    else:
        allocated_ranks_k, allocated_ranks_v = allocator.allocate_joint(
            stacked_k, stacked_v, budget, decomp
        )
        joint_bits_k = joint_bits_v = None

    for kind, stacked, allocated_ranks, ranks_out in (
        ("k", stacked_k, allocated_ranks_k, ranks_k),
        ("v", stacked_v, allocated_ranks_v, ranks_v),
    ):
        for g_pos, (group_idxs, X, ranks) in enumerate(zip(groups, stacked, allocated_ranks)):
            approx = decomp.fit(X, tuple(ranks))
            X_hat = approx.reconstruct()
            core_b, factor_b = mem_tucker4d_breakdown(tuple(ranks), tuple(X.shape), cfg.dtype_bytes)
            core_bytes_total += core_b
            factor_bytes_total += factor_b
            if kind == "k":
                compressed_bytes_k += core_b + factor_b
            else:
                compressed_bytes_v += core_b + factor_b
            ranks_out[g_pos] = tuple(int(r) for r in ranks)
            for slot, layer_idx in enumerate(group_idxs):
                per_kind_orig[kind][layer_idx] = X[slot]
                per_kind_recon[kind][layer_idx] = X_hat[slot]

    selected_groups: set[int] | None = None
    if residual_enc is not None and adaptive_residual:
        selected_groups = _select_residual_groups(
            groups,
            per_kind_orig,
            per_kind_recon,
            residual_enc,
            int(cfg.residual_byte_fraction * original_bytes),
        )

    if joint_alloc:
        # Reuse one encoder per bit width across layers and tensor kinds.
        from .residuals.jl_rotated import JLRotatedResidual
        joint_enc_cache: dict[int, JLRotatedResidual] = {}

        def _joint_encoder(bits: int) -> JLRotatedResidual:
            enc = joint_enc_cache.get(bits)
            if enc is None:
                enc = JLRotatedResidual(bits=bits, seed=cfg.residual_seed)
                joint_enc_cache[bits] = enc
            return enc

        layer_to_group = {
            int(idx): g for g, group_idxs in enumerate(groups) for idx in group_idxs
        }
        for kind, bits_list in (("k", joint_bits_k), ("v", joint_bits_v)):
            assert bits_list is not None
            for layer_idx, X_layer in per_kind_orig[kind].items():
                g = layer_to_group[layer_idx]
                b = int(bits_list[g])
                if b == 0:
                    continue
                enc = _joint_encoder(b)
                Xh_layer = per_kind_recon[kind][layer_idx]
                E = X_layer - Xh_layer
                encoded = enc.encode(E)
                rb = int(enc.footprint_bytes(encoded))
                residual_bytes_total += rb
                if kind == "k":
                    compressed_bytes_k += rb
                else:
                    compressed_bytes_v += rb
                E_hat = enc.decode(encoded)
                per_kind_recon[kind][layer_idx] = Xh_layer + E_hat
    elif residual_enc is not None:
        layer_to_group = {
            int(idx): g for g, group_idxs in enumerate(groups) for idx in group_idxs
        }
        for kind in ("k", "v"):
            for layer_idx, X_layer in per_kind_orig[kind].items():
                if selected_groups is not None and layer_to_group[layer_idx] not in selected_groups:
                    continue
                Xh_layer = per_kind_recon[kind][layer_idx]
                E = X_layer - Xh_layer
                encoded = residual_enc.encode(E)
                rb = int(residual_enc.footprint_bytes(encoded))
                residual_bytes_total += rb
                if kind == "k":
                    compressed_bytes_k += rb
                else:
                    compressed_bytes_v += rb
                E_hat = residual_enc.decode(encoded)
                per_kind_recon[kind][layer_idx] = Xh_layer + E_hat

    def global_rel_frob(kind: str) -> float:
        num_sq = 0.0
        den_sq = 0.0
        for layer_idx in per_kind_orig[kind]:
            X = per_kind_orig[kind][layer_idx].to(torch.float32)
            Xh = per_kind_recon[kind][layer_idx].to(torch.float32)
            num_sq += float(torch.sum((X - Xh) ** 2).item())
            den_sq += float(torch.sum(X ** 2).item())
        return 0.0 if den_sq == 0 else (num_sq / den_sq) ** 0.5

    error_k = global_rel_frob("k")
    error_v = global_rel_frob("v")

    compressed_bytes = core_bytes_total + factor_bytes_total + residual_bytes_total
    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")

    return CompressionResult(
        compression_ratio=ratio,
        original_bytes=int(original_bytes),
        original_bytes_k=int(orig_k),
        original_bytes_v=int(orig_v),
        compressed_bytes_k=int(compressed_bytes_k),
        compressed_bytes_v=int(compressed_bytes_v),
        core_bytes=int(core_bytes_total),
        factor_bytes=int(factor_bytes_total),
        residual_bytes=int(residual_bytes_total),
        error_k=error_k,
        error_v=error_v,
        groups=groups,
        ranks_k=ranks_k,
        ranks_v=ranks_v,
        config=cfg,
        k_target_used=k_target_resolved,
    )


def compress(
    layers: dict[int, dict[str, torch.Tensor]],
    config: CompressionConfig,
) -> CompressionResult | tuple[CompressionResult, CompressionResult]:
    resolved = _resolve_k_target(layers, config.k_target)
    if resolved == "both":
        r_pre = _compress_one(layers, config, "pre_rope")
        r_post = _compress_one(layers, config, "post_rope")
        return (r_pre, r_post)
    return _compress_one(layers, config, resolved)
