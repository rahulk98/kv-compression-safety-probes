"""Conformance vectors for FlashJoLT bit-unpacking semantics."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

VALID_UNPACK_BITS = (2, 4, 8)


# Independent oracle; keep it separate from the implementation under test.


def _quant_range(bits: int) -> tuple[int, int]:
    qmax = (1 << (bits - 1)) - 1
    return -qmax, qmax


def _decode_byte(byte_val: int, bits: int) -> list[int]:
    """Decode one packed byte into signed codes ordered by low-order field first."""
    if bits == 8:
        return [byte_val if byte_val < 128 else byte_val - 256]
    vpb = 8 // bits
    offset = 1 << (bits - 1)
    mask = (1 << bits) - 1
    return [((byte_val >> (bits * i)) & mask) - offset for i in range(vpb)]


def _encode_group(values: list[int], bits: int) -> int:
    """Pack up to `vpb` signed codes, using shifted-zero padding for a short group."""
    if bits == 8:
        assert len(values) == 1
        v = values[0]
        return v if v >= 0 else v + 256
    vpb = 8 // bits
    offset = 1 << (bits - 1)
    mask = (1 << bits) - 1
    byte_val = 0
    for i in range(vpb):
        shifted = (values[i] + offset) if i < len(values) else 0
        byte_val |= (shifted & mask) << (bits * i)
    return byte_val


def _encode_codes(codes: list[int], bits: int) -> list[int]:
    """Encode a flat code sequence with shifted-zero padding in the final group."""
    vpb = 8 // bits
    n_bytes = -(-len(codes) // vpb) if vpb > 1 else len(codes)  # ceil div
    out = []
    for g in range(n_bytes):
        group = codes[g * vpb : g * vpb + vpb]
        out.append(_encode_group(group, bits))
    return out




@dataclass(frozen=True)
class ConformanceVector:
    """One serializable packed-input and expected-output conformance case."""

    id: str
    category: str
    bits: int
    packed: list[int]
    total_numel: int
    start: int
    end: int
    expected: list[int]

    def packed_tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.tensor(self.packed, dtype=torch.uint8, device=device)




def _byte_enum_vectors() -> list[ConformanceVector]:
    """Return exhaustive per-byte vectors for every supported bit width."""
    vectors = []
    for bits in VALID_UNPACK_BITS:
        vpb = 8 // bits
        for byte_val in range(256):
            expected = _decode_byte(byte_val, bits)
            vectors.append(
                ConformanceVector(
                    id=f"byte_enum/bits={bits}/byte={byte_val}",
                    category="byte_enum",
                    bits=bits,
                    packed=[byte_val],
                    total_numel=vpb,
                    start=0,
                    end=vpb,
                    expected=expected,
                )
            )
    return vectors




def _padding_vectors() -> list[ConformanceVector]:
    """Return vectors covering every packed-group remainder per bit width."""
    vectors = []
    for bits in VALID_UNPACK_BITS:
        vpb = 8 // bits
        qmin, qmax = _quant_range(bits)
        span = qmax - qmin + 1
        for r in range(vpb):
            numel = 2 * vpb + r
            codes = [qmin + (i % span) for i in range(numel)]
            packed = _encode_codes(codes, bits)
            vectors.append(
                ConformanceVector(
                    id=f"padding/bits={bits}/numel={numel}/rem={r}",
                    category="padding",
                    bits=bits,
                    packed=packed,
                    total_numel=numel,
                    start=0,
                    end=numel,
                    expected=codes,
                )
            )
    return vectors




def _block_slice_vectors_for(bits: int, numel: int, label: str) -> list[ConformanceVector]:
    """Return exhaustive slice vectors for every valid interval of one payload."""
    qmin, qmax = _quant_range(bits)
    span = qmax - qmin + 1
    codes = [qmin + (i % span) for i in range(numel)]
    packed = _encode_codes(codes, bits)
    vectors = []
    for start in range(numel + 1):
        for end in range(start, numel + 1):
            vectors.append(
                ConformanceVector(
                    id=f"block_slice/{label}/bits={bits}/numel={numel}/[{start}:{end}]",
                    category="block_slice",
                    bits=bits,
                    packed=packed,
                    total_numel=numel,
                    start=start,
                    end=end,
                    expected=codes[start:end],
                )
            )
    return vectors


def _block_slice_vectors() -> list[ConformanceVector]:
    vectors = []
    # Byte-aligned total_numel (a clean multiple of vpb): isolates the
    # block-slicing concern from padding.
    for bits, numel in ((2, 12), (4, 10), (8, 8)):
        vectors += _block_slice_vectors_for(bits, numel, "aligned")
    # total_numel with a nonzero remainder mod vpb: the realistic case in
    # production, where `residual.numel = n_kv * T * head_dim` is essentially
    # never a multiple of vpb -- combines block-slicing with padding.
    # bits=8 has vpb=1, so no remainder is possible; skipped.
    for bits, numel in ((2, 13), (4, 11)):
        vectors += _block_slice_vectors_for(bits, numel, "padded")
    return vectors




def _empty_payload_vectors() -> list[ConformanceVector]:
    """Return empty-payload vectors for every supported unpacking bit width."""
    return [
        ConformanceVector(
            id=f"empty/bits={bits}",
            category="empty",
            bits=bits,
            packed=[],
            total_numel=0,
            start=0,
            end=0,
            expected=[],
        )
        for bits in VALID_UNPACK_BITS
    ]


def _multi_block_vectors() -> list[ConformanceVector]:
    """Return whole-array vectors that span one or more 1024-element blocks."""
    lengths = (1023, 1024, 1025, 2048, 2049, 5000)
    vectors = []
    for bits in VALID_UNPACK_BITS:
        qmin, qmax = _quant_range(bits)
        span = qmax - qmin + 1
        for numel in lengths:
            codes = [qmin + (i % span) for i in range(numel)]
            packed = _encode_codes(codes, bits)
            vectors.append(
                ConformanceVector(
                    id=f"multi_block/bits={bits}/numel={numel}",
                    category="multi_block",
                    bits=bits,
                    packed=packed,
                    total_numel=numel,
                    start=0,
                    end=numel,
                    expected=codes,
                )
            )
    return vectors




def _large_offset_vectors() -> list[ConformanceVector]:
    """Exercise large absolute offsets for every packed bit width."""
    vectors = [
        ConformanceVector(
            id="large_offset/bits=8/literal_repro_start16",
            category="large_offset",
            bits=8,
            packed=list(range(17)),
            total_numel=17,
            start=16,
            end=17,
            expected=[16],
        )
    ]
    starts = (17, 1024, 4096)
    block_len = 5
    for bits in VALID_UNPACK_BITS:
        qmin, qmax = _quant_range(bits)
        span = qmax - qmin + 1
        for s in starts:
            total_numel = s + block_len + 3  # a little slack past `end`
            codes = [qmin + (i % span) for i in range(total_numel)]
            packed = _encode_codes(codes, bits)
            start, end = s, s + block_len
            vectors.append(
                ConformanceVector(
                    id=f"large_offset/bits={bits}/start={s}",
                    category="large_offset",
                    bits=bits,
                    packed=packed,
                    total_numel=total_numel,
                    start=start,
                    end=end,
                    expected=codes[start:end],
                )
            )
    return vectors




def _full_range_offset_vectors() -> list[ConformanceVector]:
    """Return offset vectors covering every representable packed-code value."""
    vectors = []
    for bits in (2, 4):
        offset = 1 << (bits - 1)
        full_qmin = -offset
        full_span = 1 << bits  # full bit-field range, e.g. 4 at bits=2
        vpb = 8 // bits
        numel = full_span * 2 * vpb + 1  # >= two full cycles, byte-crossing
        codes = [full_qmin + (i % full_span) for i in range(numel)]
        packed = _encode_codes(codes, bits)
        for start in range(numel - 1):
            end = start + 1
            vectors.append(
                ConformanceVector(
                    id=f"full_range_offset/bits={bits}/numel={numel}/[{start}:{end}]",
                    category="full_range_offset",
                    bits=bits,
                    packed=packed,
                    total_numel=numel,
                    start=start,
                    end=end,
                    expected=codes[start:end],
                )
            )
    return vectors


def generate_conformance_vectors() -> list[ConformanceVector]:
    """Return all unpacking conformance vectors."""
    return (
        _byte_enum_vectors()
        + _padding_vectors()
        + _block_slice_vectors()
        + _empty_payload_vectors()
        + _multi_block_vectors()
        + _large_offset_vectors()
        + _full_range_offset_vectors()
    )




@dataclass(frozen=True)
class ConformanceFailure:
    vector_id: str
    category: str
    bits: int
    start: int
    end: int
    expected: list[int]
    got: list[int] | str  # str only when the impl raised


@dataclass(frozen=True)
class ConformanceReport:
    total: int
    failed: list[ConformanceFailure] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return self.total - len(self.failed)

    @property
    def all_passed(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        by_cat: dict[str, int] = {}
        for f in self.failed:
            by_cat[f.category] = by_cat.get(f.category, 0) + 1
        cat_str = ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())) or "none"
        return f"{self.passed}/{self.total} passed, failures by category: {cat_str}"


def _check_result_contract(
    got_t, v: ConformanceVector, want_device: torch.device
) -> str | None:
    """Validate an unpacking result's tensor contract for one vector."""
    if not isinstance(got_t, torch.Tensor):
        return f"return type: expected torch.Tensor, got {type(got_t).__name__}"
    if got_t.dtype != torch.int8:
        return f"dtype: expected torch.int8, got {got_t.dtype}"
    if got_t.dim() != 1:
        return f"shape: expected a 1-D tensor, got shape {tuple(got_t.shape)}"
    expected_len = v.end - v.start
    if got_t.shape[0] != expected_len:
        return f"length: expected {expected_len} (end-start), got {got_t.shape[0]}"
    if got_t.device.type != want_device.type or (
        want_device.index is not None and got_t.device.index != want_device.index
    ):
        return f"device: expected {want_device}, got {got_t.device}"
    return None


def run_conformance(
    impl,
    vectors: list[ConformanceVector] | None = None,
    device: torch.device | str = "cpu",
) -> ConformanceReport:
    """Run every conformance vector against `impl` and return structured pass/fail detail."""
    if vectors is None:
        vectors = generate_conformance_vectors()
    want_device = torch.device(device)
    failed: list[ConformanceFailure] = []
    for v in vectors:
        packed_t = v.packed_tensor(device)
        try:
            got_t = impl(packed_t, v.bits, v.total_numel, v.start, v.end)
            reason = _check_result_contract(got_t, v, want_device)
            if reason is not None:
                failed.append(
                    ConformanceFailure(
                        v.id, v.category, v.bits, v.start, v.end, v.expected, reason
                    )
                )
                continue
            got = [int(x) for x in got_t.reshape(-1).tolist()]
        except Exception as e:  # noqa: BLE001 -- a raising impl is a reportable failure, not a crash
            failed.append(
                ConformanceFailure(
                    v.id, v.category, v.bits, v.start, v.end, v.expected,
                    f"raised {type(e).__name__}: {e}",
                )
            )
            continue
        if got != v.expected:
            failed.append(
                ConformanceFailure(v.id, v.category, v.bits, v.start, v.end, v.expected, got)
            )
    return ConformanceReport(total=len(vectors), failed=failed)


def unpack_adapter(unpack_fn):
    """Adapt a whole-array unpacker to the range-unpacker conformance signature."""

    def _call(packed: torch.Tensor, bits: int, total_numel: int, start: int, end: int) -> torch.Tensor:
        return unpack_fn(packed, bits, total_numel)[start:end]

    return _call
