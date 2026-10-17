"""Experimental `switch` and `crop` codecs, a sketch for zarr-developers/zarr-specs#399.

Neither codec is a registered zarr extension. Both live in this one module so they can
move to a standalone package; the only other change in zarr-python is their two
`register_codec` lines in `zarr.codecs`.

`switch` is an array -> bytes codec whose configuration is a table of codec chains::

    {"name": "switch", "configuration": {"cases": {
        "<key>": {"codecs": [...], "parameters": ["float64", ...]}, ...}}}

A chunk's bytes are a header followed by the bytes its case's chain decodes:

- the case key: 8 bytes, the first 8 bytes of the SHA-256 of the case's canonical JSON
  (`case_key`), written in the table as 16 hex digits;
- then the case's parameters, each little-endian in its declared data type.

Anywhere in a case's codec configurations, `{"parameter": k}` stands for the chunk's
k-th parameter. The key names a case by its content, so arrays that share a chain share
its key and two case tables merge by union.

`switch` hands the chosen chain the chunk with its header removed: a buffer slice for a
full decode, and for a partial decode a byte getter that shifts every request past the
header, so a `sharding_indexed` codec inside a case reads only the inner chunks it needs.

`crop` is an array -> array codec, `{"shape": [...]}`: the stored chunk's shape, at
least the chunk's shape in every dimension. Decoding keeps the leading corner (HDF5 and
regular Zarr grids store edge chunks whole); encoding pads with the fill value.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.codec import Codec, CodecPipeline
    from zarr.abc.store import ByteGetter, ByteRequest
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.indexing import SelectorTuple

KEY_BYTES = 8


def canonical(case: dict[str, Any]) -> bytes:
    """A case's canonical JSON: sorted keys, no whitespace, UTF-8.

    shortcut: Python's `json.dumps`, not RFC 8785, so number formatting is
    implementation-defined; a registered spec must name RFC 8785.
    """
    return json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def case_key(case: dict[str, Any]) -> bytes:
    """The 8-byte key of a case: the first 8 bytes of the SHA-256 of its canonical JSON."""
    return hashlib.sha256(canonical(case)).digest()[:KEY_BYTES]


def _json_number(v: float) -> JSON:
    """A parameter as JSON, as Zarr writes fill values: NaN and the infinities as strings."""
    if np.isfinite(v):
        return v
    return "NaN" if np.isnan(v) else ("Infinity" if v > 0 else "-Infinity")


@lru_cache(maxsize=4096)
def _completed(codecs: str, values: tuple[float | int, ...]) -> tuple[Codec, ...]:
    """A case's codecs with its parameters filled in, parsed. Cached, so chunks that
    share a case and its values share one parse."""
    from zarr.core.metadata.v3 import parse_codecs

    def fill(node: Any) -> Any:
        if isinstance(node, dict):
            if set(node) == {"parameter"}:
                return _json_number(values[node["parameter"]])
            return {k: fill(v) for k, v in node.items()}
        if isinstance(node, list):
            return [fill(v) for v in node]
        return node

    return tuple(parse_codecs(fill(json.loads(codecs))))


def _is_total(selection: SelectorTuple, shape: tuple[int, ...]) -> bool:
    if not isinstance(selection, tuple) or len(selection) != len(shape):
        return False
    return all(
        isinstance(s, slice)
        and s.start in (None, 0)
        and (s.stop is None or s.stop >= n)
        and s.step in (None, 1)
        for s, n in zip(selection, shape, strict=True)
    )


@dataclass(frozen=True)
class _HeaderSkippingByteGetter:
    """A view of a chunk's bytes with its first `skip` bytes removed."""

    inner: ByteGetter
    skip: int

    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        shifted: ByteRequest
        if byte_range is None:
            shifted = OffsetByteRequest(self.skip)
        elif isinstance(byte_range, RangeByteRequest):
            shifted = RangeByteRequest(byte_range.start + self.skip, byte_range.end + self.skip)
        elif isinstance(byte_range, OffsetByteRequest):
            shifted = OffsetByteRequest(byte_range.offset + self.skip)
        elif isinstance(byte_range, SuffixByteRequest):
            # a suffix counts from the end, which the header does not move
            # shortcut: a suffix longer than the body would reach into the header;
            # sharding only asks for its index, which is always inside the body.
            shifted = byte_range
        else:
            raise TypeError(f"Unexpected byte_range, got {byte_range}")
        return await self.inner.get(prototype=prototype, byte_range=shifted)


@dataclass(frozen=True)
class CropCodec(ArrayArrayCodec):
    """`crop`: the stored chunk has `shape`; decoding keeps its leading corner."""

    shape: tuple[int, ...]
    is_fixed_size = True

    def __init__(self, *, shape: Any) -> None:
        object.__setattr__(self, "shape", tuple(int(n) for n in shape))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration = parse_named_configuration(data, "crop")
        return cls(**configuration)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "crop", "configuration": {"shape": list(self.shape)}}

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if len(self.shape) != len(chunk_spec.shape) or any(
            s < c for s, c in zip(self.shape, chunk_spec.shape, strict=True)
        ):
            raise ValueError(
                f"crop: stored shape {self.shape} is smaller than the chunk {chunk_spec.shape}"
            )
        return replace(chunk_spec, shape=self.shape)

    async def _decode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        return chunk_array[tuple(slice(0, n) for n in chunk_spec.shape)]

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        a = chunk_array.as_numpy_array()
        out = np.full(self.shape, chunk_spec.fill_value, dtype=a.dtype)
        out[tuple(slice(0, n) for n in a.shape)] = a
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length // int(np.prod(chunk_spec.shape)) * int(np.prod(self.shape))


@dataclass(frozen=True)
class SwitchCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """`switch`: each chunk's header picks its codec chain and fills in its parameters.

    Decode only: a writer would have to choose a case for each chunk, which this sketch
    leaves to whoever builds the references (VirtualiZarr).
    """

    # (key hex, codecs as JSON text, parameter data types), in table order
    cases: tuple[tuple[str, str, tuple[str, ...]], ...]
    is_fixed_size = False

    def __init__(self, *, cases: dict[str, dict[str, Any]]) -> None:
        parsed = []
        for key, case in cases.items():
            case = {"codecs": case["codecs"], "parameters": list(case.get("parameters", []))}
            if case_key(case).hex() != key:
                raise ValueError(
                    f"switch: case {key} is not the key of its contents ({case_key(case).hex()})"
                )
            for dtype in case["parameters"]:
                if np.dtype(dtype).kind not in "iuf":
                    raise ValueError(f"switch: parameter type {dtype!r} is not a number type")
            parsed.append((key, json.dumps(case["codecs"]), tuple(case["parameters"])))
        object.__setattr__(self, "cases", tuple(parsed))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration = parse_named_configuration(data, "switch")
        return cls(**configuration)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "switch",
            "configuration": {
                "cases": {
                    key: {"codecs": json.loads(codecs), "parameters": list(params)}
                    for key, codecs, params in self.cases
                }
            },
        }

    def _case(self, head: bytes) -> tuple[str, tuple[str, ...]]:
        key = head[:KEY_BYTES].hex()
        for k, codecs, params in self.cases:
            if k == key:
                return codecs, params
        raise ValueError(f"switch: the chunk names case {key}, which the array does not list")

    @staticmethod
    def _values(raw: bytes, params: tuple[str, ...]) -> tuple[float | int, ...]:
        values, pos = [], 0
        for dtype in params:
            dt = np.dtype(dtype).newbyteorder("<")
            values.append(np.frombuffer(raw, dtype=dt, count=1, offset=pos)[0].item())
            pos += dt.itemsize
        return tuple(values)

    @staticmethod
    def _header_size(params: tuple[str, ...]) -> int:
        return KEY_BYTES + sum(np.dtype(d).itemsize for d in params)

    @staticmethod
    def _pipeline(
        codecs: str, values: tuple[float | int, ...], chunk_spec: ArraySpec
    ) -> CodecPipeline:
        from zarr.registry import get_pipeline_class

        return (
            get_pipeline_class()
            .from_codecs(_completed(codecs, values))
            .evolve_from_array_spec(chunk_spec)
        )

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        codecs, params = self._case(chunk_bytes[:KEY_BYTES].to_bytes())
        size = self._header_size(params)
        values = self._values(chunk_bytes[KEY_BYTES:size].to_bytes(), params)
        pipeline = self._pipeline(codecs, values, chunk_spec)
        (out,) = await pipeline.decode([(chunk_bytes[size:], chunk_spec)])
        if out is None:
            raise ValueError("switch: the case's chain decoded the chunk to nothing")
        return out

    async def _decode_partial_single(
        self, byte_getter: ByteGetter, selection: SelectorTuple, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        prototype = chunk_spec.prototype
        if _is_total(selection, chunk_spec.shape):
            # the whole chunk: one request, as without switch
            whole = await byte_getter.get(prototype=prototype)
            return None if whole is None else await self._decode_single(whole, chunk_spec)
        head = await byte_getter.get(prototype=prototype, byte_range=RangeByteRequest(0, KEY_BYTES))
        if head is None:
            return None
        codecs, params = self._case(head.to_bytes())
        size = self._header_size(params)
        values: tuple[float | int, ...] = ()
        if params:
            raw = await byte_getter.get(
                prototype=prototype, byte_range=RangeByteRequest(KEY_BYTES, size)
            )
            if raw is None:
                return None
            values = self._values(raw.to_bytes(), params)
        body = _HeaderSkippingByteGetter(byte_getter, size)
        chain = _completed(codecs, values)
        if len(chain) == 1 and isinstance(chain[0], ArrayBytesCodecPartialDecodeMixin):
            # a lone partial-decode codec (sharding): the same rule as the pipelines'
            codec = chain[0].evolve_from_array_spec(chunk_spec)
            (part,) = await codec.decode_partial([(body, selection, chunk_spec)])
            return part
        pipeline = self._pipeline(codecs, values, chunk_spec)
        # shortcut: a partial read of a case without partial decode costs a request
        # for the header and one for the body, against one without switch.
        buf = await body.get(prototype=prototype)
        if buf is None:
            return None
        (out,) = await pipeline.decode([(buf, chunk_spec)])
        return None if out is None else out[selection]

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        raise NotImplementedError(
            "switch decodes only: a writer would choose each chunk's case and write its header"
        )

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError("switch chunks are variable-size")
