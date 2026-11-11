"""A read-only array-to-bytes codec that decodes runs of chunks in different ways.

An array made of files written at different times can have chunks that are not
all encoded alike: the compression changed, or a file's last chunk is stored
larger than its place in the grid. This codec holds a short table of *cases*,
each covering a run of chunks along one axis, and decodes a chunk with the case
its grid position falls in. Nothing is stored per chunk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final

from zarr.abc.codec import ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin
from zarr.core.array_spec import ArraySpec
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.store import ByteGetter
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.indexing import SelectorTuple

__all__ = ["SwitchCodec", "rle_cases"]

CODEC_NAME: Final = "switch"
_CASE_KEYS: Final = {
    "chunks",
    "codecs",
    "encoded_chunk_shape",
    "fill_value",
    "coalesce_max_gap_bytes",
}


@lru_cache(maxsize=256)
def _pipeline(codecs_json: str, spec: ArraySpec) -> Any:
    from zarr.core.codec_pipeline import BatchedCodecPipeline
    from zarr.core.metadata.v3 import parse_codecs

    codecs = parse_codecs(json.loads(codecs_json))
    return BatchedCodecPipeline.from_codecs(codecs).evolve_from_array_spec(spec)


@lru_cache(maxsize=256)
def _partial_decoder(codecs_json: str, spec: ArraySpec) -> Any:
    """The chain's only codec, when it can decode part of a chunk by itself (sharding)."""
    from zarr.core.metadata.v3 import parse_codecs

    codecs = parse_codecs(json.loads(codecs_json))
    if len(codecs) == 1 and isinstance(codecs[0], ArrayBytesCodecPartialDecodeMixin):
        return codecs[0].evolve_from_array_spec(spec)
    return None


@dataclass(frozen=True)
class SwitchCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """Decode each chunk with the case of the run of chunks it belongs to.

    Parameters
    ----------
    axis : int
        The axis along which the runs lie.
    cases : sequence of dict
        The runs, in order along `axis`. Each case has:

        - ``chunks``: the number of chunks along `axis` that the run covers;
        - ``codecs``: the codec chain that decodes them;
        - ``encoded_chunk_shape`` (optional): the shape a stored chunk decodes
          to, when it is larger than the chunk's place in the grid; the codec
          keeps its leading corner. An entry of ``null`` means "the size of the
          chunk's place".
        - ``fill_value`` (optional): what a chunk of the run that is not stored
          reads as, when that is not the array's fill value;
        - ``coalesce_max_gap_bytes`` (optional): for a chain that reads parts of
          a chunk by byte range (sharding), how close two ranges must be to be
          read in one request. 0 reads exactly the ranges asked for.

    Notes
    -----
    The codec MUST be the array's only codec, and is read-only.
    """

    is_fixed_size = False
    needs_chunk_coords = True

    axis: int
    cases_json: str

    def __init__(self, *, axis: int, cases: Any) -> None:
        cases = json.loads(cases) if isinstance(cases, str) else [dict(c) for c in cases]
        axis_given: object = axis
        if isinstance(axis_given, bool) or not isinstance(axis_given, int) or axis_given < 0:
            raise TypeError(f"axis must be a non-negative integer. Got {axis!r}.")
        if not cases:
            raise ValueError("The switch codec needs at least one case.")
        for case in cases:
            if not {"chunks", "codecs"} <= set(case) <= _CASE_KEYS:
                raise ValueError(
                    f"A case has 'chunks' and 'codecs', and optionally "
                    f"{sorted(_CASE_KEYS - {'chunks', 'codecs'})}. Got {sorted(case)}."
                )
            if (
                isinstance(case["chunks"], bool)
                or not isinstance(case["chunks"], int)
                or case["chunks"] < 1
            ):
                raise ValueError(f"A case covers at least one chunk. Got {case['chunks']!r}.")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "cases_json", json.dumps(cases, sort_keys=True))

    @property
    def cases(self) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = json.loads(self.cases_json)
        return cases

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration = parse_named_configuration(data, CODEC_NAME)
        return cls(axis=configuration["axis"], cases=configuration["cases"])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": CODEC_NAME, "configuration": {"axis": self.axis, "cases": self.cases}}

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError

    def _case(self, index: int) -> dict[str, Any]:
        start = 0
        for case in self.cases:
            start += case["chunks"]
            if index < start:
                return case
        raise IndexError(f"Chunk {index} along axis {self.axis} lies beyond the last case.")

    def _index(self, byte_getter: ByteGetter, chunk_spec: ArraySpec) -> int:
        if chunk_spec.chunk_coords is not None:
            return chunk_spec.chunk_coords[self.axis]
        # An array that does not pass coordinates (an older reader): read them back
        # from the chunk's key, in the `default` encoding with the `/` separator.
        key = getattr(byte_getter, "path", None)
        if not isinstance(key, str):
            raise TypeError("The switch codec cannot tell which chunk it is decoding.")
        return int(key.split("/")[-len(chunk_spec.shape) :][self.axis])

    async def _decode_partial_single(
        self, byte_getter: ByteGetter, selection: SelectorTuple, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        case = self._case(self._index(byte_getter, chunk_spec))
        encoded_shape = tuple(
            n if e is None else e
            for e, n in zip(
                case.get("encoded_chunk_shape") or chunk_spec.shape, chunk_spec.shape, strict=True
            )
        )
        own_fill = "fill_value" in case
        fill = (
            chunk_spec.dtype.from_json_scalar(case["fill_value"], zarr_format=3)
            if own_fill
            else chunk_spec.fill_value
        )
        config = chunk_spec.config
        if "coalesce_max_gap_bytes" in case:
            config = replace(config, sharding_coalesce_max_gap_bytes=case["coalesce_max_gap_bytes"])
        spec = ArraySpec(
            shape=encoded_shape,
            dtype=chunk_spec.dtype,
            fill_value=fill,
            config=config,
            prototype=chunk_spec.prototype,
        )
        codecs_json = json.dumps(case["codecs"], sort_keys=True)
        inner = _partial_decoder(codecs_json, spec)
        if inner is not None:
            # The chunk's place is the leading corner of the stored shape, so the
            # selection means the same thing in both. Missing inner chunks are
            # filled from `spec`.
            out = await inner._decode_partial_single(byte_getter, selection, spec)
            return self._filled(selection, chunk_spec, fill) if out is None and own_fill else out
        chunk_bytes = await byte_getter.get(prototype=chunk_spec.prototype)
        if chunk_bytes is None:
            # None makes the reader use the array's fill value.
            return self._filled(selection, chunk_spec, fill) if own_fill else None
        (decoded,) = await _pipeline(codecs_json, spec).decode([(chunk_bytes, spec)])
        if encoded_shape != chunk_spec.shape:
            decoded = decoded[tuple(slice(0, n) for n in chunk_spec.shape)]
        return decoded[selection]  # type: ignore[no-any-return]

    @staticmethod
    def _filled(selection: SelectorTuple, chunk_spec: ArraySpec, fill: Any) -> NDBuffer:
        """What a selection of a chunk that is not stored reads as: all `fill`."""
        from zarr.core.chunk_grids import ChunkGrid
        from zarr.core.indexing import get_indexer

        shape = chunk_spec.shape
        indexer = get_indexer(
            selection, shape=shape, chunk_grid=ChunkGrid.from_sizes(shape, list(shape))
        )
        out = chunk_spec.prototype.nd_buffer.create(
            shape=indexer.shape,
            dtype=chunk_spec.dtype.to_native_dtype(),
            order=chunk_spec.order,
            fill_value=fill,
        )
        return out.reshape(indexer.sel_shape) if hasattr(indexer, "sel_shape") else out

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        raise NotImplementedError("The switch codec must be the array's only codec.")

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        raise NotImplementedError("The switch codec is read-only.")


def rle_cases(
    codecs_per_file: list[list[dict[str, Any]]], chunks_per_file: list[int]
) -> list[dict[str, Any]]:
    """Run-length encode per-file codec chains into cases."""
    cases: list[dict[str, Any]] = []
    for codecs, n in zip(codecs_per_file, chunks_per_file, strict=True):
        if cases and cases[-1]["codecs"] == codecs:
            cases[-1]["chunks"] += n
        else:
            cases.append({"chunks": n, "codecs": codecs})
    return cases
