"""An array-to-bytes codec that encodes chunks in different ways.

An array made of files written at different times can have chunks that are not
all encoded alike: the compression changed, or a file's last chunk is stored
larger than its place in the grid. This codec holds a short table of *cases*
and decodes each chunk with one of them. Which one is found in either of two
ways:

- ``"select": "key"``: the chunk's bytes begin with a short key that names its
  case. The chunk describes itself, so it can be moved, re-keyed or nested
  without the codec knowing where it is. The store has to supply the key in
  front of the chunk (a virtual reference with a literal prefix does).
- ``"select": "position"``: each case covers a run of chunks along one axis,
  and the chunk's grid position picks the case. Nothing is stored per chunk,
  but the codec has to be told which chunk it is decoding.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, Literal

from zarr.abc.codec import ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.array_spec import ArraySpec
from zarr.core.common import JSON, parse_named_configuration
from zarr.storage._wrapper import WrapperStore

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.store import ByteGetter, ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.indexing import SelectorTuple

__all__ = ["SwitchCodec", "rle_cases"]

CODEC_NAME: Final = "switch"
_CASE_KEYS: Final = {
    "chunks",
    "key",
    "codecs",
    "encoded_chunk_shape",
    "fill_value",
    "coalesce_max_gap_bytes",
}


@dataclass(frozen=True)
class _Table:
    """The cases, parsed once, with what is needed to find one in constant time."""

    cases: list[dict[str, Any]]
    codecs_json: list[str]  # each case's chain, as the key of the pipeline cache
    ends: list[int]  # by position: the first chunk after each case
    by_key: dict[bytes, int]  # by key: the case each key names
    reads_ranges: bool  # some case's chain may read part of a chunk (sharding)


@lru_cache(maxsize=64)
def _table(cases_json: str, select: str) -> _Table:
    from zarr.core.metadata.v3 import parse_codecs

    cases: list[dict[str, Any]] = json.loads(cases_json)
    ends, total = [], 0
    for case in cases:
        total += case.get("chunks", 0)
        ends.append(total)
    by_key = {bytes.fromhex(c["key"]): i for i, c in enumerate(cases)} if select == "key" else {}
    reads_ranges = False
    for case in cases:
        chain = case["codecs"]
        if len(chain) == 1 and isinstance(
            parse_codecs(chain)[0], ArrayBytesCodecPartialDecodeMixin
        ):
            reads_ranges = True
            break
    return _Table(
        cases, [json.dumps(c["codecs"], sort_keys=True) for c in cases], ends, by_key, reads_ranges
    )


# One entry per distinct chain and chunk shape in use. A record whose every granule
# has a chain of its own will exceed this and rebuild chains as it reads.
@lru_cache(maxsize=4096)
def _pipeline(codecs_json: str, spec: ArraySpec) -> Any:
    from zarr.core.codec_pipeline import BatchedCodecPipeline
    from zarr.core.metadata.v3 import parse_codecs

    codecs = parse_codecs(json.loads(codecs_json))
    return BatchedCodecPipeline.from_codecs(codecs).evolve_from_array_spec(spec)


@lru_cache(maxsize=4096)
def _partial_decoder(codecs_json: str, spec: ArraySpec) -> Any:
    """The chain's only codec, when it can decode part of a chunk by itself (sharding)."""
    from zarr.core.metadata.v3 import parse_codecs

    codecs = parse_codecs(json.loads(codecs_json))
    if len(codecs) == 1 and isinstance(codecs[0], ArrayBytesCodecPartialDecodeMixin):
        return codecs[0].evolve_from_array_spec(spec)
    return None


def _shift(byte_range: ByteRequest | None, k: int) -> ByteRequest:
    """The same range of a chunk, counted from before its `k`-byte key."""
    if byte_range is None:
        return OffsetByteRequest(k)
    if isinstance(byte_range, RangeByteRequest):
        return RangeByteRequest(byte_range.start + k, byte_range.end + k)
    if isinstance(byte_range, OffsetByteRequest):
        return OffsetByteRequest(byte_range.offset + k)
    if isinstance(byte_range, SuffixByteRequest):
        return byte_range
    raise TypeError(f"Unexpected byte range {byte_range!r}.")


class _AfterKey:
    """A chunk's bytes without the key in front of them.

    A codec that reads byte ranges of the chunk (sharding) sees offsets that
    start after the key, so its own index stays valid.
    """

    def __init__(self, byte_getter: ByteGetter, key_bytes: int) -> None:
        self._byte_getter = byte_getter
        self._key_bytes = key_bytes

    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        return await self._byte_getter.get(prototype, _shift(byte_range, self._key_bytes))


class _AfterKeyStore(WrapperStore[Any]):
    """A store whose values are read from after a key in front of each of them.

    Sharding fetches many ranges at once, and merges them, only from a store. This
    keeps that path: the ranges are shifted and handed to the store underneath.
    """

    def __init__(self, store: Any, key_bytes: int) -> None:
        super().__init__(store)
        self._key_bytes = key_bytes

    def _with_store(self, store: Any) -> Self:
        return type(self)(store, self._key_bytes)

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        out: Buffer | None = await self._store.get(
            key, prototype, _shift(byte_range, self._key_bytes)
        )
        return out

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        return super().get_sync(
            key, prototype=prototype, byte_range=_shift(byte_range, self._key_bytes)
        )

    async def get_partial_values(self, prototype: BufferPrototype, key_ranges: Any) -> Any:
        shifted = [(key, _shift(r, self._key_bytes)) for key, r in key_ranges]
        return await self._store.get_partial_values(prototype, shifted)

    async def get_ranges(
        self, key: str, byte_ranges: Any, *, prototype: BufferPrototype, **kwargs: Any
    ) -> Any:
        shifted = [_shift(r, self._key_bytes) for r in byte_ranges]
        async for group in self._store.get_ranges(key, shifted, prototype=prototype, **kwargs):
            yield group


def _after_key(byte_getter: ByteGetter, key_bytes: int) -> Any:
    from zarr.storage._common import StorePath

    if isinstance(byte_getter, StorePath):
        return StorePath(_AfterKeyStore(byte_getter.store, key_bytes), byte_getter.path)
    return _AfterKey(byte_getter, key_bytes)


@dataclass(frozen=True)
class SwitchCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """Decode each chunk with one of several cases.

    Parameters
    ----------
    cases : sequence of dict
        Each case has:

        - ``codecs``: the codec chain of its chunks;
        - ``key`` (with ``select="key"``): the hex of the bytes that a chunk of
          this case begins with;
        - ``chunks`` (with ``select="position"``): the number of chunks along
          `axis` that the case covers; the cases are then in order along it;
        - ``encoded_chunk_shape`` (optional): the shape a stored chunk decodes
          to, when it is larger than the chunk's place in the grid; the codec
          keeps its leading corner. An entry of ``null`` means "the size of the
          chunk's place".
        - ``fill_value`` (optional): what a chunk of the case that stores
          nothing reads as, when that is not the array's fill value;
        - ``coalesce_max_gap_bytes`` (optional): for a chain that reads parts of
          a chunk by byte range (sharding), how close two ranges must be to be
          read in one request. 0 reads exactly the ranges asked for.
    select : {"position", "key"}
        How a chunk's case is found.
    axis : int
        With ``select="position"``, the axis along which the cases lie.
    key_bytes : int
        With ``select="key"``, the length of the key in front of every chunk.

    Notes
    -----
    The codec MUST be the array's only codec. With ``select="position"`` it also
    encodes: a chunk written through the array gets its case's chain, padded to
    the case's stored shape. With ``select="key"`` it is read-only.
    """

    is_fixed_size = False
    needs_chunk_coords = True

    axis: int
    cases_json: str
    select: Literal["position", "key"]
    key_bytes: int

    def __init__(
        self,
        *,
        cases: Any,
        axis: int = 0,
        select: Literal["position", "key"] = "position",
        key_bytes: int = 8,
    ) -> None:
        cases = json.loads(cases) if isinstance(cases, str) else [dict(c) for c in cases]
        axis_given: object = axis
        if isinstance(axis_given, bool) or not isinstance(axis_given, int) or axis_given < 0:
            raise TypeError(f"axis must be a non-negative integer. Got {axis!r}.")
        if select not in ("position", "key"):
            raise ValueError(f"select must be 'position' or 'key'. Got {select!r}.")
        if not cases:
            raise ValueError("The switch codec needs at least one case.")
        required = {"codecs", "chunks" if select == "position" else "key"}
        seen: set[str] = set()
        for case in cases:
            if not required <= set(case) <= _CASE_KEYS:
                raise ValueError(
                    f"With select={select!r} a case has {sorted(required)}, and optionally "
                    f"{sorted(_CASE_KEYS - required)}. Got {sorted(case)}."
                )
            if select == "position":
                n = case["chunks"]
                if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                    raise ValueError(f"A case covers at least one chunk. Got {n!r}.")
            else:
                key = case["key"]
                if not isinstance(key, str) or len(bytes.fromhex(key)) != key_bytes:
                    raise ValueError(f"A case's key is {key_bytes} bytes of hex. Got {key!r}.")
                if key in seen:
                    raise ValueError(f"Two cases have the key {key!r}.")
                seen.add(key)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "cases_json", json.dumps(cases, sort_keys=True))
        object.__setattr__(self, "select", select)
        object.__setattr__(self, "key_bytes", key_bytes)

    @property
    def _tab(self) -> _Table:
        # Kept on the instance: finding it by the cases' text costs time in proportion
        # to that text, which a long table would pay again for every chunk.
        tab: _Table | None = self.__dict__.get("_table_cache")
        if tab is None:
            tab = _table(self.cases_json, self.select)
            object.__setattr__(self, "_table_cache", tab)
        return tab

    @property
    def cases(self) -> list[dict[str, Any]]:
        return self._tab.cases

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration = parse_named_configuration(data, CODEC_NAME)
        return cls(
            cases=configuration["cases"],
            axis=configuration.get("axis", 0),  # type: ignore[arg-type]
            select=configuration.get("select", "position"),  # type: ignore[arg-type]
            key_bytes=configuration.get("key_bytes", 8),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, JSON]:
        configuration: dict[str, JSON]
        if self.select == "key":
            configuration = {"select": "key", "key_bytes": self.key_bytes, "cases": self.cases}
        else:
            configuration = {"axis": self.axis, "cases": self.cases}
        return {"name": CODEC_NAME, "configuration": configuration}

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError

    # --- finding a chunk's case

    def _case_at(self, index: int) -> int:
        table = self._tab
        i = bisect_right(table.ends, index)
        if index < 0 or i == len(table.ends):
            raise IndexError(f"Chunk {index} along axis {self.axis} lies beyond the last case.")
        return i

    def _case_named(self, key: bytes) -> int:
        try:
            return self._tab.by_key[key]
        except KeyError:
            raise ValueError(
                f"A chunk names the case {key.hex()!r}, which the array does not list."
            ) from None

    def _position(self, chunk_spec: ArraySpec, byte_getter: ByteGetter | None = None) -> int:
        if chunk_spec.chunk_coords is not None:
            return chunk_spec.chunk_coords[self.axis]
        # A reader that does not pass coordinates: read them back from the chunk's
        # key, in the `default` encoding with the `/` separator.
        key = getattr(byte_getter, "path", None)
        if not isinstance(key, str):
            raise NotImplementedError(
                "The switch codec was not told which chunk this is. It must be the "
                "array's only codec, in a reader that passes chunk coordinates."
            )
        return int(key.split("/")[-len(chunk_spec.shape) :][self.axis])

    def _resolve(self, index: int, chunk_spec: ArraySpec) -> tuple[ArraySpec, str, bool]:
        """What a case means for a chunk: the spec its chain works with (the stored
        shape, the case's fill value), the chain as JSON, and whether the case has
        a fill value of its own."""
        table = self._tab
        case = table.cases[index]
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
        return spec, table.codecs_json[index], own_fill

    # --- decoding

    async def _decode_partial_single(
        self, byte_getter: ByteGetter, selection: SelectorTuple, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        source: Any = byte_getter
        if self.select == "key" and not self._tab.reads_ranges:
            # No case reads part of a chunk: one read brings the key and the chunk.
            whole = await byte_getter.get(prototype=chunk_spec.prototype)
            if whole is None:
                return None
            decoded = await self._decode_single(whole, chunk_spec)
            return decoded[selection]
        if self.select == "key":
            head = await byte_getter.get(chunk_spec.prototype, RangeByteRequest(0, self.key_bytes))
            if head is None:
                return None  # nothing stored: the array's fill value
            case = self._case_named(head.to_bytes())
            source = _after_key(byte_getter, self.key_bytes)
        else:
            case = self._case_at(self._position(chunk_spec, byte_getter))
        spec, codecs_json, own_fill = self._resolve(case, chunk_spec)
        inner = _partial_decoder(codecs_json, spec)
        if inner is not None and own_fill and self.select == "key":
            from zarr.codecs.sharding import ShardingCodec

            if not isinstance(inner, ShardingCodec):
                # The chunk may be its key alone. A shard says so in its index; any
                # other chain is read whole, where an empty chunk shows as one.
                inner = None
        if inner is not None:
            # The chunk's place is the leading corner of the stored shape, so the
            # selection means the same thing in both. Missing inner chunks are
            # filled from `spec`.
            out = await inner._decode_partial_single(source, selection, spec)
            if out is None and own_fill:
                return self._filled(selection, chunk_spec, spec.fill_value)
            return out  # type: ignore[no-any-return]
        if source is byte_getter:
            chunk_bytes = await source.get(prototype=chunk_spec.prototype)
        else:
            # Read the chunk with its key and drop the key here: asking a store for
            # "everything after the key" fails where the chunk is the key alone.
            whole = await byte_getter.get(prototype=chunk_spec.prototype)
            chunk_bytes = (
                None
                if whole is None
                else chunk_spec.prototype.buffer.from_bytes(whole.to_bytes()[self.key_bytes :])
            )
        if chunk_bytes is None or len(chunk_bytes) == 0:
            # A key with nothing after it is a chunk that stores nothing but still
            # names its case. None makes the reader use the array's fill value.
            return self._filled(selection, chunk_spec, spec.fill_value) if own_fill else None
        (decoded,) = await _pipeline(codecs_json, spec).decode([(chunk_bytes, spec)])
        if spec.shape != chunk_spec.shape:
            decoded = decoded[tuple(slice(0, n) for n in chunk_spec.shape)]
        return decoded[selection]  # type: ignore[no-any-return]

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        if self.select == "key":
            raw = chunk_bytes.to_bytes()
            case = self._case_named(raw[: self.key_bytes])
            if len(raw) == self.key_bytes:
                # A key and nothing else: the chunk stores no data and reads as its
                # case's fill value.
                spec, _, _ = self._resolve(case, chunk_spec)
                whole = tuple(slice(0, n) for n in chunk_spec.shape)
                return self._filled(whole, chunk_spec, spec.fill_value)
            chunk_bytes = chunk_spec.prototype.buffer.from_bytes(raw[self.key_bytes :])
        else:
            case = self._case_at(self._position(chunk_spec))
        spec, codecs_json, _ = self._resolve(case, chunk_spec)
        (decoded,) = await _pipeline(codecs_json, spec).decode([(chunk_bytes, spec)])
        if spec.shape != chunk_spec.shape:
            decoded = decoded[tuple(slice(0, n) for n in chunk_spec.shape)]
        return decoded  # type: ignore[no-any-return]

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        """Encode a chunk with its case's chain, padded to the case's stored shape."""
        if self.select == "key":
            raise NotImplementedError(
                "With select='key' the switch codec is read-only: nothing says which "
                "case a new chunk belongs to."
            )
        spec, codecs_json, _ = self._resolve(self._case_at(self._position(chunk_spec)), chunk_spec)
        if spec.shape != chunk_spec.shape:
            padded = chunk_spec.prototype.nd_buffer.create(
                shape=spec.shape,
                dtype=chunk_spec.dtype.to_native_dtype(),
                order=chunk_spec.order,
                fill_value=spec.fill_value,
            )
            padded[tuple(slice(0, n) for n in chunk_spec.shape)] = chunk_array
            chunk_array = padded
        (encoded,) = await _pipeline(codecs_json, spec).encode([(chunk_array, spec)])
        return encoded  # type: ignore[no-any-return]

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
