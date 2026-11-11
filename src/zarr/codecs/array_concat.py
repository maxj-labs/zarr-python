"""A read-only array-to-bytes codec that presents several arrays as one.

An array that uses this codec stores no chunks of its own. Its single chunk
spans the whole array, and the codec resolves a read of any part of that chunk
by reading the matching parts of other arrays in the same store (its
*pieces*), joined end to end along one axis.

Each piece is an ordinary Zarr array with its own chunk grid, codecs and fill
value. So arrays that cannot be merged into one chunk grid (because their
chunk shapes or codecs differ, or because a chunk of one is cut short where
the next begins) can still be read as one array.
"""

from __future__ import annotations

import asyncio
import posixpath
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from zarr.abc.codec import ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.indexing import get_indexer
from zarr.core.metadata.v3 import RegularChunkGridMetadata

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.store import ByteGetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
    from zarr.core.indexing import SelectorTuple
    from zarr.core.metadata.v3 import ChunkGridMetadata

__all__ = ["ArrayConcatCodec", "ArrayConcatPiece"]

CODEC_NAME: Final = "array_concat"

# A chunk's key, in the `default` and `v2` chunk key encodings with either separator.
_CHUNK_KEY: Final = re.compile(r"(?:^|/)(?:c(?:[/.]\d+)+|\d+(?:[/.]\d+)*)$")


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class ArrayConcatPiece:
    """One piece of a concatenated array.

    Parameters
    ----------
    path : str
        The path of the piece's array node, relative to the node of the array
        that uses the codec (so a sibling array ``b`` is ``"../b"``).
    size : int
        The length of the piece along the concatenation axis.
    """

    path: str
    size: int

    def __post_init__(self) -> None:
        path: object = self.path
        if not isinstance(path, str) or path == "":
            raise TypeError(f"A piece's path must be a non-empty string. Got {path!r}.")
        if not _is_count(self.size):
            raise TypeError(f"A piece's size must be a non-negative integer. Got {self.size!r}.")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if not isinstance(data, dict) or set(data) != {"path", "size"}:
            raise ValueError(
                f"A piece must be an object with the keys 'path' and 'size'. Got {data!r}."
            )
        return cls(path=data["path"], size=data["size"])

    def to_dict(self) -> dict[str, JSON]:
        return {"path": self.path, "size": self.size}


def _shift(
    selection: tuple[Any, ...], origin: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> tuple[Any, ...]:
    """A selection of a chunk, as a selection of the array the chunk starts at `origin` in."""
    out: list[Any] = []
    for sel, start, length in zip(selection, origin, chunk_shape, strict=True):
        if isinstance(sel, slice):
            lo, hi, step = sel.indices(length)
            out.append(slice(lo + start, hi + start, step))
        elif np.ndim(sel) == 0:
            out.append(int(sel) + start)
        else:
            arr = np.asarray(sel)
            if arr.dtype == bool:
                arr = np.nonzero(arr)[0]
            out.append(arr + start)
    return tuple(out)


def _array_path(chunk_key: str) -> str:
    """The path of the array node, from the key of one of its chunks."""
    path, n = _CHUNK_KEY.subn("", chunk_key)
    if n != 1:
        raise ValueError(
            f"Cannot tell the array's path from the chunk key {chunk_key!r}. The "
            f"{CODEC_NAME!r} codec needs the 'default' or 'v2' chunk key encoding."
        )
    return path


def _resolve(array_path: str, piece_path: str) -> str:
    resolved = posixpath.normpath(posixpath.join(array_path, piece_path))
    if resolved == "." or resolved.startswith(("../", "/")) or resolved == "..":
        raise ValueError(
            f"The piece path {piece_path!r} of the array at {array_path!r} leaves the store."
        )
    return resolved


def _bounding_box(
    selection: tuple[Any, ...], shape: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[Any, ...]]:
    """Split a selection of a piece into the box to fetch and the selection of that box.

    ``piece[selection]`` equals ``piece[box][relative]``.
    """
    box: list[slice] = []
    relative: list[Any] = []
    for sel, length in zip(selection, shape, strict=True):
        if isinstance(sel, slice):
            start, stop, step = sel.indices(length)
            if step < 1:
                raise IndexError("Negative steps are not supported.")
            box.append(slice(start, max(start, stop)))
            relative.append(slice(None, None, step))
        else:
            arr = np.asarray(sel)
            if arr.dtype == bool:
                box.append(slice(0, length))
                relative.append(arr)
            elif arr.size == 0:
                box.append(slice(0, 0))
                relative.append(arr)
            else:
                lo, hi = int(arr.min()), int(arr.max())
                box.append(slice(lo, hi + 1))
                relative.append(arr - lo)
    return tuple(box), tuple(relative)


@dataclass(frozen=True)
class ArrayConcatCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """Read other arrays of the same store, joined along one axis, as one array.

    The array that uses this codec MUST have a regular chunk grid and MUST NOT
    have other codecs. It stores no chunks, so its chunk shape is only what it
    reports to readers: any shape works, and one that lines up with the pieces'
    own chunks lets a reader that works chunk by chunk read each stored chunk
    once. It is read-only: its contents change when its pieces do.

    Parameters
    ----------
    axis : int
        The axis along which the pieces are joined.
    pieces : sequence of ArrayConcatPiece
        The pieces, in order. Every piece has the array's data type and its
        shape on every other axis.
    """

    is_fixed_size = False
    needs_chunk_coords = True

    axis: int
    pieces: tuple[ArrayConcatPiece, ...]

    def __init__(self, *, axis: int, pieces: Any) -> None:
        if not _is_count(axis):
            raise TypeError(f"axis must be a non-negative integer. Got {axis!r}.")
        pieces_parsed = tuple(
            p if isinstance(p, ArrayConcatPiece) else ArrayConcatPiece.from_dict(p) for p in pieces
        )
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "pieces", pieces_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration = parse_named_configuration(data, CODEC_NAME)
        if set(configuration) != {"axis", "pieces"}:
            raise ValueError(
                f"The {CODEC_NAME!r} codec's configuration has the keys 'axis' and 'pieces'. "
                f"Got {sorted(configuration)!r}."
            )
        return cls(axis=configuration["axis"], pieces=configuration["pieces"])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": CODEC_NAME,
            "configuration": {"axis": self.axis, "pieces": [p.to_dict() for p in self.pieces]},
        }

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        if self.axis >= len(shape):
            raise ValueError(f"axis {self.axis} is out of bounds for an array of shape {shape}.")
        if not isinstance(chunk_grid, RegularChunkGridMetadata):
            raise ValueError(
                f"An array with the {CODEC_NAME!r} codec has a regular chunk grid. "
                f"Got {chunk_grid!r}."
            )
        total = sum(p.size for p in self.pieces)
        if total != shape[self.axis]:
            raise ValueError(
                f"The pieces' sizes add up to {total}, but the array has length "
                f"{shape[self.axis]} along axis {self.axis}."
            )

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError(f"An array with the {CODEC_NAME!r} codec stores no chunks.")

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        raise ValueError(
            f"An array with the {CODEC_NAME!r} codec stores no chunks, but bytes were found "
            "at its chunk key."
        )

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        raise NotImplementedError(
            f"An array with the {CODEC_NAME!r} codec is read-only. Write to its pieces."
        )

    async def _decode_partial_single(
        self, byte_getter: ByteGetter, selection: SelectorTuple, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        from zarr.core.array import AsyncArray
        from zarr.storage._common import StorePath

        if not isinstance(byte_getter, StorePath):
            raise TypeError(
                f"The {CODEC_NAME!r} codec reads its pieces from the array's store, so it "
                f"cannot be nested in another codec. Got {type(byte_getter)!r}."
            )
        # Where the requested chunk starts in the array. Chunks of a regular grid
        # all have the grid's shape, so the origin follows from the coordinates.
        coords = chunk_spec.chunk_coords or (0,) * len(chunk_spec.shape)
        origin = tuple(int(c) * int(n) for c, n in zip(coords, chunk_spec.shape, strict=True))
        array_path = _array_path(byte_getter.path)
        selection = _shift(
            selection if isinstance(selection, tuple) else (selection,), origin, chunk_spec.shape
        )
        # An extent that holds the selection: the pieces' total along the axis, and
        # up to the end of this chunk elsewhere.
        shape = tuple(
            sum(p.size for p in self.pieces) if d == self.axis else int(o + n)
            for d, (o, n) in enumerate(zip(origin, chunk_spec.shape, strict=True))
        )

        # The pieces tile the array along `axis`, so a chunk grid over them
        # splits any selection of the array into selections of pieces.
        sizes: list[int | list[int]] = list(shape)
        sizes[self.axis] = [p.size for p in self.pieces]
        indexer = get_indexer(selection, shape=shape, chunk_grid=ChunkGrid.from_sizes(shape, sizes))
        out = chunk_spec.prototype.nd_buffer.empty(
            shape=indexer.shape,
            dtype=chunk_spec.dtype.to_native_dtype(),
            order=chunk_spec.order,
        )
        native_dtype = chunk_spec.dtype.to_native_dtype()

        async def read_piece(index: int, piece_selection: SelectorTuple) -> NDBuffer:
            piece = self.pieces[index]
            path = _resolve(array_path, piece.path)
            array = await AsyncArray.open(StorePath(byte_getter.store, path), zarr_format=3)
            piece_shape = array.shape
            if len(piece_shape) != len(shape) or piece_shape[self.axis] != piece.size:
                raise ValueError(
                    f"The piece at {path!r} has shape {array.shape}, but the array that "
                    f"joins it expects length {piece.size} along axis {self.axis}."
                )
            if array.dtype != native_dtype:
                raise ValueError(
                    f"The piece at {path!r} has data type {array.dtype}, but the array that "
                    f"joins it has {native_dtype}."
                )
            sel = piece_selection if isinstance(piece_selection, tuple) else (piece_selection,)
            box, relative = _bounding_box(sel, piece_shape)
            values = await array.getitem(box, prototype=chunk_spec.prototype)
            return chunk_spec.prototype.nd_buffer.from_ndarray_like(np.asarray(values)[relative])

        parts = [(coords[self.axis], sel, out_sel) for coords, sel, out_sel, _ in indexer]
        results = await asyncio.gather(*(read_piece(i, sel) for i, sel, _ in parts))
        for (_, _, out_selection), values in zip(parts, results, strict=True):
            out[out_selection] = values

        if hasattr(indexer, "sel_shape"):
            return out.reshape(indexer.sel_shape)
        return out
