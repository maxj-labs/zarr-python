from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.codecs import ArrayConcatCodec, ArrayConcatPiece, GzipCodec
from zarr.storage import MemoryStore


def _concat_array(
    store: MemoryStore,
    pieces: list[np.ndarray[Any, Any]],
    *,
    axis: int = 0,
    chunks: list[tuple[int, ...]] | None = None,
    compressors: list[Any] | None = None,
    path: str = "joined",
) -> tuple[zarr.Array[Any], np.ndarray[Any, Any]]:
    """Write `pieces` as arrays under `_pieces/` and an array at `path` that joins them."""
    specs = []
    for i, data in enumerate(pieces):
        zarr.create_array(
            store,
            name=f"_pieces/{path}/{i}",
            data=data,
            chunks=chunks[i] if chunks else "auto",
            compressors=compressors[i] if compressors else "auto",
        )
        depth = path.count("/") + 1
        specs.append(
            ArrayConcatPiece(path="../" * depth + f"_pieces/{path}/{i}", size=data.shape[axis])
        )
    expected = np.concatenate(pieces, axis=axis)
    arr = zarr.create_array(
        store,
        name=path,
        shape=expected.shape,
        chunks=expected.shape,
        dtype=expected.dtype,
        serializer=ArrayConcatCodec(axis=axis, pieces=specs),
        compressors=None,
        filters=None,
    )
    return arr, expected


@pytest.fixture
def mixed() -> tuple[zarr.Array[Any], np.ndarray[Any, Any]]:
    """Three pieces that share no chunk grid: different chunk shapes and codecs, and
    a piece whose last chunk along the joined axis is cut short."""
    rng = np.random.default_rng(0)
    pieces = [
        rng.integers(0, 1000, size=(7, 12, 10)).astype("float32"),
        rng.integers(0, 1000, size=(5, 12, 10)).astype("float32"),
        rng.integers(0, 1000, size=(9, 12, 10)).astype("float32"),
    ]
    return _concat_array(
        MemoryStore(),
        pieces,
        chunks=[(1, 12, 10), (2, 5, 5), (4, 3, 7)],
        compressors=[None, GzipCodec(level=1), None],
    )


@pytest.mark.parametrize(
    "selection",
    [
        (slice(None),),
        (slice(6, 8),),
        (slice(0, 21, 5), slice(2, 9), slice(None, None, 3)),
        (11, 3, 4),
        (slice(20, 21), 0),
        (slice(7, 12),),
        (slice(3, 3),),
        (-1,),
    ],
)
def test_basic_selection(
    mixed: tuple[zarr.Array[Any], np.ndarray[Any, Any]], selection: tuple[Any, ...]
) -> None:
    arr, expected = mixed
    np.testing.assert_array_equal(arr[selection], expected[selection])


def test_orthogonal_coordinate_and_mask_selection(
    mixed: tuple[zarr.Array[Any], np.ndarray[Any, Any]],
) -> None:
    arr, expected = mixed
    t, y, x = [0, 6, 7, 13, 20], [1, 11], [0, 4, 9]
    oindex: Any = arr.oindex
    np.testing.assert_array_equal(oindex[t, y, x], expected[np.ix_(t, y, x)])
    np.testing.assert_array_equal(oindex[t, slice(2, 5), 3], expected[t, 2:5, 3])
    tt, yy, xx = [20, 0, 7, 7, 12], [0, 11, 5, 5, 3], [9, 0, 2, 2, 7]
    np.testing.assert_array_equal(arr.vindex[tt, yy, xx], expected[tt, yy, xx])
    mask = expected > 900
    np.testing.assert_array_equal(arr.vindex[mask], expected[mask])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis(axis: int) -> None:
    rng = np.random.default_rng(1)
    shapes = [[6, 5, 4], [6, 5, 4]]
    shapes[0][axis], shapes[1][axis] = 3, 8
    pieces = [rng.random(tuple(s)) for s in shapes]
    arr, expected = _concat_array(MemoryStore(), pieces, axis=axis)
    np.testing.assert_array_equal(arr[:], expected)
    np.testing.assert_array_equal(arr[1:, 2:, 1:], expected[1:, 2:, 1:])


def test_nested_group_and_reopen() -> None:
    store = MemoryStore()
    pieces = [np.arange(6).reshape(3, 2), np.arange(6, 16).reshape(5, 2)]
    _, expected = _concat_array(store, pieces, path="a/b/joined")
    reopened = zarr.open_array(store, path="a/b/joined", mode="r")
    codecs: Any = reopened.metadata.to_dict()["codecs"]
    assert list(codecs) == [
        {
            "name": "array_concat",
            "configuration": {
                "axis": 0,
                "pieces": [
                    {"path": "../../../_pieces/a/b/joined/0", "size": 3},
                    {"path": "../../../_pieces/a/b/joined/1", "size": 5},
                ],
            },
        }
    ]
    np.testing.assert_array_equal(reopened[:], expected)


def test_piece_keeps_its_own_fill_value() -> None:
    store = MemoryStore()
    zarr.create_array(store, name="p0", shape=(2, 3), chunks=(1, 3), dtype="int16", fill_value=-1)
    zarr.create_array(store, name="p1", shape=(2, 3), chunks=(2, 3), dtype="int16", fill_value=7)
    arr = zarr.create_array(
        store,
        name="joined",
        shape=(4, 3),
        chunks=(4, 3),
        dtype="int16",
        fill_value=0,
        serializer=ArrayConcatCodec(
            axis=0, pieces=[{"path": "../p0", "size": 2}, {"path": "../p1", "size": 2}]
        ),
        compressors=None,
        filters=None,
    )
    np.testing.assert_array_equal(arr[:, 0], [-1, -1, 7, 7])


def test_fused_pipeline(mixed: tuple[zarr.Array[Any], np.ndarray[Any, Any]]) -> None:
    arr, expected = mixed
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}):
        reopened = zarr.open_array(arr.store, path="joined", mode="r")
        np.testing.assert_array_equal(reopened[2:19, 1:, ::2], expected[2:19, 1:, ::2])


def test_read_only(mixed: tuple[zarr.Array[Any], np.ndarray[Any, Any]]) -> None:
    arr, _ = mixed
    with pytest.raises(NotImplementedError, match="read-only"):
        arr[0] = 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"chunks": (2, 3)}, "one chunk"),
        ({"pieces": [{"path": "../p0", "size": 3}]}, "add up to 3"),
        ({"axis": 2}, "out of bounds"),
        ({"compressors": GzipCodec()}, "only codec"),
    ],
)
def test_invalid_metadata(kwargs: dict[str, Any], match: str) -> None:
    pieces = kwargs.pop("pieces", [{"path": "../p0", "size": 4}])
    axis = kwargs.pop("axis", 0)
    args: dict[str, Any] = {"chunks": (4, 3), "compressors": None} | kwargs
    with pytest.raises(ValueError, match=match):
        zarr.create_array(
            MemoryStore(),
            name="joined",
            shape=(4, 3),
            dtype="int16",
            serializer=ArrayConcatCodec(axis=axis, pieces=pieces),
            filters=None,
            **args,
        )


def test_piece_mismatch_and_escape() -> None:
    store = MemoryStore()
    zarr.create_array(store, name="p0", shape=(2, 5), chunks=(1, 5), dtype="int16")

    def joined(name: str, path: str) -> zarr.Array[Any]:
        return zarr.create_array(
            store,
            name=name,
            shape=(2, 3),
            chunks=(2, 3),
            dtype="int16",
            serializer=ArrayConcatCodec(axis=0, pieces=[{"path": path, "size": 2}]),
            compressors=None,
            filters=None,
        )

    with pytest.raises(ValueError, match="has shape"):
        joined("bad_shape", "../p0")[:]
    with pytest.raises(ValueError, match="leaves the store"):
        joined("escapes", "../../p0")[:]
