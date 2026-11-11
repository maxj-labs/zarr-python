from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.codecs import BytesCodec, GzipCodec, ShardingCodec, SwitchCodec, ZstdCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore


def _copy_chunks(
    store: MemoryStore, src: str, dst: str, shift: int, drop: set[int] | None = None
) -> int:
    """Copy an array's stored chunks to another array's keys, moved along axis 0."""
    proto = default_buffer_prototype()
    n = 0
    for key in sorted(k for k in store._store_dict if k.startswith(f"{src}/c/")):
        first, *rest = key[len(src) + 3 :].split("/")
        if drop and int(first) in drop:
            continue
        buf = sync(store.get(key, proto))
        assert buf is not None
        sync(store.set("/".join([dst, "c", str(int(first) + shift), *rest]), buf))
        n = max(n, int(first) + 1)
    return n


def _codec_dicts(arr: zarr.Array[Any]) -> list[dict[str, Any]]:
    codecs: Any = arr.metadata.to_dict()["codecs"]
    return [dict(c) for c in codecs]


@pytest.fixture
def joined() -> tuple[zarr.Array[Any], np.ndarray[Any, Any], MemoryStore]:
    """Three sources joined along axis 0: gzip, then zstd in chunks whose last one is
    cut short, then uncompressed with a chunk left unstored and its own fill value."""
    rng = np.random.default_rng(0)
    store = MemoryStore()
    a = zarr.create_array(
        store,
        name="a",
        data=rng.integers(0, 99, (6, 5, 4)).astype("int32"),
        chunks=(2, 5, 4),
        compressors=GzipCodec(),
    )
    b = zarr.create_array(
        store,
        name="b",
        data=rng.integers(0, 99, (7, 5, 4)).astype("int32"),
        chunks=(3, 5, 4),
        compressors=ZstdCodec(),
        fill_value=0,
    )
    c_data = rng.integers(0, 99, (4, 5, 4)).astype("int32")
    c = zarr.create_array(
        store, name="c", data=c_data, chunks=(2, 5, 4), compressors=None, fill_value=-5
    )
    with zarr.config.set({"array.rectilinear_chunks": True}):
        j = zarr.create_array(
            store,
            name="j",
            shape=(17, 5, 4),
            chunks=[[2, 2, 2, 3, 3, 1, 2, 2], [5], [4]],
            dtype="int32",
            fill_value=0,
            serializer=SwitchCodec(
                axis=0,
                cases=[
                    {"chunks": 3, "codecs": _codec_dicts(a)},
                    {"chunks": 3, "codecs": _codec_dicts(b), "encoded_chunk_shape": [3, 5, 4]},
                    {"chunks": 2, "codecs": _codec_dicts(c), "fill_value": -5},
                ],
            ),
            compressors=None,
            filters=None,
        )
    _copy_chunks(store, "a", "j", 0)
    _copy_chunks(store, "b", "j", 3)
    _copy_chunks(store, "c", "j", 6, drop={1})  # the last chunk of `c` is not stored
    c_data = c_data.copy()
    c_data[2:] = -5
    expected = np.concatenate([a[:], b[:], c_data])
    return j, expected, store


@pytest.mark.parametrize(
    "selection",
    [
        (slice(None),),
        (slice(5, 8),),
        (12,),
        (slice(0, 17, 3), slice(1, 4), 2),
        (slice(11, 17),),
        (16, 4, 3),
    ],
)
def test_reads(
    joined: tuple[zarr.Array[Any], np.ndarray[Any, Any], MemoryStore], selection: tuple[Any, ...]
) -> None:
    arr, expected, _ = joined
    with zarr.config.set({"array.rectilinear_chunks": True}):
        np.testing.assert_array_equal(arr[selection], expected[selection])


def test_fancy_reads_and_reopen(
    joined: tuple[zarr.Array[Any], np.ndarray[Any, Any], MemoryStore],
) -> None:
    _, expected, store = joined
    with zarr.config.set({"array.rectilinear_chunks": True}):
        arr = zarr.open_array(store, path="j", mode="r")
        t, y, x = [0, 5, 6, 12, 16], [0, 4], [1, 3]
        oindex: Any = arr.oindex
        np.testing.assert_array_equal(oindex[t, y, x], expected[np.ix_(t, y, x)])
        np.testing.assert_array_equal(
            arr.vindex[t, [0, 1, 2, 3, 4], [0, 1, 2, 3, 0]],
            expected[t, [0, 1, 2, 3, 4], [0, 1, 2, 3, 0]],
        )
        codec = arr.metadata.to_dict()["codecs"]
        assert [c["name"] for c in codec] == ["switch"]  # type: ignore[index,union-attr]


def test_shards_are_read_by_range() -> None:
    """A case whose chain is a sharding codec reads only the inner chunks it needs."""
    rng = np.random.default_rng(1)
    store = MemoryStore()
    data = rng.random((8, 12)).astype("float32")
    src = zarr.create_array(
        store, name="s", data=data, shards=(8, 12), chunks=(2, 3), compressors=GzipCodec()
    )
    sharding: Any = _codec_dicts(src)[0]
    arr = zarr.create_array(
        store,
        name="j",
        shape=(6, 10),  # smaller than the shard: the stored shape is cropped
        chunks=(6, 10),
        dtype="float32",
        serializer=SwitchCodec(
            axis=0,
            cases=[
                {
                    "chunks": 1,
                    "codecs": [sharding],
                    "encoded_chunk_shape": [8, 12],
                    "coalesce_max_gap_bytes": 0,
                }
            ],
        ),
        compressors=None,
        filters=None,
    )
    assert isinstance(ShardingCodec.from_dict(sharding), ShardingCodec)
    _copy_chunks(store, "s", "j", 0)
    np.testing.assert_array_equal(arr[:], data[:6, :10])

    class Counting(zarr.storage.WrapperStore[MemoryStore]):
        nbytes = 0

        async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
            out = await self._store.get(key, prototype, byte_range)
            type(self).nbytes += 0 if out is None else len(out)
            return out

    counted = zarr.open_array(Counting(store), path="j", mode="r")
    Counting.nbytes = 0
    np.testing.assert_array_equal(counted[3, 4], data[3, 4])
    whole = len(store._store_dict["j/c/0/0"])
    assert 0 < Counting.nbytes < whole / 2


def test_must_be_the_only_codec() -> None:
    cases = [{"chunks": 1, "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]}]
    with pytest.raises(ValueError, match="only codec"):
        zarr.create_array(
            MemoryStore(),
            name="x",
            shape=(2,),
            chunks=(2,),
            dtype="int8",
            serializer=SwitchCodec(axis=0, cases=cases),
            compressors=GzipCodec(),
            filters=None,
        )
    assert isinstance(BytesCodec(), BytesCodec)


def test_writes_encode_each_chunk_with_its_case() -> None:
    """Writing through the array gives the bytes each case's own chain would write."""
    rng = np.random.default_rng(2)
    data = rng.integers(0, 50, size=(9, 4, 3)).astype("int32")
    store = MemoryStore()
    # references: the same data written by plain arrays, one per case
    a = zarr.create_array(store, name="a", data=data[:4], chunks=(2, 4, 3), compressors=GzipCodec())
    b = zarr.create_array(store, name="b", data=data[4:], chunks=(3, 4, 3), compressors=None)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        j = zarr.create_array(
            store,
            name="j",
            shape=(9, 4, 3),
            chunks=[[2, 2, 3, 2], [4], [3]],  # the last chunk of `b` is cut short
            dtype="int32",
            serializer=SwitchCodec(
                axis=0,
                cases=[
                    {"chunks": 2, "codecs": _codec_dicts(a)},
                    {"chunks": 2, "codecs": _codec_dicts(b), "encoded_chunk_shape": [3, 4, 3]},
                ],
            ),
            compressors=None,
            filters=None,
        )
        j[:] = data
        np.testing.assert_array_equal(j[:], data)
        j[3:6, 1, :] = -1  # a partial write: read, change, encode again
        data[3:6, 1, :] = -1
        np.testing.assert_array_equal(j[:], data)
    proto = default_buffer_prototype()
    # an uncompressed chunk of `b`'s case is stored at its full shape, cut or not
    for key in ("j/c/2/0/0", "j/c/3/0/0"):
        stored = sync(store.get(key, proto))
        assert stored is not None
        assert len(stored) == 3 * 4 * 3 * 4
    # a chunk of `a`'s case is what `a`'s chain writes
    a[:] = data[:4]
    one = sync(store.get("j/c/0/0/0", proto))
    two = sync(store.get("a/c/0/0/0", proto))
    assert one is not None
    assert two is not None
    assert one.to_bytes() == two.to_bytes()


@pytest.mark.parametrize(
    "cases",
    [
        [],
        [{"codecs": []}],
        [{"chunks": 0, "codecs": []}],
        [{"chunks": 1, "codecs": [], "other": 1}],
    ],
)
def test_invalid_cases(cases: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError):
        SwitchCodec(axis=0, cases=cases)
