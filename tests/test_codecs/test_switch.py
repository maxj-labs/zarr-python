"""Tests for the experimental `switch` and `crop` codecs (zarr.codecs.switch).

Every chunk is written by an ordinary array with the case's own codecs, then copied
under a `switch` array with the case's header in front, as a virtual reference would
serve it.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import zarr
from zarr.abc.store import ByteRequest, RangeByteRequest
from zarr.codecs.switch import CropCodec, SwitchCodec, case_key
from zarr.core.buffer import Buffer, BufferPrototype, cpu
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

GZIP = [
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "gzip", "configuration": {"level": 5}},
]
RAW = [{"name": "bytes", "configuration": {"endian": "little"}}]


def array_doc(shape: list[int], chunk_grid: dict[str, Any], codecs: list[Any], **kw: Any) -> bytes:
    doc = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": kw.get("data_type", "float32"),
        "chunk_grid": chunk_grid,
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": kw.get("fill_value", "NaN"),
        "codecs": codecs,
        "attributes": {},
    }
    return json.dumps(doc).encode()


def regular(chunks: list[int]) -> dict[str, Any]:
    return {"name": "regular", "configuration": {"chunk_shape": chunks}}


def put(store: MemoryStore, key: str, value: bytes) -> None:
    sync(store.set(key, cpu.Buffer.from_bytes(value)))


def raw(store: MemoryStore, key: str) -> bytes:
    buf = sync(store.get(key, default_buffer_prototype()))
    assert buf is not None
    return buf.to_bytes()


def encoded_chunk(
    codecs: list[Any], data: np.ndarray, *, stored: tuple[int, ...] | None = None
) -> bytes:
    """`data` encoded by an ordinary one-chunk array with `codecs`. A `stored` shape
    larger than `data` gives the chunk HDF5's whole-stored edge chunk."""
    stored = stored or data.shape
    store = MemoryStore()
    put(
        store,
        "zarr.json",
        array_doc(
            list(data.shape), regular(list(stored)), codecs, data_type=data.dtype.name, fill_value=0
        ),
    )
    zarr.open_array(store, mode="r+")[...] = data
    return raw(store, "c/" + "/".join("0" * data.ndim))


def header(case: dict[str, Any], values: tuple[float, ...] = ()) -> bytes:
    return case_key(case) + b"".join(
        np.array([v], dtype=np.dtype(d).newbyteorder("<")).tobytes()
        for d, v in zip(case["parameters"], values, strict=True)
    )


def switch(*cases: dict[str, Any]) -> dict[str, Any]:
    return {"name": "switch", "configuration": {"cases": {case_key(c).hex(): c for c in cases}}}


def test_metadata_round_trip() -> None:
    case = {"codecs": GZIP, "parameters": []}
    codec = SwitchCodec.from_dict(switch(case))
    assert codec.to_dict() == switch(case)
    assert zarr.registry.get_codec_class("switch") is SwitchCodec
    assert zarr.registry.get_codec_class("crop") is CropCodec
    crop = CropCodec(shape=[4, 6])
    assert CropCodec.from_dict(crop.to_dict()) == crop


def test_refuses_a_key_that_is_not_the_hash_of_its_case() -> None:
    with pytest.raises(ValueError, match="is not the key of its contents"):
        SwitchCodec(cases={"00" * 8: {"codecs": GZIP, "parameters": []}})


def test_chunks_choose_their_chain() -> None:
    """Two codec eras in one array: gzip, then uncompressed."""
    data = np.arange(16, dtype="float32").reshape(4, 4)
    gz, plain = ({"codecs": c, "parameters": []} for c in (GZIP, RAW))
    store = MemoryStore()
    put(store, "zarr.json", array_doc([4, 4], regular([2, 4]), [switch(gz, plain)]))
    put(store, "c/0/0", header(gz) + encoded_chunk(GZIP, data[:2]))
    put(store, "c/1/0", header(plain) + encoded_chunk(RAW, data[2:]))
    np.testing.assert_array_equal(zarr.open_array(store, mode="r")[...], data)


def test_parameters_fill_open_values() -> None:
    """One case, CF-style per-chunk scale and offset as parameters."""
    case = {
        "codecs": [
            {
                "name": "scale_offset",
                "configuration": {"offset": {"parameter": 0}, "scale": {"parameter": 1}},
            },
            *RAW,
        ],
        "parameters": ["float64", "float64"],
    }
    store = MemoryStore()
    put(
        store,
        "zarr.json",
        array_doc([4], regular([2]), [switch(case)], data_type="float64", fill_value=0),
    )
    put(store, "c/0", header(case, (10.0, 2.0)) + np.array([2.0, 4.0]).tobytes())  # /2 +10
    put(store, "c/1", header(case, (-1.0, 0.5)) + np.array([3.0, 4.0]).tobytes())  # /0.5 -1
    np.testing.assert_array_equal(zarr.open_array(store, mode="r")[...], [11.0, 12.0, 5.0, 7.0])


def test_crop_reads_whole_stored_edge_chunks_in_a_rectilinear_grid() -> None:
    """Files of 6 steps chunked by 4 (cells 4, 2, 4, 2), every chunk stored whole."""
    data = np.arange(12, dtype="float32")
    case = {"codecs": [{"name": "crop", "configuration": {"shape": [4]}}, *GZIP], "parameters": []}
    grid = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [[4, 2, 4, 2]]},
    }
    store = MemoryStore()
    put(store, "zarr.json", array_doc([12], grid, [switch(case)]))
    for i, (start, stop) in enumerate([(0, 4), (4, 6), (6, 10), (10, 12)]):
        put(store, f"c/{i}", header(case) + encoded_chunk(GZIP, data[start:stop], stored=(4,)))
    with zarr.config.set({"array.rectilinear_chunks": True}):
        np.testing.assert_array_equal(zarr.open_array(store, mode="r")[...], data)


def test_unknown_case_is_an_error() -> None:
    case = {"codecs": RAW, "parameters": []}
    other = {"codecs": GZIP, "parameters": []}
    store = MemoryStore()
    put(store, "zarr.json", array_doc([2], regular([2]), [switch(case)]))
    put(store, "c/0", header(other) + encoded_chunk(GZIP, np.ones(2, dtype="float32")))
    with pytest.raises(ValueError, match="which the array does not list"):
        zarr.open_array(store, mode="r")[...]


def test_sharding_inside_a_case_reads_through_the_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """A case whose chain is `sharding_indexed`: a full read sees the shard without its
    header, and a partial read fetches the index and one inner chunk, every request
    shifted past the header."""
    requests: list[tuple[str, ByteRequest | None]] = []
    get = MemoryStore.get

    async def recording_get(
        self: MemoryStore,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        requests.append((key, byte_range))
        return await get(self, key, prototype, byte_range)

    monkeypatch.setattr(MemoryStore, "get", recording_get)
    data = np.arange(64, dtype="float32").reshape(8, 8)
    for index_location in ("end", "start"):
        shard = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [4, 4],
                    "codecs": GZIP,
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                    "index_location": index_location,
                },
            }
        ]
        case = {"codecs": shard, "parameters": []}
        body = encoded_chunk(shard, data)
        store = MemoryStore()
        put(store, "zarr.json", array_doc([8, 8], regular([8, 8]), [switch(case)]))
        put(store, "c/0/0", header(case) + body)
        a = zarr.open_array(store, mode="r")
        np.testing.assert_array_equal(a[...], data)

        requests.clear()
        np.testing.assert_array_equal(a[1:3, 5:7], data[1:3, 5:7])
        ranges = [r for k, r in requests if k == "c/0/0"]
        assert ranges[0] == RangeByteRequest(0, 8)  # the case key
        assert None not in ranges  # never the whole shard
        fetched = sum(r.end - r.start for r in ranges if isinstance(r, RangeByteRequest))
        assert fetched < len(body) // 2


def test_partial_read_without_partial_decode_reads_the_body() -> None:
    data = np.arange(16, dtype="float32").reshape(4, 4)
    case = {"codecs": GZIP, "parameters": []}
    store = MemoryStore()
    put(store, "zarr.json", array_doc([4, 4], regular([4, 4]), [switch(case)]))
    put(store, "c/0/0", header(case) + encoded_chunk(GZIP, data))
    np.testing.assert_array_equal(zarr.open_array(store, mode="r")[1:3, 2], data[1:3, 2])


def test_switch_does_not_encode() -> None:
    case = {"codecs": RAW, "parameters": []}
    store = MemoryStore()
    put(store, "zarr.json", array_doc([2], regular([2]), [switch(case)]))
    with pytest.raises(NotImplementedError, match="switch decodes only"):
        zarr.open_array(store, mode="r+")[...] = 1.0


def test_cached_chain_equals_a_fresh_parse_and_evolve() -> None:
    """Only codecs with open values are built per chunk; the result must be what parsing
    the filled-in chain and evolving it would give."""
    from zarr.codecs.switch import _chain, _fill
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.chunk_utils import evolve_codecs
    from zarr.core.dtype import Float32
    from zarr.core.metadata.v3 import parse_codecs

    codecs = [
        {
            "name": "scale_offset",
            "configuration": {"offset": {"parameter": 0}, "scale": {"parameter": 1}},
        },
        {
            "name": "cast_value",
            "configuration": {
                "data_type": "int16",
                "scalar_map": {"encode": [["NaN", -1]], "decode": [[-1, "NaN"]]},
            },
        },
        *GZIP,
    ]
    spec = ArraySpec(
        shape=(4, 6),
        dtype=Float32(),
        fill_value=np.float32("nan"),
        config=ArrayConfig.from_dict({}),
        prototype=default_buffer_prototype(),
    )
    for values in [(280.0, 100.0), (270.0, 50.0), (280.0, 100.0), (float("nan"), 2.0)]:
        want = evolve_codecs(parse_codecs(_fill(codecs, values)), spec)
        assert _chain(json.dumps(codecs), values, spec) == want

    # one literal argument and one open one
    mixed = [
        {"name": "scale_offset", "configuration": {"offset": 5, "scale": {"parameter": 0}}},
        *GZIP,
    ]
    for one in [(2.0,), (0.5,)]:
        want = evolve_codecs(parse_codecs(_fill(mixed, one)), spec)
        assert _chain(json.dumps(mixed), one, spec) == want


def test_scale_offset_takes_the_field_update_and_others_reparse() -> None:
    from zarr.codecs.scale_offset import ScaleOffset
    from zarr.codecs.switch import _case_codecs, _Open

    so = {"name": "scale_offset", "configuration": {"offset": {"parameter": 0}, "scale": 4}}
    crop = {"name": "crop", "configuration": {"shape": [{"parameter": 0}]}}
    nested = {"name": "scale_offset", "configuration": {"offset": [{"parameter": 0}]}}
    built = _case_codecs(json.dumps([so, crop, nested, *GZIP]))
    assert isinstance(built[0], _Open)
    assert built[0].cls is ScaleOffset
    assert built[0].fixed == (("scale", 4),)
    assert built[0].slots == (("offset", 0),)
    assert isinstance(built[1], _Open)
    assert built[1].cls is None  # not a plain-argument codec
    assert isinstance(built[2], _Open)
    assert built[2].cls is None  # not a whole argument
    assert built[0].build((3.0,)) == ScaleOffset(offset=3.0, scale=4)
    assert built[0].build((float("nan"),)) == ScaleOffset(offset="NaN", scale=4)


def test_a_parameter_that_changes_downstream_specs() -> None:
    """`crop`'s stored shape as a parameter: each chunk's later codecs see a different
    shape, so their cached forms must be keyed on what they see."""
    data = np.arange(6, dtype="float32")
    case = {
        "codecs": [{"name": "crop", "configuration": {"shape": [{"parameter": 0}]}}, *GZIP],
        "parameters": ["int64"],
    }
    store = MemoryStore()
    put(store, "zarr.json", array_doc([6], regular([3]), [switch(case)]))
    put(store, "c/0", header(case, (4,)) + encoded_chunk(GZIP, data[0:3], stored=(4,)))
    put(store, "c/1", header(case, (5,)) + encoded_chunk(GZIP, data[3:6], stored=(5,)))
    np.testing.assert_array_equal(zarr.open_array(store, mode="r")[...], data)
