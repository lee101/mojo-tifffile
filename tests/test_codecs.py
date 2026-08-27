import numpy as np
import pytest

import imagecodecs
import mojo_tifffile as mt
from mojo_tifffile._lib import lib
from mojo_tifffile.codecs import _predictor_copy_into


@pytest.mark.parametrize("size", [0, 1, 2, 3, 127, 128, 129, 4097])
def test_packbits_roundtrip_and_imagecodecs_parity(size):
    rng = np.random.default_rng(size)
    data = rng.integers(0, 8, size=size, dtype=np.uint8).tobytes()
    encoded = mt.packbits_encode(data)
    assert mt.packbits_decode(encoded, size) == data
    assert imagecodecs.packbits_decode(encoded) == data
    reference = imagecodecs.packbits_encode(data)
    assert mt.packbits_decode(reference, size) == data


def test_packbits_published_control_byte_cases():
    encoded = b"\x02ABC\xfeZ\x80\x00Q"
    assert mt.packbits_decode(encoded, 7) == b"ABCZZZQ"


@pytest.mark.parametrize("size", [3, 4, 5, 127, 128, 129, 133])
def test_packbits_literal_simd_tails(size):
    data = bytes(i % 251 for i in range(size))
    encoded = mt.packbits_encode(data)
    assert imagecodecs.packbits_decode(encoded) == data
    assert mt.packbits_decode(encoded, size) == data


def test_packbits_rejects_truncated_stream():
    with pytest.raises(ValueError, match="truncated"):
        mt.packbits_decode(b"\x04abc", 5)
    with pytest.raises(ValueError, match="too small"):
        mt.packbits_decode(b"\x00A", 0)
    assert mt.packbits_decode(b"\x80", 0) == b""


@pytest.mark.parametrize("dtype", ["u1", "<u2", ">u2", "<u4", ">u8"])
@pytest.mark.parametrize("samples", [1, 3, 4])
def test_predictor_matches_imagecodecs(dtype, samples):
    rng = np.random.default_rng(1)
    array = rng.integers(0, 128, size=(7, 31, samples), dtype=np.uint64).astype(dtype)
    byteorder = ">" if array.dtype.byteorder == ">" else "<"
    encoded = mt.delta_encode(
        array.tobytes(), array.shape, array.dtype.itemsize, byteorder
    )
    reference = imagecodecs.delta_encode(array, axis=-2)
    assert encoded == reference.tobytes()
    assert mt.delta_decode(encoded, array.shape, array.dtype.itemsize, byteorder) == array.tobytes()


@pytest.mark.parametrize("width", [4, 5, 7, 8, 9])
@pytest.mark.parametrize("samples", [1, 3])
def test_predictor_simd_remainders(width, samples):
    array = np.arange(3 * width * samples, dtype=np.uint16).reshape(
        3, width, samples
    )
    encoded = mt.delta_encode(array, array.shape, array.dtype.itemsize)
    assert isinstance(encoded, bytes)
    assert encoded == imagecodecs.delta_encode(array, axis=-2).tobytes()
    assert mt.delta_decode(encoded, array.shape, array.dtype.itemsize) == array.tobytes()


@pytest.mark.parametrize("rows", [1023, 1025])
def test_predictor_parallel_threshold(rows):
    rng = np.random.default_rng(11)
    array = rng.integers(
        0, 65536, size=(rows, 2049, 1), dtype=np.uint16
    )
    encoded = mt.delta_encode(array, array.shape, array.dtype.itemsize)
    assert encoded == imagecodecs.delta_encode(array, axis=-2).tobytes()
    assert mt.delta_decode(encoded, array.shape, array.dtype.itemsize) == array.tobytes()


def test_bitorder_is_an_involution():
    data = bytes(range(256))
    assert mt.bitorder_decode(mt.bitorder_decode(data)) == data
    assert mt.bitorder_decode(b"\x01\x80\x55") == b"\x80\x01\xaa"


def test_predictor_rejects_invalid_layouts_and_overlapping_views():
    with pytest.raises(ValueError, match="samples positive"):
        mt.delta_encode(b"", (1, 1, 0), 1)
    with pytest.raises(ValueError, match="itemsize"):
        mt.delta_encode(b"\0" * 3, (1, 1, 1), 3)
    with pytest.raises(ValueError, match="byteorder"):
        mt.delta_encode(b"\0", (1, 1, 1), 1, "=")
    data = bytearray(range(16))
    with pytest.raises(ValueError, match="overlap"):
        _predictor_copy_into(memoryview(data)[:8], memoryview(data)[1:9], (1, 8, 1), 1, "<", False)


def test_mojo_ffi_rejects_null_nonempty_buffers():
    kernels = lib()
    assert kernels.mti_predictor(0, 1, 1, 1, 1, 0, 1) < 0
    assert kernels.mti_predictor_copy(0, 0, 1, 1, 1, 1, 0, 1) < 0
    assert kernels.mti_packbits_encode(0, 1, 0, 2) < 0
    assert kernels.mti_packbits_decode(0, 1, 0, 1) < 0
    assert kernels.mti_reverse_bits(0, 1) < 0
