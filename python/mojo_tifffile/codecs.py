"""TIFF codec primitives backed by Mojo."""

from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._lib import lib

_new_bytes = ctypes.pythonapi.PyBytes_FromStringAndSize
_new_bytes.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
_new_bytes.restype = ctypes.py_object
_bytes_address = ctypes.pythonapi.PyBytes_AsString
_bytes_address.argtypes = [ctypes.py_object]
_bytes_address.restype = ctypes.c_void_p
_PARALLEL_PREDICTOR_BYTES = 4 * 1024 * 1024
_MAX_PREDICTOR_WORKERS = 8


def _buffer(data: bytes | bytearray | memoryview, *, writable: bool = False) -> np.ndarray:
    try:
        view = memoryview(data)
    except TypeError as exc:
        raise TypeError("expected a bytes-like object") from exc
    if not view.c_contiguous:
        raise ValueError("buffer must be C-contiguous")
    array = np.frombuffer(view, dtype=np.uint8)
    return array.copy() if writable and not array.flags.writeable else array


def _predictor_layout(shape, itemsize):
    try:
        rows, width, samples = (int(i) for i in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("shape must contain exactly three integers") from exc
    if rows < 0 or width < 0 or samples <= 0:
        raise ValueError("rows and width must be non-negative and samples positive")
    if itemsize not in (1, 2, 4, 8):
        raise ValueError("itemsize must be 1, 2, 4, or 8")
    expected = rows * width * samples * itemsize
    if expected > np.iinfo(np.int64).max:
        raise OverflowError("predictor buffer is too large for the Mojo ABI")
    return rows, width, samples, expected


def _byteorder(value):
    if value not in ("<", ">"):
        raise ValueError("byteorder must be '<' or '>'")
    return value


def packbits_encode(data: bytes | bytearray | memoryview) -> bytes:
    """Return TIFF PackBits encoding of a bytes-like object."""
    return _packbits_encode_buffer(data).tobytes()


def _packbits_encode_buffer(data):
    source = _buffer(data)
    if not source.size:
        return np.empty(0, dtype=np.uint8)
    destination = np.empty(source.size + (source.size + 127) // 128 + 2, dtype=np.uint8)
    size = lib().mti_packbits_encode(
        source.ctypes.data, source.size, destination.ctypes.data, destination.size
    )
    if size < 0:
        raise RuntimeError(f"PackBits encoder failed with status {size}")
    return destination[:size]


def packbits_decode(
    data: bytes | bytearray | memoryview, out: int | bytearray | None = None
) -> bytes | bytearray:
    """Decode TIFF PackBits data.

    ``out`` is either the exact decoded byte count or a destination bytearray.
    """
    source = _buffer(data)
    if isinstance(out, bytearray):
        destination = np.frombuffer(out, dtype=np.uint8)
    elif isinstance(out, int):
        destination = np.empty(out, dtype=np.uint8)
    else:
        raise TypeError("out must be the decoded byte count or a bytearray")
    size = lib().mti_packbits_decode(
        source.ctypes.data, source.size, destination.ctypes.data, destination.size
    )
    if size < 0:
        reason = "output is too small" if size == -2 else "input is truncated"
        raise ValueError(f"invalid PackBits stream: {reason}")
    if isinstance(out, bytearray):
        if size != len(out):
            del out[size:]
        return out
    return destination[:size].tobytes()


def delta_encode(
    data: bytes | bytearray | memoryview,
    shape: tuple[int, int, int],
    itemsize: int,
    byteorder: str = "<",
) -> bytes:
    """Apply TIFF horizontal differencing to rows of interleaved samples."""
    return _predictor(data, shape, itemsize, byteorder, False)


def delta_decode(
    data: bytes | bytearray | memoryview,
    shape: tuple[int, int, int],
    itemsize: int,
    byteorder: str = "<",
) -> bytes:
    """Reverse TIFF horizontal differencing."""
    return _predictor(data, shape, itemsize, byteorder, True)


def _predictor(data, shape, itemsize, byteorder, decode):
    rows, width, samples, expected = _predictor_layout(shape, itemsize)
    byteorder = _byteorder(byteorder)
    source = _buffer(data)
    if source.size != expected:
        raise ValueError(f"buffer has {source.size} bytes, expected {expected}")
    if not source.size:
        return b""
    result = _new_bytes(None, source.size)
    _predictor_copy_addresses(
        source.ctypes.data,
        _bytes_address(result),
        rows,
        width,
        samples,
        itemsize,
        int(decode),
        int(byteorder != ">"),
    )
    return result


def _predictor_copy_addresses(
    source, destination, rows, width, samples, itemsize, decode, little
):
    kernels = lib()
    rowbytes = width * samples * itemsize
    workers = (
        min(_MAX_PREDICTOR_WORKERS, rows)
        if rows * rowbytes >= _PARALLEL_PREDICTOR_BYTES
        else 1
    )

    def process(worker):
        y0 = worker * rows // workers
        y1 = (worker + 1) * rows // workers
        offset = y0 * rowbytes
        status = kernels.mti_predictor_copy(
            source + offset,
            destination + offset,
            y1 - y0,
            width,
            samples,
            itemsize,
            decode,
            little,
        )
        if status:
            raise ValueError(f"unsupported predictor layout (status {status})")

    if workers == 1:
        process(0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(process, range(workers)))


def _predictor_inplace(data, shape, itemsize, byteorder, decode):
    destination = _buffer(data)
    if not destination.flags.writeable:
        raise TypeError("predictor destination is not writable")
    _predictor_buffer(destination, shape, itemsize, byteorder, decode)


def _predictor_copy_into(data, destination, shape, itemsize, byteorder, decode):
    source = _buffer(data)
    target = _buffer(destination)
    if not target.flags.writeable:
        raise TypeError("predictor destination is not writable")
    rows, width, samples, expected = _predictor_layout(shape, itemsize)
    byteorder = _byteorder(byteorder)
    if source.size != expected or target.size != expected:
        raise ValueError(
            f"predictor buffers have {source.size} and {target.size} bytes, "
            f"expected {expected}"
        )
    if not expected:
        return
    if np.shares_memory(source, target):
        if source.ctypes.data != target.ctypes.data:
            raise ValueError("predictor source and destination overlap")
        _predictor_buffer(target, shape, itemsize, byteorder, decode)
        return
    _predictor_copy_addresses(
        source.ctypes.data,
        target.ctypes.data,
        rows,
        width,
        samples,
        itemsize,
        int(decode),
        int(byteorder != ">"),
    )


def _predictor_buffer(destination, shape, itemsize, byteorder, decode):
    rows, width, samples, expected = _predictor_layout(shape, itemsize)
    byteorder = _byteorder(byteorder)
    if destination.size != expected:
        raise ValueError(f"buffer has {destination.size} bytes, expected {expected}")
    if not destination.size:
        return
    status = lib().mti_predictor(
        destination.ctypes.data,
        rows,
        width,
        samples,
        itemsize,
        int(decode),
        int(byteorder != ">"),
    )
    if status:
        raise ValueError(f"unsupported predictor layout (status {status})")


def _packbits_decode_into(data, destination):
    source = _buffer(data)
    target = _buffer(destination)
    if not target.flags.writeable:
        raise TypeError("PackBits destination is not writable")
    if np.shares_memory(source, target):
        raise ValueError("PackBits source and destination overlap")
    size = lib().mti_packbits_decode(
        source.ctypes.data, source.size, target.ctypes.data, target.size
    )
    if size < 0:
        reason = "output is too small" if size == -2 else "input is truncated"
        raise ValueError(f"invalid PackBits stream: {reason}")
    return size


def bitorder_decode(data: bytes | bytearray | memoryview) -> bytes:
    """Reverse bit order in every byte."""
    destination = _buffer(data, writable=True)
    if destination.size:
        status = lib().mti_reverse_bits(destination.ctypes.data, destination.size)
        if status:
            raise RuntimeError(f"bit reversal failed with status {status}")
    return destination.tobytes()
