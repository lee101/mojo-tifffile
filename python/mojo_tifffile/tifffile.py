"""A compact TIFF reader and writer with Mojo-accelerated codecs."""

from __future__ import annotations

import io
import json
import os
import struct
import zlib
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum
from math import prod
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from ._lib import lib as _codec_lib
from .codecs import (
    _packbits_decode_into,
    _predictor_copy_into,
    _predictor_inplace,
    bitorder_decode,
    delta_encode,
    packbits_encode,
)

__version__ = "0.1.0"
_PARALLEL_STRIP_BYTES = 8 * 1024 * 1024
_MAX_STRIP_WORKERS = 8


class COMPRESSION(IntEnum):
    NONE = 1
    LZW = 5
    ADOBE_DEFLATE = 8
    DEFLATE = 32946
    PACKBITS = 32773


class PHOTOMETRIC(IntEnum):
    MINISWHITE = 0
    MINISBLACK = 1
    RGB = 2
    PALETTE = 3


class PLANARCONFIG(IntEnum):
    CONTIG = 1
    SEPARATE = 2


class PREDICTOR(IntEnum):
    NONE = 1
    HORIZONTAL = 2


class SAMPLEFORMAT(IntEnum):
    UINT = 1
    INT = 2
    IEEEFP = 3


class RESUNIT(IntEnum):
    NONE = 1
    INCH = 2
    CENTIMETER = 3


TIFF = type(
    "TIFF",
    (),
    {
        "COMPRESSION": COMPRESSION,
        "PHOTOMETRIC": PHOTOMETRIC,
        "PLANARCONFIG": PLANARCONFIG,
        "PREDICTOR": PREDICTOR,
        "SAMPLEFORMAT": SAMPLEFORMAT,
        "RESUNIT": RESUNIT,
    },
)

_TAG_NAMES = {
    254: "NewSubfileType",
    256: "ImageWidth",
    257: "ImageLength",
    258: "BitsPerSample",
    259: "Compression",
    262: "PhotometricInterpretation",
    266: "FillOrder",
    270: "ImageDescription",
    273: "StripOffsets",
    274: "Orientation",
    277: "SamplesPerPixel",
    278: "RowsPerStrip",
    279: "StripByteCounts",
    282: "XResolution",
    283: "YResolution",
    284: "PlanarConfiguration",
    296: "ResolutionUnit",
    305: "Software",
    306: "DateTime",
    317: "Predictor",
    320: "ColorMap",
    322: "TileWidth",
    323: "TileLength",
    324: "TileOffsets",
    325: "TileByteCounts",
    338: "ExtraSamples",
    339: "SampleFormat",
}
_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}


class TiffFileError(ValueError):
    pass


class TiffTag:
    def __init__(self, code: int, dtype: int, count: int, value: Any, offset: int):
        self.code = code
        self.name = _TAG_NAMES.get(code, str(code))
        self.dtype = dtype
        self.count = count
        self.value = value
        self.offset = offset

    def __repr__(self) -> str:
        return f"<TiffTag {self.code} {self.name!r} @{self.offset} {self.value!r}>"


class TiffTags(Mapping):
    def __init__(self, tags: dict[int, TiffTag]):
        self._tags = tags
        self._names = {tag.name: tag for tag in tags.values()}

    def __getitem__(self, key):
        return self._names[key] if isinstance(key, str) else self._tags[key]

    def __iter__(self):
        return iter(self._tags)

    def __len__(self):
        return len(self._tags)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _scalar(value, default):
    if value is None:
        return default
    value = value.value if isinstance(value, TiffTag) else value
    return value[0] if isinstance(value, tuple) else value


def _decode_values(raw: bytes, dtype: int, count: int, endian: str):
    if dtype == 2:
        return raw[:count].split(b"\0", 1)[0].decode("utf-8", "replace")
    formats = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f", 12: "d", 16: "Q", 17: "q", 18: "Q"}
    if dtype in (5, 10):
        signed = dtype == 10
        fmt = "ii" if signed else "II"
        values = struct.unpack(endian + fmt * count, raw[: count * 8])
        result = tuple((values[i], values[i + 1]) for i in range(0, len(values), 2))
    elif dtype == 7:
        result = (raw[:count],)
    else:
        try:
            result = struct.unpack(endian + formats[dtype] * count, raw[: count * _TYPE_SIZES[dtype]])
        except KeyError as exc:
            raise TiffFileError(f"unsupported TIFF field type {dtype}") from exc
    return result[0] if count == 1 else result


def _open_file(file, mode: str) -> tuple[BinaryIO, bool]:
    if hasattr(file, "read") or hasattr(file, "write"):
        return file, False
    return open(os.fspath(file), mode), True


class TiffPage:
    def __init__(self, parent: "TiffFile", offset: int, index: int):
        self.parent = parent
        self.offset = offset
        self.index = index
        self.tags, self._next_offset = parent._read_ifd(offset)
        if "TileOffsets" in self.tags:
            raise NotImplementedError("tiled TIFF images are not supported")
        self.imagewidth = int(_scalar(self.tags.get("ImageWidth"), 0))
        self.imagelength = int(_scalar(self.tags.get("ImageLength"), 0))
        self.samplesperpixel = int(_scalar(self.tags.get("SamplesPerPixel"), 1))
        self.bitspersample = _scalar(self.tags.get("BitsPerSample"), 1)
        self.sampleformat = int(_scalar(self.tags.get("SampleFormat"), 1))
        self.compression = COMPRESSION(int(_scalar(self.tags.get("Compression"), 1)))
        self.photometric = PHOTOMETRIC(int(_scalar(self.tags.get("PhotometricInterpretation"), 1)))
        self.planarconfig = PLANARCONFIG(int(_scalar(self.tags.get("PlanarConfiguration"), 1)))
        self.predictor = PREDICTOR(int(_scalar(self.tags.get("Predictor"), 1)))
        self.rowsperstrip = int(_scalar(self.tags.get("RowsPerStrip"), self.imagelength))
        if self.imagewidth <= 0 or self.imagelength <= 0:
            raise TiffFileError("TIFF image dimensions must be positive")
        if self.samplesperpixel <= 0 or self.rowsperstrip <= 0:
            raise TiffFileError("SamplesPerPixel and RowsPerStrip must be positive")
        self.dtype = self._dtype()
        if self.samplesperpixel == 1:
            self.shape = (self.imagelength, self.imagewidth)
            self.axes = "YX"
        elif self.planarconfig == PLANARCONFIG.SEPARATE:
            self.shape = (self.samplesperpixel, self.imagelength, self.imagewidth)
            self.axes = "SYX"
        else:
            self.shape = (self.imagelength, self.imagewidth, self.samplesperpixel)
            self.axes = "YXS"

    def _dtype(self) -> np.dtype:
        bits = self.bitspersample
        if isinstance(bits, tuple):
            if len(set(bits)) != 1:
                raise NotImplementedError("mixed BitsPerSample values are not supported")
            bits = bits[0]
        if bits not in (8, 16, 32, 64):
            raise NotImplementedError(f"{bits}-bit samples are not supported")
        kind = {1: "u", 2: "i", 3: "f"}.get(self.sampleformat)
        if kind is None or (kind == "f" and bits not in (32, 64)):
            raise NotImplementedError("unsupported TIFF SampleFormat")
        return np.dtype(self.parent.byteorder + kind + str(bits // 8))

    @property
    def description(self):
        tag = self.tags.get("ImageDescription")
        return tag.value if tag else None

    def asarray(self, *, out=None, squeeze=True, **kwargs):
        offsets = self.tags["StripOffsets"].value
        counts = self.tags["StripByteCounts"].value
        offsets = (offsets,) if isinstance(offsets, int) else offsets
        counts = (counts,) if isinstance(counts, int) else counts
        if len(offsets) != len(counts):
            raise TiffFileError("StripOffsets and StripByteCounts lengths differ")
        strips_per_plane = (self.imagelength + self.rowsperstrip - 1) // self.rowsperstrip
        expected_strips = strips_per_plane * (
            self.samplesperpixel if self.planarconfig == PLANARCONFIG.SEPARATE else 1
        )
        if len(offsets) != expected_strips:
            raise TiffFileError(f"expected {expected_strips} strips, found {len(offsets)}")
        row_samples = self.imagewidth * (
            1 if self.planarconfig == PLANARCONFIG.SEPARATE else self.samplesperpixel
        )
        rowbytes = row_samples * self.dtype.itemsize
        decoded = bytearray(prod(self.shape) * self.dtype.itemsize)
        decoded_view = memoryview(decoded)
        fill_order = int(_scalar(self.tags.get("FillOrder"), 1))
        if fill_order not in (1, 2):
            raise NotImplementedError(f"TIFF FillOrder {fill_order} is not supported")
        strip_specs = []
        for strip_index, (offset, count) in enumerate(zip(offsets, counts)):
            plane_strip = strip_index % strips_per_plane
            row = plane_strip * self.rowsperstrip
            rows = min(self.rowsperstrip, self.imagelength - row)
            self.parent._fh.seek(self.parent._base + offset)
            encoded = self.parent._fh.read(count)
            if len(encoded) != count:
                raise TiffFileError("truncated TIFF strip")
            size = rows * rowbytes
            if self.planarconfig == PLANARCONFIG.SEPARATE:
                plane = strip_index // strips_per_plane
                destination = (plane * self.imagelength + row) * rowbytes
            else:
                destination = row * rowbytes
            strip_specs.append((encoded, destination, size, rows))

        def decode_strip(spec):
            encoded, destination, size, rows = spec
            if fill_order == 2:
                encoded = bitorder_decode(encoded)
            target = decoded_view[destination : destination + size]
            predicted = False
            if self.compression == COMPRESSION.NONE:
                if len(encoded) < size:
                    raise TiffFileError(
                        f"decoded strip has {len(encoded)} bytes, expected {size}"
                    )
                chunk = encoded[:size]
                if self.predictor == PREDICTOR.HORIZONTAL:
                    _predictor_copy_into(
                        chunk,
                        target,
                        (rows, self.imagewidth, row_samples // self.imagewidth),
                        self.dtype.itemsize,
                        self.parent.byteorder,
                        True,
                    )
                    predicted = True
                else:
                    target[:] = chunk
            elif self.compression == COMPRESSION.PACKBITS:
                actual = _packbits_decode_into(encoded, target)
                if actual < size:
                    raise TiffFileError(
                        f"decoded strip has {actual} bytes, expected {size}"
                    )
            elif self.compression in (COMPRESSION.ADOBE_DEFLATE, COMPRESSION.DEFLATE):
                chunk = zlib.decompress(encoded)
                if len(chunk) < size:
                    raise TiffFileError(
                        f"decoded strip has {len(chunk)} bytes, expected {size}"
                    )
                chunk = chunk[:size]
                if self.predictor == PREDICTOR.HORIZONTAL:
                    _predictor_copy_into(
                        chunk,
                        target,
                        (rows, self.imagewidth, row_samples // self.imagewidth),
                        self.dtype.itemsize,
                        self.parent.byteorder,
                        True,
                    )
                    predicted = True
                else:
                    target[:] = chunk
            elif self.compression == COMPRESSION.LZW:
                chunk = _lzw_decode(encoded)
                if len(chunk) < size:
                    raise TiffFileError(
                        f"decoded strip has {len(chunk)} bytes, expected {size}"
                    )
                target[:] = chunk[:size]
            else:
                raise NotImplementedError(f"compression {self.compression.name} is not supported")
            if self.predictor == PREDICTOR.HORIZONTAL and not predicted:
                _predictor_inplace(
                    target,
                    (rows, self.imagewidth, row_samples // self.imagewidth),
                    self.dtype.itemsize,
                    self.parent.byteorder,
                    True,
                )

        parallel = (
            len(decoded) >= _PARALLEL_STRIP_BYTES
            and len(strip_specs) > 1
            and (
                self.compression not in (COMPRESSION.NONE, COMPRESSION.LZW)
                or self.predictor == PREDICTOR.HORIZONTAL
            )
        )
        if parallel:
            _codec_lib()
            workers = min(_MAX_STRIP_WORKERS, len(strip_specs))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(decode_strip, strip_specs))
        else:
            for spec in strip_specs:
                decode_strip(spec)
        array = np.frombuffer(decoded, dtype=self.dtype).reshape(self.shape)
        array = array.astype(self.dtype.newbyteorder("="), copy=False)
        if squeeze:
            array = np.squeeze(array)
        return _copy_to_out(array, out)

    def __repr__(self):
        return f"<TiffPage {self.index} @{self.offset} {self.shape} {self.dtype}>"


class TiffPageSeries:
    def __init__(self, pages: Sequence[TiffPage]):
        self.pages = list(pages)
        first = self.pages[0]
        self.dtype = first.dtype.newbyteorder("=")
        self.shape = first.shape if len(self.pages) == 1 else (len(self.pages),) + first.shape
        self.axes = first.axes if len(self.pages) == 1 else "Q" + first.axes

    def asarray(self, **kwargs):
        arrays = [page.asarray(**kwargs) for page in self.pages]
        return arrays[0] if len(arrays) == 1 else np.stack(arrays)


class TiffFile:
    def __init__(
        self,
        file,
        /,
        *,
        mode=None,
        name=None,
        offset=None,
        size=None,
        omexml=None,
        superres=None,
        _multifile=None,
        _useframes=None,
        _root=None,
        **is_flags,
    ):
        del name, size, omexml, superres, _multifile, _useframes, _root, is_flags
        open_mode = "rb" if mode in (None, "r") else "r+b"
        self._fh, self._close = _open_file(file, open_mode)
        self.filehandle = self._fh
        self.filename = getattr(self._fh, "name", None)
        self._base = int(offset or 0)
        self._fh.seek(self._base)
        marker = self._fh.read(4)
        if marker[:2] == b"II":
            self.byteorder = "<"
        elif marker[:2] == b"MM":
            self.byteorder = ">"
        else:
            raise TiffFileError("not a TIFF file")
        version = struct.unpack(self.byteorder + "H", marker[2:])[0]
        if version == 42:
            self.is_bigtiff = False
            first = struct.unpack(self.byteorder + "I", self._fh.read(4))[0]
        elif version == 43:
            self.is_bigtiff = True
            offsetsize, zero = struct.unpack(self.byteorder + "HH", self._fh.read(4))
            if offsetsize != 8 or zero:
                raise TiffFileError("invalid BigTIFF header")
            first = struct.unpack(self.byteorder + "Q", self._fh.read(8))[0]
        else:
            raise TiffFileError(f"unsupported TIFF version {version}")
        self.pages: list[TiffPage] = []
        seen = set()
        while first:
            if first in seen:
                raise TiffFileError("circular TIFF IFD chain")
            seen.add(first)
            page = TiffPage(self, self._base + first, len(self.pages))
            self.pages.append(page)
            first = page._next_offset
        if not self.pages:
            raise TiffFileError("TIFF contains no pages")
        compatible = all(
            page.shape == self.pages[0].shape and page.dtype == self.pages[0].dtype
            for page in self.pages
        )
        self.series = [TiffPageSeries(self.pages)] if compatible else [
            TiffPageSeries([page]) for page in self.pages
        ]

    def _read_ifd(self, offset: int):
        self._fh.seek(offset)
        if self.is_bigtiff:
            count = struct.unpack(self.byteorder + "Q", self._fh.read(8))[0]
            entry_size, inline, entry_fmt, next_fmt = 20, 8, "HHQQ", "Q"
        else:
            count = struct.unpack(self.byteorder + "H", self._fh.read(2))[0]
            entry_size, inline, entry_fmt, next_fmt = 12, 4, "HHII", "I"
        if count > 4096:
            raise TiffFileError(f"implausible IFD entry count {count}")
        tags = {}
        for _ in range(count):
            entry_offset = self._fh.tell()
            entry = self._fh.read(entry_size)
            if len(entry) != entry_size:
                raise TiffFileError("truncated TIFF IFD")
            code, dtype, value_count, value_offset = struct.unpack(self.byteorder + entry_fmt, entry)
            try:
                nbytes = _TYPE_SIZES[dtype] * value_count
            except KeyError as exc:
                raise TiffFileError(f"unsupported TIFF field type {dtype}") from exc
            if nbytes <= inline:
                raw = entry[-inline:][:nbytes]
            else:
                position = self._fh.tell()
                self._fh.seek(self._base + value_offset)
                raw = self._fh.read(nbytes)
                self._fh.seek(position)
                if len(raw) != nbytes:
                    raise TiffFileError("truncated TIFF tag value")
            tags[code] = TiffTag(
                code, dtype, value_count, _decode_values(raw, dtype, value_count, self.byteorder), entry_offset
            )
        next_offset = struct.unpack(self.byteorder + next_fmt, self._fh.read(struct.calcsize(next_fmt)))[0]
        return TiffTags(tags), next_offset

    def asarray(self, key=None, series=None, level=None, squeeze=None, out=None, **kwargs):
        del level
        if series is not None:
            array = self.series[int(series)].asarray(squeeze=squeeze is not False, **kwargs)
        elif key is None:
            array = self.series[0].asarray(squeeze=squeeze is not False, **kwargs)
        elif isinstance(key, slice):
            array = np.stack([p.asarray(squeeze=squeeze is not False, **kwargs) for p in self.pages[key]])
        elif isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
            array = np.stack([self.pages[int(i)].asarray(squeeze=squeeze is not False, **kwargs) for i in key])
        else:
            array = self.pages[int(key)].asarray(squeeze=squeeze is not False, **kwargs)
        return _copy_to_out(array, out)

    def close(self):
        if self._close:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __len__(self):
        return len(self.pages)


def _lzw_decode(data: bytes) -> bytes:
    bitpos = 0
    width = 9
    table = [bytes((i,)) for i in range(256)] + [b"", b""]
    next_code = 258
    previous = None
    result = bytearray()

    def read_code(bits):
        nonlocal bitpos
        if bitpos + bits > len(data) * 8:
            return None
        value = 0
        for _ in range(bits):
            byte, shift = divmod(bitpos, 8)
            value = (value << 1) | ((data[byte] >> (7 - shift)) & 1)
            bitpos += 1
        return value

    saw_eoi = False
    while True:
        code = read_code(width)
        if code is None:
            break
        if code == 257:
            saw_eoi = True
            break
        if code == 256:
            table = [bytes((i,)) for i in range(256)] + [b"", b""]
            next_code, width, previous = 258, 9, None
            continue
        if code < len(table) and table[code]:
            entry = table[code]
        elif code == next_code and previous is not None:
            entry = previous + previous[:1]
        else:
            raise TiffFileError("invalid TIFF LZW stream")
        result.extend(entry)
        if previous is not None and next_code < 4096:
            table.append(previous + entry[:1])
            next_code += 1
            if next_code == (1 << width) - 1 and width < 12:
                width += 1
        previous = entry
    if not saw_eoi:
        raise TiffFileError("truncated TIFF LZW stream")
    return bytes(result)


def _enum(value, enum, aliases, default):
    if value is None:
        return enum(default)
    if isinstance(value, bool):
        return enum(2 if value else default)
    if isinstance(value, str):
        key = value.lower().replace("-", "_").replace(" ", "_")
        if key in aliases:
            return enum(aliases[key])
        try:
            return enum[key.upper()]
        except KeyError:
            pass
    try:
        return enum(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unknown {enum.__name__.lower()} {value!r}") from exc


def _compression(value):
    return _enum(
        value,
        COMPRESSION,
        {"none": 1, "raw": 1, "deflate": 32946, "zlib": 32946, "adobe_deflate": 8, "packbits": 32773},
        1,
    )


def _rational(value):
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    denominator = 1_000_000
    return int(round(float(value) * denominator)), denominator


def _pack_values(endian, dtype, value):
    if dtype == 2:
        raw = value.encode() if isinstance(value, str) else bytes(value)
        return raw if raw.endswith(b"\0") else raw + b"\0", len(raw) + (not raw.endswith(b"\0"))
    values = value if isinstance(value, (tuple, list)) else (value,)
    if dtype == 5:
        flat = [part for pair in values for part in pair] if values and isinstance(values[0], tuple) else list(values)
        return struct.pack(endian + "I" * len(flat), *flat), len(flat) // 2
    fmt = {1: "B", 3: "H", 4: "I", 7: "B", 9: "i", 11: "f", 12: "d", 16: "Q", 17: "q"}[dtype]
    return struct.pack(endian + fmt * len(values), *values), len(values)


class TiffWriter:
    def __init__(
        self,
        file,
        /,
        *,
        mode=None,
        bigtiff=False,
        byteorder=None,
        append=False,
        kind=None,
        imagej=False,
        ome=None,
        shaped=None,
    ):
        del kind, imagej, ome, shaped
        if append:
            raise NotImplementedError("append mode is not supported")
        mode = mode or "w"
        if mode not in ("w", "x"):
            raise NotImplementedError(f"mode {mode!r} is not supported")
        self._fh, self._close = _open_file(file, mode + "b")
        self.byteorder = ">" if byteorder in (">", "big", "be") else "<"
        self.bigtiff = bool(bigtiff)
        self._previous_next = None
        self._first_page = True
        if self.bigtiff:
            self._fh.write((b"MM" if self.byteorder == ">" else b"II") + struct.pack(self.byteorder + "HHHQ", 43, 8, 0, 0))
            self._header_pointer = 8
        else:
            self._fh.write((b"MM" if self.byteorder == ">" else b"II") + struct.pack(self.byteorder + "HI", 42, 0))
            self._header_pointer = 4

    def write(
        self,
        data=None,
        *,
        shape=None,
        dtype=None,
        photometric=None,
        planarconfig=None,
        extrasamples=None,
        volumetric=False,
        tile=None,
        rowsperstrip=None,
        bitspersample=None,
        compression=None,
        compressionargs=None,
        predictor=None,
        subsampling=None,
        jpegtables=None,
        iccprofile=None,
        colormap=None,
        description=None,
        datetime=None,
        resolution=None,
        resolutionunit=None,
        subfiletype=None,
        software=None,
        metadata={},
        contiguous=False,
        truncate=False,
        align=None,
        maxworkers=None,
        buffersize=None,
        returnoffset=False,
    ):
        del volumetric, compressionargs, subsampling, jpegtables, iccprofile, colormap
        del contiguous, truncate, align, maxworkers, buffersize
        if tile is not None:
            raise NotImplementedError("tiled TIFF writing is not supported")
        if data is None:
            if shape is None or dtype is None:
                raise ValueError("data or both shape and dtype are required")
            array = np.zeros(shape, dtype=dtype)
        else:
            array = np.asarray(data)
            if dtype is not None:
                requested = np.dtype(dtype)
                if not np.can_cast(array.dtype, requested, casting="safe"):
                    raise TypeError(
                        f"refusing unsafe dtype conversion from {array.dtype} to {requested}"
                    )
                array = array.astype(requested, copy=False)
            if shape is not None:
                array = array.reshape(shape)
        if array.dtype.kind not in "uif" or array.dtype.itemsize not in (1, 2, 4, 8):
            raise TypeError(f"unsupported TIFF dtype {array.dtype}")
        if bitspersample not in (None, array.dtype.itemsize * 8):
            raise NotImplementedError("packed or truncated integer samples are not supported")
        photo = _enum(
            photometric,
            PHOTOMETRIC,
            {"minisblack": 1, "miniswhite": 0, "rgb": 2},
            2 if array.ndim >= 3 and array.shape[-1] in (3, 4) else 1,
        )
        planar = _enum(planarconfig, PLANARCONFIG, {"contig": 1, "separate": 2}, 1)
        if photo == PHOTOMETRIC.RGB:
            if planar == PLANARCONFIG.CONTIG:
                if array.ndim < 3 or array.shape[-1] not in (3, 4):
                    raise ValueError("contiguous RGB data must end in 3 or 4 samples")
                page_ndim = 3
            else:
                if array.ndim < 3 or array.shape[-3] not in (3, 4):
                    raise ValueError("separate RGB data must have 3 or 4 samples on axis -3")
                page_ndim = 3
        else:
            if array.ndim < 2:
                raise ValueError("TIFF images must be at least two-dimensional")
            page_ndim = 2
        page_shape = array.shape[-page_ndim:]
        pages = array.reshape((-1,) + page_shape)
        first_result = None
        for index, page in enumerate(pages):
            page_description = description if index == 0 else None
            if index == 0 and page_description is None and metadata:
                page_description = json.dumps({"shape": list(array.shape), **dict(metadata)}, separators=(",", ":"))
            result = self._write_page(
                page,
                photo,
                planar,
                extrasamples,
                rowsperstrip,
                _compression(compression),
                _enum(predictor, PREDICTOR, {"none": 1, "horizontal": 2}, 1),
                page_description,
                datetime,
                resolution,
                resolutionunit,
                subfiletype,
                software,
            )
            first_result = result if first_result is None else first_result
        return first_result if returnoffset else None

    def _write_page(
        self,
        page,
        photo,
        planar,
        extrasamples,
        rowsperstrip,
        compression,
        predictor,
        description,
        datetime,
        resolution,
        resolutionunit,
        subfiletype,
        software,
    ):
        if predictor == PREDICTOR.HORIZONTAL and page.dtype.kind not in "ui":
            raise ValueError("horizontal predictor requires integer samples")
        if predictor == PREDICTOR.HORIZONTAL and compression == COMPRESSION.NONE:
            raise ValueError("cannot use predictor without compression")
        if planar == PLANARCONFIG.CONTIG:
            height, width = page.shape[:2]
            samples = page.shape[2] if page.ndim == 3 else 1
            planes = [page]
        else:
            samples, height, width = page.shape
            planes = list(page)
        file_dtype = page.dtype.newbyteorder(self.byteorder)
        rowbytes = width * (1 if planar == PLANARCONFIG.SEPARATE else samples) * file_dtype.itemsize
        if rowsperstrip is None:
            rowsperstrip = max(1, min(height, 256 * 1024 // max(1, rowbytes)))
        rowsperstrip = max(1, min(height, int(rowsperstrip)))
        offsets, bytecounts = [], []
        if predictor == PREDICTOR.HORIZONTAL:
            predictor_samples = (
                1 if planar == PLANARCONFIG.SEPARATE else samples
            )
            planes = [
                delta_encode(
                    np.ascontiguousarray(plane, dtype=file_dtype),
                    (height, width, predictor_samples),
                    file_dtype.itemsize,
                    self.byteorder,
                )
                for plane in planes
            ]
        strip_specs = [
            (plane, row)
            for plane in planes
            for row in range(0, height, rowsperstrip)
        ]

        def encode_strip(spec):
            plane, row = spec
            if predictor == PREDICTOR.HORIZONTAL:
                rows = min(rowsperstrip, height - row)
                start = row * rowbytes
                raw = memoryview(plane)[
                    start : start + rows * rowbytes
                ]
            else:
                strip = np.ascontiguousarray(
                    plane[row : row + rowsperstrip], dtype=file_dtype
                )
                raw = memoryview(strip).cast("B")
            if compression == COMPRESSION.NONE:
                encoded = raw
            elif compression == COMPRESSION.PACKBITS:
                encoded = packbits_encode(raw)
            elif compression in (COMPRESSION.ADOBE_DEFLATE, COMPRESSION.DEFLATE):
                encoded = zlib.compress(raw)
            else:
                raise NotImplementedError(f"{compression.name} writing is not supported")
            return encoded, len(raw)

        parallel = (
            page.nbytes >= _PARALLEL_STRIP_BYTES
            and len(strip_specs) > 1
            and compression != COMPRESSION.NONE
        )
        if parallel:
            _codec_lib()
            workers = min(_MAX_STRIP_WORKERS, len(strip_specs))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                encoded_strips = list(executor.map(encode_strip, strip_specs))
        else:
            encoded_strips = [encode_strip(spec) for spec in strip_specs]
        total_uncompressed = 0
        for encoded, raw_size in encoded_strips:
            offsets.append(self._fh.tell())
            bytecounts.append(len(encoded))
            total_uncompressed += raw_size
            self._fh.write(encoded)
        tags = [
            (256, 4, width),
            (257, 4, height),
            (258, 3, (page.dtype.itemsize * 8,) * samples),
            (259, 3, int(compression)),
            (262, 3, int(photo)),
            (273, 16 if self.bigtiff else 4, tuple(offsets)),
            (274, 3, 1),
            (277, 3, samples),
            (278, 4, rowsperstrip),
            (279, 16 if self.bigtiff else 4, tuple(bytecounts)),
            (284, 3, int(planar)),
            (339, 3, ({"u": 1, "i": 2, "f": 3}[page.dtype.kind],) * samples),
        ]
        if predictor != PREDICTOR.NONE:
            tags.append((317, 3, int(predictor)))
        if samples == 4 and photo == PHOTOMETRIC.RGB:
            extra = tuple(extrasamples) if extrasamples not in (None, False) else (2,)
            tags.append((338, 3, extra))
        if description is not None:
            tags.append((270, 2, description))
        if software is not False:
            tags.append((305, 2, software or "mojo-tifffile"))
        if datetime not in (None, False):
            tags.append((306, 2, str(datetime)))
        if resolution is not None:
            tags.extend([(282, 5, (_rational(resolution[0]),)), (283, 5, (_rational(resolution[1]),))])
            tags.append((296, 3, int(_enum(resolutionunit, RESUNIT, {"inch": 2, "centimeter": 3, "cm": 3}, 2))))
        if subfiletype is not None:
            tags.append((254, 4, int(subfiletype)))
        tags.sort()
        ifd_offset = self._fh.tell()
        self._link_ifd(ifd_offset)
        count_size, entry_size, inline, next_size = ((8, 20, 8, 8) if self.bigtiff else (2, 12, 4, 4))
        external_offset = ifd_offset + count_size + entry_size * len(tags) + next_size
        entries, external = [], bytearray()
        for code, field_type, value in tags:
            raw, count = _pack_values(self.byteorder, field_type, value)
            if len(raw) <= inline:
                value_field = raw.ljust(inline, b"\0")
            else:
                if external_offset % 2:
                    external.append(0)
                    external_offset += 1
                pointer = external_offset
                external.extend(raw)
                external_offset += len(raw)
                value_field = struct.pack(self.byteorder + ("Q" if self.bigtiff else "I"), pointer)
            if self.bigtiff:
                entries.append(struct.pack(self.byteorder + "HHQ", code, field_type, count) + value_field)
            else:
                entries.append(struct.pack(self.byteorder + "HHI", code, field_type, count) + value_field)
        self._fh.write(struct.pack(self.byteorder + ("Q" if self.bigtiff else "H"), len(tags)))
        for entry in entries:
            self._fh.write(entry)
        self._previous_next = self._fh.tell()
        self._fh.write(b"\0" * next_size)
        self._fh.write(external)
        self._fh.flush()
        return offsets[0], total_uncompressed

    def _link_ifd(self, offset):
        current = self._fh.tell()
        pointer = self._header_pointer if self._first_page else self._previous_next
        self._fh.seek(pointer)
        self._fh.write(struct.pack(self.byteorder + ("Q" if self.bigtiff else "I"), offset))
        self._fh.seek(current)
        self._first_page = False

    def close(self):
        if self._close:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _copy_to_out(array, out):
    if out is None:
        return array
    if isinstance(out, np.ndarray):
        out[...] = array
        return out
    destination = np.memmap(out, dtype=array.dtype, mode="w+", shape=array.shape)
    destination[...] = array
    return destination


def imread(
    files=None,
    *,
    selection=None,
    return_as=None,
    aszarr=False,
    key=None,
    series=None,
    kind=None,
    level=None,
    squeeze=None,
    maxworkers=None,
    buffersize=None,
    mode=None,
    name=None,
    offset=None,
    size=None,
    pattern=None,
    axesorder=None,
    categories=None,
    imread=None,
    imreadargs=None,
    sort=None,
    container=None,
    chunkshape=None,
    chunkdtype=None,
    axestiled=None,
    ioworkers=1,
    chunkmode=None,
    fillvalue=None,
    zattrs=None,
    multiscales=None,
    omexml=None,
    superres=None,
    out=None,
    out_inplace=None,
    _multifile=None,
    _useframes=None,
    **kwargs,
):
    del kind, maxworkers, buffersize, pattern, axesorder, categories, imread, imreadargs
    del sort, container, chunkshape, chunkdtype, axestiled, ioworkers, chunkmode, fillvalue
    del zattrs, multiscales, out_inplace
    if files is None:
        raise ValueError("a TIFF file is required")
    if selection is not None or return_as is not None or aszarr:
        raise NotImplementedError("lazy, selected, and non-NumPy reads are not supported")
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes, os.PathLike)):
        return np.stack([imread(file, key=key, squeeze=squeeze, **kwargs) for file in files])
    with TiffFile(
        files,
        mode=mode,
        name=name,
        offset=offset,
        size=size,
        omexml=omexml,
        superres=superres,
        _multifile=_multifile,
        _useframes=_useframes,
    ) as tif:
        return tif.asarray(key=key, series=series, level=level, squeeze=squeeze, out=out, **kwargs)


def imwrite(
    file,
    /,
    data=None,
    *,
    mode=None,
    bigtiff=None,
    byteorder=None,
    kind=None,
    imagej=False,
    ome=None,
    shaped=None,
    append=False,
    shape=None,
    dtype=None,
    photometric=None,
    planarconfig=None,
    extrasamples=None,
    volumetric=False,
    tile=None,
    rowsperstrip=None,
    bitspersample=None,
    compression=None,
    compressionargs=None,
    predictor=None,
    subsampling=None,
    jpegtables=None,
    iccprofile=None,
    colormap=None,
    description=None,
    datetime=None,
    resolution=None,
    resolutionunit=None,
    subfiletype=None,
    software=None,
    metadata={},
    contiguous=False,
    truncate=False,
    align=None,
    maxworkers=None,
    buffersize=None,
    returnoffset=False,
):
    with TiffWriter(
        file,
        mode=mode,
        bigtiff=bool(bigtiff),
        byteorder=byteorder,
        append=append,
        kind=kind,
        imagej=imagej,
        ome=ome,
        shaped=shaped,
    ) as tif:
        return tif.write(
            data,
            shape=shape,
            dtype=dtype,
            photometric=photometric,
            planarconfig=planarconfig,
            extrasamples=extrasamples,
            volumetric=volumetric,
            tile=tile,
            rowsperstrip=rowsperstrip,
            bitspersample=bitspersample,
            compression=compression,
            compressionargs=compressionargs,
            predictor=predictor,
            subsampling=subsampling,
            jpegtables=jpegtables,
            iccprofile=iccprofile,
            colormap=colormap,
            description=description,
            datetime=datetime,
            resolution=resolution,
            resolutionunit=resolutionunit,
            subfiletype=subfiletype,
            software=software,
            metadata=metadata,
            contiguous=contiguous,
            truncate=truncate,
            align=align,
            maxworkers=maxworkers,
            buffersize=buffersize,
            returnoffset=returnoffset,
        )
