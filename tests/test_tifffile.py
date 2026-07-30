import io

import numpy as np
import pytest
import tifffile

import mojo_tifffile as mt


@pytest.fixture
def integer_image():
    rng = np.random.default_rng(3)
    return rng.integers(0, 65536, size=(37, 53), dtype=np.uint16)


@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
    ],
)
@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("compression", [None, "deflate", "packbits"])
def test_our_grayscale_files_match_upstream(tmp_path, dtype, byteorder, compression):
    array = np.arange(19 * 23, dtype=dtype).reshape(19, 23)
    predictor = 2 if compression and np.dtype(dtype).kind in "ui" else None
    path = tmp_path / "ours.tif"
    mt.imwrite(
        path,
        array,
        byteorder=byteorder,
        compression=compression,
        predictor=predictor,
        rowsperstrip=5,
    )
    assert np.array_equal(mt.imread(path), array)
    assert np.array_equal(tifffile.imread(path), array)


@pytest.mark.parametrize("compression", [None, "deflate", "packbits", "lzw"])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_reads_upstream_files(tmp_path, integer_image, compression, byteorder):
    path = tmp_path / "upstream.tif"
    tifffile.imwrite(
        path,
        integer_image,
        byteorder=byteorder,
        compression=compression,
        predictor=2 if compression else None,
        rowsperstrip=7,
    )
    assert np.array_equal(mt.imread(path), integer_image)


@pytest.mark.parametrize("bigtiff", [False, True])
def test_multipage_stack_interoperability(tmp_path, bigtiff):
    array = np.arange(5 * 11 * 13, dtype=np.int16).reshape(5, 11, 13)
    path = tmp_path / "stack.tif"
    mt.imwrite(path, array, bigtiff=bigtiff, compression="deflate", predictor=True)
    assert np.array_equal(mt.imread(path), array)
    assert np.array_equal(tifffile.imread(path), array)
    with mt.TiffFile(path) as tif:
        assert len(tif.pages) == 5
        assert tif.series[0].shape == array.shape
        assert np.array_equal(tif.asarray(key=2), array[2])
        assert np.array_equal(tif.asarray(key=slice(1, 3)), array[1:3])


@pytest.mark.parametrize("samples", [3, 4])
def test_contiguous_rgb_interoperability(tmp_path, samples):
    array = np.arange(17 * 19 * samples, dtype=np.uint16).reshape(17, 19, samples)
    path = tmp_path / "rgb.tif"
    mt.imwrite(path, array, compression="packbits", predictor=2, rowsperstrip=4)
    assert np.array_equal(mt.imread(path), array)
    assert np.array_equal(tifffile.imread(path), array)
    with mt.TiffFile(path) as tif:
        assert tif.pages[0].photometric == mt.PHOTOMETRIC.RGB
        assert tif.pages[0].axes == "YXS"


def test_separate_planar_rgb_interoperability(tmp_path):
    array = np.arange(3 * 17 * 19, dtype=np.uint8).reshape(3, 17, 19)
    path = tmp_path / "separate.tif"
    mt.imwrite(
        path,
        array,
        photometric="rgb",
        planarconfig="separate",
        compression="deflate",
        predictor=2,
        rowsperstrip=4,
    )
    assert np.array_equal(mt.imread(path), array)
    assert np.array_equal(tifffile.imread(path), array)


def test_file_objects_and_writer_context():
    stream = io.BytesIO()
    first = np.arange(20, dtype=np.float32).reshape(4, 5)
    second = first + 100
    with mt.TiffWriter(stream) as writer:
        writer.write(first)
        writer.write(second)
    stream.seek(0)
    with mt.TiffFile(stream) as tif:
        assert len(tif.pages) == 2
        assert np.array_equal(tif.asarray(), np.stack([first, second]))
    embedded = io.BytesIO(b"container-prefix" + stream.getvalue())
    with mt.TiffFile(embedded, offset=len(b"container-prefix")) as tif:
        assert np.array_equal(tif.asarray(), np.stack([first, second]))


def test_tags_description_resolution_and_software(tmp_path):
    path = tmp_path / "tags.tif"
    mt.imwrite(
        path,
        np.zeros((7, 9), dtype=np.uint8),
        description="calibration frame",
        resolution=(300, 150),
        resolutionunit="inch",
        software="test suite",
    )
    with mt.TiffFile(path, mode="r") as tif:
        tags = tif.pages[0].tags
        assert tags["ImageDescription"].value == "calibration frame"
        assert tags[305].value == "test suite"
        assert tags["XResolution"].value == (300000000, 1000000)
        assert tags["ResolutionUnit"].value == mt.RESUNIT.INCH
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].description == "calibration frame"


def test_returnoffset_points_to_uncompressed_pixels(tmp_path, integer_image):
    path = tmp_path / "offset.tif"
    offset, size = mt.imwrite(path, integer_image, returnoffset=True)
    assert size == integer_image.nbytes
    with path.open("rb") as handle:
        handle.seek(offset)
        assert handle.read(size) == integer_image.tobytes()


def test_out_array_and_memmap(tmp_path, integer_image):
    path = tmp_path / "image.tif"
    target = tmp_path / "array.bin"
    mt.imwrite(path, integer_image)
    destination = np.empty_like(integer_image)
    assert mt.imread(path, out=destination) is destination
    assert np.array_equal(destination, integer_image)
    mapped = mt.imread(path, out=target)
    assert isinstance(mapped, np.memmap)
    assert np.array_equal(mapped, integer_image)


@pytest.mark.parametrize("side", [127, 2048])
@pytest.mark.parametrize("compression", ["deflate", "packbits"])
def test_strip_parallel_threshold_roundtrip(tmp_path, side, compression):
    row = np.arange(side, dtype=np.uint16)
    array = np.broadcast_to(row, (side, side)).copy()
    path = tmp_path / f"{compression}-{side}.tif"
    mt.imwrite(
        path,
        array,
        compression=compression,
        predictor=2,
        rowsperstrip=128,
    )
    assert np.array_equal(mt.imread(path), array)
    assert np.array_equal(tifffile.imread(path), array)


def test_shape_and_dtype_without_data(tmp_path):
    path = tmp_path / "empty.tif"
    mt.imwrite(path, shape=(5, 7), dtype="uint16")
    assert np.array_equal(mt.imread(path), np.zeros((5, 7), dtype=np.uint16))


def test_explicit_failures_are_not_silent(tmp_path, integer_image):
    with pytest.raises(NotImplementedError, match="tiled"):
        mt.imwrite(tmp_path / "tile.tif", integer_image, tile=(16, 16))
    with pytest.raises(ValueError, match="without compression"):
        mt.imwrite(tmp_path / "predict.tif", integer_image, predictor=True)
    with pytest.raises(NotImplementedError, match="LZW writing"):
        mt.imwrite(tmp_path / "lzw.tif", integer_image, compression="lzw")
    with pytest.raises(NotImplementedError, match="selected"):
        mt.imread(io.BytesIO(b""), selection=(slice(None),))
    with pytest.raises(TypeError, match="unsafe dtype conversion"):
        mt.imwrite(tmp_path / "narrowed.tif", integer_image, dtype="uint8")


def test_truncated_lzw_stream_is_rejected():
    from mojo_tifffile.tifffile import _lzw_decode

    with pytest.raises(mt.TiffFileError, match="truncated"):
        _lzw_decode(b"\x80")
