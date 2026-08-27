# mojo-tifffile

`mojo-tifffile` is a standalone TIFF reader and writer whose PackBits and
horizontal-predictor loops are implemented in compiled
[Mojo](https://www.modular.com/mojo). Its covered API follows the names and
signatures of Python's [`tifffile`](https://github.com/cgohlke/tifffile):
`imread`, `imwrite`, `TiffFile`, and `TiffWriter`.

The runtime depends on NumPy, not on upstream tifffile. Upstream tifffile and
imagecodecs are development dependencies used for bidirectional parity tests
and benchmarks.

## Coverage

The reader supports:

- classic TIFF and BigTIFF, little- and big-endian;
- stripped grayscale, RGB/RGBA contiguous, RGB planar-separate, and compatible
  multi-page stacks;
- unsigned integer, signed integer, and IEEE floating-point samples at 8, 16,
  32, or 64 bits;
- uncompressed, PackBits, Deflate, and LZW compression;
- horizontal prediction, arbitrary rows per strip, file objects,
  page keys, and NumPy or memory-mapped output.

The writer supports those stripped layouts and dtypes, classic TIFF and BigTIFF,
uncompressed/PackBits/Deflate output, horizontal prediction, multi-page
writing, descriptions, resolution, software, and JSON shape metadata.
Files written here are read directly by upstream tifffile, and files written
by upstream are read here in the parity suite.

Not covered are tiled images, packed samples such as 1/10/12-bit data,
floating-point prediction, LZW encoding, JPEG-family and specialist scientific
compressors, OME/ImageJ/Zarr views, SubIFDs and pyramids, append/in-place modes,
or tifffile's full metadata and `extratags` surface. Unsupported encodings raise
an exception instead of returning partial data.

## Install from source and use

There are currently no prebuilt wheels. Clone the repository, then install the
pinned Mojo and Python environment and build the shared library:

```bash
pixi install
pixi run build
```

The following example writes a real Deflate-compressed, predicted TIFF and
reads it back:

```python
import numpy as np
import mojo_tifffile as tifffile

image = np.arange(1024 * 1536, dtype=np.uint16).reshape(1024, 1536)
tifffile.imwrite(
    "image.tif",
    image,
    compression="deflate",
    predictor=True,
    rowsperstrip=128,
)

restored = tifffile.imread("image.tif")
assert np.array_equal(restored, image)

with tifffile.TiffFile("image.tif") as tif:
    print(tif.pages[0].shape, tif.pages[0].dtype)
```

For explicit multi-page control:

```python
with tifffile.TiffWriter("stack.tif", bigtiff=True) as tif:
    tif.write(volume[0], compression="packbits")
    tif.write(volume[1], compression="packbits")
```

Run validation and benchmarks with:

```bash
pixi run test
pixi run bench
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Python 3.13.14. Times are the best of five runs for codec kernels and three
runs for TIFF I/O. The codec reference is imagecodecs; the file reference is
upstream tifffile using imagecodecs.

| case | mojo-tifffile | upstream | upstream / Mojo |
| --- | ---: | ---: | ---: |
| PackBits encode, 16 MiB | 18.64 ms | 21.64 ms | 1.16x (faster) |
| PackBits decode, 16 MiB | 11.44 ms | 28.40 ms | 2.48x (faster) |
| Horizontal predictor, 4096x4096 u16 | 9.00 ms | 16.02 ms | 1.78x (faster) |
| Deflate TIFF read, 4096x4096 u16 | 48.90 ms | 53.15 ms | 1.09x (faster) |
| PackBits TIFF write, 4096x4096 u16 | 43.84 ms | 37.00 ms | 0.84x (slower) |

The horizontal predictor fuses the output copy with SIMD differencing and sends
zero-copy row ranges to up to eight CPU workers above 4 MiB. Large compressed
TIFFs process independent strips with up to eight CPU workers above 8 MiB, while
small inputs remain serial. PackBits literal packets use unaligned SIMD copies
with scalar tails, and the writer passes their NumPy buffers directly to file
I/O instead of copying each strip to `bytes`. Deflate decompression starts with
the exact decoded strip size to avoid repeated output-buffer growth. The
PackBits TIFF writer remained slower than upstream in this run; the predictor
and Deflate reader moved ahead.

There is no GPU path. Horizontal prediction performs one integer operation for
roughly three element transfers, far below two operations per byte, while
PackBits is branch- and bandwidth-bound and Deflate is delegated to zlib.
Host/device transfer and launch overhead therefore cannot be justified for
these kernels.

## How it works

`src/tifffile.mojo` is one compilation unit built as
`dist/libmojo-tifffile.so`. Python loads five C-ABI functions with `ctypes`.
Buffers cross that boundary as integer addresses and Mojo reconstructs mutable
`UnsafePointer` values with `AnyOrigin[mut=True]`; no Python objects enter Mojo.

Image samples remain in C-contiguous, interleaved row-major order. Horizontal
prediction subtracts the previous pixel's corresponding sample modulo the
sample width and reverses that operation during reading. Native little-endian
8/16/32/64-bit encode paths use unaligned-safe SIMD loads and stores with
scalar tails; the general path handles big-endian words byte by byte. PackBits
operates on byte streams and bounds-checks both input and destination buffers.

The Python layer handles TIFF headers, IFDs, tag values, strip placement, byte
order, Deflate through the standard library, and a TIFF-compatible LZW decoder.
Mojo performs the compute-bound transforms; Python retains the irregular I/O
and metadata work.

MIT.
