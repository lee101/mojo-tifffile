"""Benchmarks against tifffile and imagecodecs on identical inputs."""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import imagecodecs  # noqa: E402
import mojo_tifffile as mt  # noqa: E402
import tifffile  # noqa: E402


def timeit(function, repeat=5):
    best = float("inf")
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def machine():
    model = " ".join(platform.processor().split())
    if model.lower() in ("", "x86_64", "amd64") and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    return f"{model or platform.machine()}, Python {platform.python_version()}"


def main():
    rng = np.random.default_rng(7)
    runs = rng.integers(0, 16, size=16 * 1024 * 1024, dtype=np.uint8)
    runs = np.repeat(runs[::8], 8).tobytes()
    encoded = imagecodecs.packbits_encode(runs)
    image = rng.integers(0, 65536, size=(4096, 4096), dtype=np.uint16)
    raw = image.tobytes()
    mt.packbits_encode(b"warmup")

    rows = []
    ours, ours_encoded = timeit(lambda: mt.packbits_encode(runs))
    reference, _ = timeit(lambda: imagecodecs.packbits_encode(runs))
    rows.append(("PackBits encode, 16 MiB", ours, reference))

    ours, _ = timeit(lambda: mt.packbits_decode(encoded, len(runs)))
    reference, _ = timeit(lambda: imagecodecs.packbits_decode(encoded))
    rows.append(("PackBits decode, 16 MiB", ours, reference))

    ours, _ = timeit(lambda: mt.delta_encode(raw, (4096, 4096, 1), 2))
    reference, _ = timeit(lambda: imagecodecs.delta_encode(image, axis=-1))
    rows.append(("Horizontal predictor, 4096x4096 u16", ours, reference))

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "reference.tif")
        tifffile.imwrite(path, image, compression="deflate", predictor=2, rowsperstrip=128)
        assert np.array_equal(mt.imread(path), image)
        ours, _ = timeit(lambda: mt.imread(path), repeat=3)
        reference, _ = timeit(lambda: tifffile.imread(path), repeat=3)
        rows.append(("Deflate TIFF read, 4096x4096 u16", ours, reference))

        ours_path = os.path.join(directory, "ours.tif")
        ref_path = os.path.join(directory, "upstream.tif")
        ours, _ = timeit(
            lambda: mt.imwrite(ours_path, image, compression="packbits", rowsperstrip=128),
            repeat=3,
        )
        reference, _ = timeit(
            lambda: tifffile.imwrite(ref_path, image, compression="packbits", rowsperstrip=128),
            repeat=3,
        )
        rows.append(("PackBits TIFF write, 4096x4096 u16", ours, reference))

    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-tifffile | upstream | upstream / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, reference in rows:
        ratio = reference / ours
        label = "faster" if ratio > 1 else "slower"
        print(
            f"| {name} | {ours * 1000:.2f} ms | {reference * 1000:.2f} ms | "
            f"{ratio:.2f}x ({label}) |"
        )


if __name__ == "__main__":
    main()
