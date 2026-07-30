"""ctypes bindings for the Mojo codec kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_TIFFFILE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-tifffile.so"
)
I = ctypes.c_int64


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "tifffile.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    pixi = shutil.which("pixi")
    command = (
        [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
        if pixi
        else ["bash", os.path.join(ROOT, "build", "build.sh")]
    )
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_instance: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _instance
    if _instance is None:
        _instance = ctypes.CDLL(build())
        signatures = {
            "mti_predictor": ([I, I, I, I, I, I, I], I),
            "mti_predictor_copy": ([I, I, I, I, I, I, I, I], I),
            "mti_packbits_encode": ([I, I, I, I], I),
            "mti_packbits_decode": ([I, I, I, I], I),
            "mti_reverse_bits": ([I, I], I),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(_instance, name)
            function.argtypes = argtypes
            function.restype = restype
    return _instance
