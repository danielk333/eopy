"""ctypes access to compiled eopy kernels."""

from __future__ import annotations

import os
import sys
import sysconfig
from ctypes import CDLL, POINTER, c_char_p, c_double, c_int, c_size_t
from pathlib import Path

import numpy as np

from .types import NDArray_NxM


class NativeUnavailableError(RuntimeError):
    pass


class NativeKernelError(RuntimeError):
    pass


_LIB = None
_LOAD_ERROR: Exception | None = None


def _candidate_library_names() -> list[str]:
    if sys.platform.startswith("win"):
        return ["eopykernels.dll", "libeopykernels.dll"]
    if sys.platform == "darwin":
        return ["libeopykernels.dylib"]
    return ["libeopykernels.so"]


def _candidate_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    paths = []
    if "EOPY_NATIVE_LIBRARY" in os.environ:
        paths.append(Path(os.environ["EOPY_NATIVE_LIBRARY"]))
    paths.extend(here / name for name in _candidate_library_names())
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if ext_suffix:
        paths.append(here / f"eopykernels{ext_suffix}")
    return paths


def library_diagnostics() -> str:
    """Return human-readable native-library search diagnostics."""

    lines = ["compiled eopy native kernels are not available", "checked paths:"]
    for path in _candidate_paths():
        lines.append(f"  - {path} ({'exists' if path.exists() else 'missing'})")
    lines.extend(
        [
            "build the source-tree library with:",
            "  cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release",
            "  cmake --build build/native --config Release",
            "or point eopy at an existing build with:",
            "  EOPY_NATIVE_LIBRARY=/path/to/libeopykernels.so",
        ]
    )
    if _LOAD_ERROR is not None:
        lines.append(f"last load error: {_LOAD_ERROR}")
    return "\n".join(lines)


def _load() -> CDLL:
    global _LIB, _LOAD_ERROR
    if _LIB is not None:
        return _LIB
    for path in _candidate_paths():
        if not path.exists():
            continue
        try:
            lib = CDLL(str(path))
            lib.eopy_native_version.restype = c_char_p
            lib.eopy_central_j2_accel_batch.argtypes = [
                c_size_t,
                POINTER(c_double),
                c_double,
                c_double,
                c_double,
                POINTER(c_double),
            ]
            lib.eopy_central_j2_accel_batch.restype = c_int
            lib.eopy_spherical_harmonic_accel.argtypes = [
                c_size_t,
                POINTER(c_double),
                c_size_t,
                c_size_t,
                POINTER(c_double),
                POINTER(c_double),
                c_double,
                c_double,
                c_int,
                POINTER(c_double),
            ]
            lib.eopy_spherical_harmonic_accel.restype = c_int
            _LIB = lib
            return lib
        except (OSError, AttributeError) as exc:
            _LOAD_ERROR = exc
    raise NativeUnavailableError(library_diagnostics()) from _LOAD_ERROR


def available() -> bool:
    try:
        _load()
    except NativeUnavailableError:
        return False
    return True


def version() -> str:
    return _load().eopy_native_version().decode("ascii")


def _as_position_batch(position_m: NDArray_NxM) -> tuple[NDArray_NxM, bool]:
    arr = np.asarray(position_m, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("position array must have shape (3,) or (N, 3)")
    return np.ascontiguousarray(arr), single


def central_j2_acceleration(
    position_m: NDArray_NxM, mu: float, reference_radius: float, j2: float
) -> NDArray_NxM:
    position, single = _as_position_batch(position_m)
    out = np.empty_like(position)
    status = _load().eopy_central_j2_accel_batch(
        position.shape[0],
        position.ctypes.data_as(POINTER(c_double)),
        float(mu),
        float(reference_radius),
        float(j2),
        out.ctypes.data_as(POINTER(c_double)),
    )
    if status:
        raise NativeKernelError(f"eopy_central_j2_accel_batch failed with status {status}")
    return out[0] if single else out


def spherical_harmonic_acceleration(
    position_m: NDArray_NxM,
    degree: int,
    order: int,
    c: NDArray_NxM,
    s: NDArray_NxM | None,
    mu: float,
    reference_radius: float,
    normalized: bool,
) -> NDArray_NxM:
    position, single = _as_position_batch(position_m)
    if degree < 0 or order < 0 or order > degree:
        raise ValueError("require 0 <= order <= degree")
    c_source = np.asarray(c, dtype=np.float64)
    s_source = np.zeros_like(c_source) if s is None else np.asarray(s, dtype=np.float64)
    if c_source.ndim != 2 or s_source.shape != c_source.shape:
        raise ValueError("c and s must be matching two-dimensional arrays")
    c_arr = np.ascontiguousarray(c_source[: degree + 1, : order + 1])
    s_arr = np.ascontiguousarray(s_source[: degree + 1, : order + 1])
    if degree >= c_source.shape[0] or order >= c_source.shape[1]:
        raise ValueError("degree/order exceed coefficient array shape")
    out = np.empty_like(position)
    status = _load().eopy_spherical_harmonic_accel(
        position.shape[0],
        position.ctypes.data_as(POINTER(c_double)),
        int(degree),
        int(order),
        c_arr.ctypes.data_as(POINTER(c_double)),
        s_arr.ctypes.data_as(POINTER(c_double)),
        float(mu),
        float(reference_radius),
        int(bool(normalized)),
        out.ctypes.data_as(POINTER(c_double)),
    )
    if status:
        raise NativeKernelError(f"eopy_spherical_harmonic_accel failed with status {status}")
    return out[0] if single else out
