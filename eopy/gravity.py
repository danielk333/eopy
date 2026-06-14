"""Gravity acceleration kernels.

The fast path implements central gravity plus J2 analytically. A slower generic
spherical-harmonic potential gradient is included for low-degree validation and
for carrying Orekit/Holmes-Featherstone style coefficient data through the API.
For production high-degree fields, replace ``_harmonic_accel_fd`` with a C++ or
Numba Holmes-Featherstone recursion over fully normalized coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import lpmv

from .constants import EARTH_J2, EARTH_MU, EARTH_RADIUS
from .data import GravityModel


@dataclass(slots=True)
class GravityField:
    mu: float = EARTH_MU
    reference_radius: float = EARTH_RADIUS
    j2: float = EARTH_J2
    degree: int = 2
    order: int = 0
    c: np.ndarray | None = None
    s: np.ndarray | None = None
    normalized: bool = False
    use_finite_difference_harmonics: bool = False
    model: GravityModel | None = None
    backend: str = "auto"

    @classmethod
    def from_model(cls, model: GravityModel, *, degree: int | None = None, order: int | None = None, backend: str = "auto") -> "GravityField":
        return cls(
            mu=model.mu,
            reference_radius=model.reference_radius,
            j2=0.0,
            degree=model.max_degree if degree is None else degree,
            order=model.max_order if order is None else order,
            c=model.c,
            s=model.s,
            normalized=model.normalized,
            model=model,
            backend=backend,
        )


def _as_batch_position(position_m: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(position_m, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.shape[-1] != 3:
        raise ValueError("position array must have shape (3,) or (N, 3)")
    return arr, single


def central_gravity(position_m: np.ndarray, mu: float = EARTH_MU) -> np.ndarray:
    r, single = _as_batch_position(position_m)
    r2 = np.sum(r * r, axis=1)
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    out = -mu * r * inv_r3[:, None]
    return out[0] if single else out


def j2_acceleration(position_m: np.ndarray, mu: float = EARTH_MU, radius: float = EARTH_RADIUS, j2: float = EARTH_J2) -> np.ndarray:
    r, single = _as_batch_position(position_m)
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    r2 = x * x + y * y + z * z
    z2_r2 = (z * z) / r2
    factor = 1.5 * j2 * mu * radius * radius / (r2 * r2 * np.sqrt(r2))
    ax = factor * x * (5.0 * z2_r2 - 1.0)
    ay = factor * y * (5.0 * z2_r2 - 1.0)
    az = factor * z * (5.0 * z2_r2 - 3.0)
    out = np.column_stack((ax, ay, az))
    return out[0] if single else out


def _potential_unnormalized(position_m: np.ndarray, field: GravityField) -> np.ndarray:
    r, _ = _as_batch_position(position_m)
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    rho = np.linalg.norm(r, axis=1)
    lat_sin = z / rho
    lon = np.arctan2(y, x)
    u = field.mu / rho

    if field.c is None:
        return u
    if field.normalized:
        raise NotImplementedError("fully normalized coefficient recursion is a TODO for the C++ gravity kernel")

    degree = min(field.degree, field.c.shape[0] - 1)
    order = min(field.order, degree)
    series = np.ones_like(rho)
    rr = field.reference_radius / rho
    rr_n = rr.copy()
    for n in range(1, degree + 1):
        rr_n *= rr if n > 1 else 1.0
        max_m = min(order, n)
        for m in range(max_m + 1):
            c_nm = field.c[n, m]
            s_nm = 0.0 if field.s is None else field.s[n, m]
            if c_nm == 0.0 and s_nm == 0.0:
                continue
            p_nm = lpmv(m, n, lat_sin)
            series += rr_n * p_nm * (c_nm * np.cos(m * lon) + s_nm * np.sin(m * lon))
    return field.mu / rho * series


def _harmonic_accel_fd(position_m: np.ndarray, field: GravityField) -> np.ndarray:
    r, single = _as_batch_position(position_m)
    step = np.maximum(0.1, 1.0e-7 * np.linalg.norm(r, axis=1))
    grad = np.empty_like(r)
    for axis in range(3):
        delta = np.zeros_like(r)
        delta[:, axis] = step
        up = _potential_unnormalized(r + delta, field)
        um = _potential_unnormalized(r - delta, field)
        grad[:, axis] = (up - um) / (2.0 * step)
    out = grad
    return out[0] if single else out


def gravity_acceleration(position_m: np.ndarray, field: GravityField | None = None) -> np.ndarray:
    """Return inertial gravity acceleration for one or many positions.

    TODO: add Earth orientation and body-fixed/inertial transforms before using
    non-zonal coefficients in operational runs.
    TODO: port Orekit's Holmes-Featherstone normalized-recursion gravity kernel
    to a compiled extension for high degree/order fields.
    """

    gf = GravityField() if field is None else field
    degree = min(gf.degree, gf.c.shape[0] - 1) if gf.c is not None else gf.degree
    order = min(gf.order, degree, gf.c.shape[1] - 1) if gf.c is not None else min(gf.order, degree)
    if gf.c is not None and (gf.model is not None or gf.use_finite_difference_harmonics):
        if gf.backend in {"auto", "native"}:
            try:
                from .native import NativeUnavailableError, holmes_featherstone_acceleration

                return holmes_featherstone_acceleration(
                    position_m,
                    degree=degree,
                    order=order,
                    c=gf.c,
                    s=gf.s,
                    mu=gf.mu,
                    reference_radius=gf.reference_radius,
                    normalized=gf.normalized,
                )
            except NativeUnavailableError:
                if gf.backend == "native":
                    raise
        if gf.normalized:
            raise NotImplementedError("normalized spherical harmonics require the native gravity backend")
        return _harmonic_accel_fd(position_m, gf)

    if gf.backend in {"auto", "native"}:
        try:
            from .native import NativeUnavailableError, central_j2_acceleration

            return central_j2_acceleration(position_m, gf.mu, gf.reference_radius, gf.j2)
        except NativeUnavailableError:
            if gf.backend == "native":
                raise
    acc = central_gravity(position_m, gf.mu)
    if gf.j2 and gf.degree >= 2:
        acc = acc + j2_acceleration(position_m, gf.mu, gf.reference_radius, gf.j2)
    if gf.use_finite_difference_harmonics and gf.c is not None and gf.degree > 2:
        acc = _harmonic_accel_fd(position_m, gf)
    return acc
