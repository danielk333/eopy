"""Third-body point-mass perturbation kernels."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from .profiling import PropagationProfile
from .types import NDArray_NxM

PositionProvider = Callable[[float], NDArray_NxM]


@dataclass(slots=True)
class ThirdBody:
    """Point-mass third body with a fixed or time-varying geocentric position."""

    mu: float
    position_m: NDArray_NxM | PositionProvider
    name: str = ""

    def position(self, t: float) -> NDArray_NxM:
        return (
            self.position_m(t)
            if callable(self.position_m)
            else np.asarray(self.position_m, dtype=float)
        )


def _as_batch_position(position_m: NDArray_NxM) -> tuple[NDArray_NxM, bool]:
    arr = np.asarray(position_m, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.shape[-1] != 3:
        raise ValueError("position array must have shape (3,) or (N, 3)")
    return arr, single


def third_body_acceleration(
    position_m: NDArray_NxM, t: float, bodies: ThirdBody | Sequence[ThirdBody]
) -> NDArray_NxM:
    """Return differential third-body acceleration in an Earth-centered frame."""

    r, single = _as_batch_position(position_m)
    body_list = [bodies] if isinstance(bodies, ThirdBody) else list(bodies)
    acc = np.zeros_like(r)
    for body in body_list:
        rb = np.asarray(body.position(t), dtype=float)
        if rb.ndim == 1:
            rb = np.broadcast_to(rb, r.shape)
        if rb.shape != r.shape:
            raise ValueError(
                "third-body position must have shape (3,) or match position batch shape"
            )
        sat_to_body = rb - r
        body_norm = np.linalg.norm(rb, axis=1)
        sat_norm = np.linalg.norm(sat_to_body, axis=1)
        acc += body.mu * (sat_to_body / sat_norm[:, None] ** 3 - rb / body_norm[:, None] ** 3)
    return acc[0] if single else acc


def astropy_geocentric_body_provider(body: str, epoch: datetime) -> PositionProvider:
    """Create a geocentric body-position provider backed by Astropy ephemerides."""

    def provider(t: float) -> NDArray_NxM:
        try:
            from astropy import units as u
            from astropy.coordinates import get_body_barycentric_posvel
            from astropy.time import Time, TimeDelta
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "astropy is required for astropy-backed third-body positions"
            ) from exc

        instant = Time(epoch) + TimeDelta(float(t), format="sec")
        body_pos, _ = get_body_barycentric_posvel(body, instant)
        earth_pos, _ = get_body_barycentric_posvel("earth", instant)
        return (body_pos.xyz - earth_pos.xyz).to_value(u.m)

    return provider


def interpolated_astropy_body_providers(
    bodies: tuple[str, ...] | list[str],
    epoch: datetime,
    t0: float,
    t1: float,
    spacing_s: float = 300.0,
    profile: PropagationProfile | None = None,
) -> dict[str, PositionProvider]:
    """Precompute geocentric body positions over propagation bounds.

    Astropy evaluates all requested epochs in one vectorized operation. Cubic
    splines then provide inexpensive positions in the propagator's GCRS-aligned
    inertial axes. The bounds are inclusive and extrapolation is disabled.
    """

    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    if spacing_s <= 0.0:
        raise ValueError("spacing_s must be positive")

    from astropy import units as u
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time, TimeDelta

    sample_times = np.arange(float(t0), float(t1), float(spacing_s))
    sample_times = np.append(sample_times, float(t1))
    epochs = Time(epoch) + TimeDelta(sample_times, format="sec")

    setup_measure = profile.measure("ephemeris_setup") if profile is not None else nullcontext()
    with setup_measure:
        earth_position, _ = get_body_barycentric_posvel("earth", epochs)
        splines = {}
        for body in bodies:
            body_position, _ = get_body_barycentric_posvel(body, epochs)
            positions_m = (body_position.xyz - earth_position.xyz).to_value(u.m).T
            splines[body] = CubicSpline(sample_times, positions_m, axis=0, extrapolate=False)

    def provider(spline: CubicSpline) -> PositionProvider:
        last_time = None
        last_position = None

        def position(time_s: float) -> NDArray_NxM:
            nonlocal last_time, last_position
            if last_time == time_s and last_position is not None:
                return last_position.copy()
            if profile is None:
                value = np.asarray(spline(time_s))
            else:
                with profile.measure("ephemeris_interpolation"):
                    value = np.asarray(spline(time_s))
            if not np.all(np.isfinite(value)):
                raise ValueError("time is outside ephemeris coverage")
            last_time = time_s
            last_position = value
            return value.copy()

        return position

    return {name: provider(spline) for name, spline in splines.items()}
