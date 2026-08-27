"""Frame and time helpers.

TODO: some of these are already implemented in spacecoords, can use those, some are dubious and
straight up copy-pasta from elsewhere - needs validation
"""

from __future__ import annotations

from datetime import datetime
import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.257_223_563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_POLE_LIMIT_M = 1.0e-9


@dataclass(slots=True)
class EarthOrientation:
    """Interpolated GCRS-to-ITRS rotation and rotation derivative."""

    _rotation: CubicSpline

    def rotation(self, time_s: float) -> np.ndarray:
        return np.asarray(self._rotation(float(time_s)), dtype=float)

    def rotation_rate(self, time_s: float) -> np.ndarray:
        return np.asarray(self._rotation(float(time_s), 1), dtype=float)

    def position_to_fixed(self, time_s: float, position_m: np.ndarray) -> np.ndarray:
        return np.asarray(position_m) @ self.rotation(time_s).T

    def state_to_fixed(
        self,
        time_s: float,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = self.rotation(time_s)
        rotation_rate = self.rotation_rate(time_s)
        position = np.asarray(position_m)
        velocity = np.asarray(velocity_mps)
        return position @ rotation.T, velocity @ rotation.T + position @ rotation_rate.T

    def vector_to_inertial(self, time_s: float, fixed_vector: np.ndarray) -> np.ndarray:
        return np.asarray(fixed_vector) @ self.rotation(time_s)


def interpolated_earth_orientation(
    epoch: datetime,
    t0: float,
    t1: float,
    *,
    spacing_s: float = 60.0,
) -> EarthOrientation:
    """Build an Astropy/spacecoords-backed GCRS-to-ITRS rotation provider."""

    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    if spacing_s <= 0.0:
        raise ValueError("spacing_s must be positive")
    from astropy.time import Time, TimeDelta
    from spacecoords import celestial

    sample_times = np.arange(float(t0), float(t1), float(spacing_s))
    sample_times = np.append(sample_times, float(t1))
    epochs = Time(epoch) + TimeDelta(sample_times, format="sec")
    rotations = np.empty((sample_times.size, 3, 3), dtype=float)
    for axis in range(3):
        states = np.zeros((6, sample_times.size), dtype=float)
        states[axis] = 1.0
        transformed = celestial.convert(epochs, states, "GCRS", "ITRS")
        rotations[:, :, axis] = transformed[:3].T
    return EarthOrientation(CubicSpline(sample_times, rotations, axis=0, extrapolate=False))


def _ecef_scalar_to_geodetic_wgs84(position_m: np.ndarray, degrees: bool) -> tuple[float, float, float]:
    """Fast scalar ECEF-to-geodetic conversion for propagator hot paths."""

    x, y, z = (float(value) for value in position_m)
    horizontal = math.hypot(x, y)
    if horizontal <= _WGS84_POLE_LIMIT_M:
        latitude = math.copysign(math.pi / 2.0, z) if z != 0.0 else 0.0
        longitude = 0.0
        altitude = abs(z) - _WGS84_B
    else:
        longitude = math.atan2(y, x)
        latitude = math.atan2(z, horizontal * (1.0 - _WGS84_E2))
        for _ in range(3):
            sin_latitude = math.sin(latitude)
            prime_vertical = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_latitude * sin_latitude)
            latitude = math.atan2(z + _WGS84_E2 * prime_vertical * sin_latitude, horizontal)
        sin_latitude = math.sin(latitude)
        prime_vertical = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_latitude * sin_latitude)
        altitude = horizontal / math.cos(latitude) - prime_vertical
    if degrees:
        latitude = math.degrees(latitude)
        longitude = math.degrees(longitude)
    return latitude, longitude, altitude


def ecef_to_geodetic_wgs84(position_m: np.ndarray, *, degrees: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF Cartesian positions to WGS84 latitude, longitude and altitude."""

    r = np.asarray(position_m, dtype=float)
    single = r.ndim == 1
    if single:
        if r.shape != (3,):
            raise ValueError("position array must have shape (3,) or (N, 3)")
        return _ecef_scalar_to_geodetic_wgs84(r, degrees)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("position array must have shape (3,) or (N, 3)")

    x, y, z = r.T
    horizontal = np.hypot(x, y)
    longitude = np.arctan2(y, x)
    latitude = np.arctan2(z, horizontal * (1.0 - _WGS84_E2))
    for _ in range(3):
        sin_latitude = np.sin(latitude)
        prime_vertical = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_latitude * sin_latitude)
        latitude = np.arctan2(z + _WGS84_E2 * prime_vertical * sin_latitude, horizontal)
    sin_latitude = np.sin(latitude)
    prime_vertical = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_latitude * sin_latitude)
    with np.errstate(divide="ignore", invalid="ignore"):
        altitude = horizontal / np.cos(latitude) - prime_vertical

    poles = horizontal <= _WGS84_POLE_LIMIT_M
    if np.any(poles):
        pole_latitude = np.sign(z) * (np.pi / 2.0)
        latitude = np.where(poles, pole_latitude, latitude)
        longitude = np.where(poles, 0.0, longitude)
        altitude = np.where(poles, np.abs(z) - _WGS84_B, altitude)
    if degrees:
        latitude = np.degrees(latitude)
        longitude = np.degrees(longitude)
    return latitude, longitude, altitude


def astropy_time(epoch: datetime, offset_seconds: float = 0.0):
    """Return an Astropy ``Time`` for an epoch plus propagation seconds."""

    try:
        from astropy.time import Time, TimeDelta
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("astropy is required for astropy-backed time handling") from exc
    return Time(epoch) + TimeDelta(float(offset_seconds), format="sec")
