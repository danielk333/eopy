"""Solar radiation pressure kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import ASTRONOMICAL_UNIT, EARTH_RADIUS, SOLAR_RADIATION_PRESSURE_1AU


@dataclass(slots=True)
class RadiationPressure:
    area_m2: float | np.ndarray = 0.0
    reflection_coefficient: float | np.ndarray = 1.0
    mass_kg: float | np.ndarray = 1.0
    global_factor: float | np.ndarray = 1.0
    solar_pressure_1au: float = SOLAR_RADIATION_PRESSURE_1AU
    earth_radius_m: float = EARTH_RADIUS
    cylindrical_shadow: bool = True


def cylindrical_shadow_factor(position_m: np.ndarray, sun_position_m: np.ndarray, earth_radius_m: float = EARTH_RADIUS) -> np.ndarray:
    """Return 0 in Earth cylindrical umbra, 1 otherwise."""

    r = np.asarray(position_m, dtype=float)
    single = r.ndim == 1
    if single:
        r = r[None, :]
    sun = np.asarray(sun_position_m, dtype=float)
    if sun.ndim == 1:
        sun = np.broadcast_to(sun, r.shape)
    sun_hat = sun / np.linalg.norm(sun, axis=1)[:, None]
    along_sun = np.sum(r * sun_hat, axis=1)
    perpendicular = r - along_sun[:, None] * sun_hat
    in_shadow = (along_sun < 0.0) & (np.linalg.norm(perpendicular, axis=1) < earth_radius_m)
    factor = np.where(in_shadow, 0.0, 1.0)
    return factor[0] if single else factor


def solar_radiation_pressure_acceleration(
    position_m: np.ndarray,
    sun_position_m: np.ndarray,
    radiation: RadiationPressure,
) -> np.ndarray:
    """Isotropic SRP acceleration following Orekit's single-coefficient model."""

    r = np.asarray(position_m, dtype=float)
    single = r.ndim == 1
    if single:
        r = r[None, :]
    sun = np.asarray(sun_position_m, dtype=float)
    if sun.ndim == 1:
        sun = np.broadcast_to(sun, r.shape)

    sun_to_sat = r - sun
    distance = np.linalg.norm(sun_to_sat, axis=1)
    direction = sun_to_sat / distance[:, None]
    pressure = radiation.solar_pressure_1au * (ASTRONOMICAL_UNIT / distance) ** 2
    if radiation.cylindrical_shadow:
        pressure = pressure * cylindrical_shadow_factor(r, sun, radiation.earth_radius_m)

    coeff = (
        np.asarray(radiation.global_factor)
        * np.asarray(radiation.area_m2)
        * np.asarray(radiation.reflection_coefficient)
        / np.asarray(radiation.mass_kg)
    )
    if coeff.ndim == 0:
        coeff = np.full(r.shape[0], float(coeff))
    acc = pressure[:, None] * coeff[:, None] * direction
    return acc[0] if single else acc
