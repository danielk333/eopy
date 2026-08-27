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
    shadow_model: str | None = None


SUN_RADIUS_M = 695_700_000.0


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


def conical_shadow_factor(
    position_m: np.ndarray,
    sun_position_m: np.ndarray,
    earth_radius_m: float = EARTH_RADIUS,
    sun_radius_m: float = SUN_RADIUS_M,
) -> np.ndarray:
    """Return the visible fraction of the solar disc through umbra/penumbra."""

    r = np.asarray(position_m, dtype=float)
    single = r.ndim == 1
    if single:
        r = r[None, :]
    sun = np.asarray(sun_position_m, dtype=float)
    if sun.ndim == 1:
        sun = np.broadcast_to(sun, r.shape)
    satellite_to_sun = sun - r
    satellite_to_earth = -r
    sun_distance = np.linalg.norm(satellite_to_sun, axis=1)
    earth_distance = np.linalg.norm(satellite_to_earth, axis=1)
    sun_angle = np.arcsin(np.clip(sun_radius_m / sun_distance, 0.0, 1.0))
    earth_angle = np.arcsin(np.clip(earth_radius_m / earth_distance, 0.0, 1.0))
    cosine = np.sum(satellite_to_sun * satellite_to_earth, axis=1) / (
        sun_distance * earth_distance
    )
    separation = np.arccos(np.clip(cosine, -1.0, 1.0))
    visible = np.ones(r.shape[0], dtype=float)
    total = separation >= earth_angle + sun_angle
    hidden = separation <= earth_angle - sun_angle
    partial = ~(total | hidden)
    visible[hidden] = 0.0
    if np.any(partial):
        d = separation[partial]
        rs = sun_angle[partial]
        re = earth_angle[partial]
        sun_sector = rs * rs * np.arccos(np.clip((d * d + rs * rs - re * re) / (2.0 * d * rs), -1.0, 1.0))
        earth_sector = re * re * np.arccos(np.clip((d * d + re * re - rs * rs) / (2.0 * d * re), -1.0, 1.0))
        lens = 0.5 * np.sqrt(
            np.maximum(0.0, (-d + rs + re) * (d + rs - re) * (d - rs + re) * (d + rs + re))
        )
        visible[partial] = 1.0 - (sun_sector + earth_sector - lens) / (np.pi * rs * rs)
    return visible[0] if single else visible


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
    shadow_model = radiation.shadow_model
    if shadow_model is None:
        shadow_model = "cylindrical" if radiation.cylindrical_shadow else "none"
    if shadow_model == "cylindrical":
        pressure = pressure * cylindrical_shadow_factor(r, sun, radiation.earth_radius_m)
    elif shadow_model == "conical":
        pressure = pressure * conical_shadow_factor(r, sun, radiation.earth_radius_m)
    elif shadow_model != "none":
        raise ValueError(f"unknown shadow model {shadow_model!r}")

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
