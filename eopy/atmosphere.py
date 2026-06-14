"""Atmosphere and solar-activity models.

MSIS density is backed by :mod:`pymsis`. The DTM2000 function remains a compact
placeholder until the DTM coefficient tables and Marshall solar activity reader
are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pymsis

from .constants import EARTH_RADIUS
from .frames import ecef_to_geodetic_wgs84


@dataclass(slots=True)
class SolarActivity:
    f10: float = 150.0
    f10_mean: float = 150.0
    ap: float = 4.0


@dataclass(slots=True)
class AtmosphereConfig:
    model: str = "dtm2000"
    epoch: datetime = datetime(2003, 3, 5, 0, 24, 0, tzinfo=timezone.utc)
    min_altitude_m: float = 120_000.0
    max_altitude_m: float = 1_000_000.0
    rho_ref: float = 2.5e-11
    h_ref_m: float = 400_000.0
    scale_height_m: float = 58_000.0
    earth_radius_m: float = EARTH_RADIUS
    co_rotate: bool = True


def _activity_scale(activity: SolarActivity) -> float:
    f = 1.0 + 0.0025 * (activity.f10_mean - 150.0) + 0.0015 * (activity.f10 - 150.0)
    g = 1.0 + 0.012 * max(activity.ap - 4.0, 0.0)
    return float(np.clip(f * g, 0.2, 8.0))


def density_dtm2000_like(position_m: np.ndarray, activity: SolarActivity | None = None, config: AtmosphereConfig | None = None) -> np.ndarray:
    """Approximate DTM2000 density shape with Marshall-style solar inputs.

    TODO: implement complete DTM2000 temperature/composition equations and wire
    a Marshall Solar Activity Future Estimation data reader.
    TODO: include local solar time, day-of-year, geodetic latitude and sun-vector
    dependencies; this placeholder only responds to altitude and activity.
    """

    act = SolarActivity() if activity is None else activity
    cfg = AtmosphereConfig(model="dtm2000") if config is None else config
    r = np.asarray(position_m, dtype=float)
    if r.ndim == 1:
        r = r[None, :]
    altitude = np.linalg.norm(r, axis=1) - cfg.earth_radius_m
    altitude = np.clip(altitude, cfg.min_altitude_m, cfg.max_altitude_m)
    rho = cfg.rho_ref * np.exp(-(altitude - cfg.h_ref_m) / cfg.scale_height_m)
    return rho * _activity_scale(act)


def density_msis_like(position_m: np.ndarray, activity: SolarActivity | None = None, config: AtmosphereConfig | None = None) -> np.ndarray:
    """Return MSIS mass density in kg/m^3 using ``pymsis``.

    Positions are interpreted as Earth-fixed coordinates and converted to WGS84
    geodetic latitude, longitude and altitude with ``spacecoords``.
    TODO: add Orekit-grade inertial/body-fixed frame transforms before passing
    inertial propagation states into this model.
    """

    act = SolarActivity() if activity is None else activity
    cfg = AtmosphereConfig(model="msis") if config is None else config
    r = np.asarray(position_m, dtype=float)
    single = r.ndim == 1
    if single:
        r = r[None, :]
    lats, lons, alts_m = ecef_to_geodetic_wgs84(r, degrees=True)
    alts_km = np.clip(alts_m, cfg.min_altitude_m, cfg.max_altitude_m) / 1000.0
    epoch = cfg.epoch.astimezone(timezone.utc).replace(tzinfo=None)
    dates = [epoch + timedelta(seconds=0.0)] * r.shape[0]
    f107 = np.full(r.shape[0], act.f10)
    f107a = np.full(r.shape[0], act.f10_mean)
    aps = np.full(r.shape[0], act.ap)
    out = pymsis.calculate(dates, lons, lats, alts_km, f107s=f107, f107as=f107a, aps=aps, version=2.1)
    rho = np.asarray(out[..., pymsis.Variable.MASS_DENSITY], dtype=float).reshape(-1)
    return rho[0] if single else rho
