"""Atmosphere and solar-activity models.

MSIS density is backed by :mod:`pymsis`. The DTM2000 function remains a compact
placeholder until the DTM coefficient tables and Marshall solar activity reader
are added.

TODO: clean up the "like" models
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import math
import csv
from pathlib import Path

import numpy as np
import pymsis
from pymsis import msis as pymsis_api

from .constants import EARTH_RADIUS
from .frames import ecef_to_geodetic_wgs84


_MSIS_OPTIONS = pymsis_api.create_options()


@lru_cache(maxsize=32)
def _msis_epoch_parts(epoch: datetime) -> tuple[float, float, datetime]:
    utc_epoch = epoch.astimezone(timezone.utc).replace(tzinfo=None)
    day = float(utc_epoch.timetuple().tm_yday)
    seconds = float(utc_epoch.hour * 3600 + utc_epoch.minute * 60 + utc_epoch.second)
    return day, seconds, utc_epoch


def _low_level_msis21_available() -> bool:
    library = getattr(pymsis_api, "msis21f", None)
    return all(
        hasattr(library, attribute)
        for attribute in ("_lock", "_last_used_options", "pyinitswitch", "pymsiscalc")
    ) and hasattr(pymsis_api, "_MSIS_PARAMETER_PATH")


def _calculate_msis21_density(
    day: float,
    seconds: float,
    lons: np.ndarray,
    lats: np.ndarray,
    alts_km: np.ndarray,
    activity: SolarActivity,
) -> np.ndarray | None:
    """Call pymsis's compiled MSIS 2.1 extension when its interface is compatible."""

    if not _low_level_msis21_available():
        return None
    count = lons.size
    days = np.full(count, day, dtype=np.float32)
    ut_seconds = np.full(count, seconds, dtype=np.float32)
    longitudes = np.asarray(lons, dtype=np.float32)
    latitudes = np.asarray(lats, dtype=np.float32)
    altitudes = np.asarray(alts_km, dtype=np.float32)
    f107 = np.full(count, activity.f10, dtype=np.float32)
    f107a = np.full(count, activity.f10_mean, dtype=np.float32)
    ap_values = np.asarray(activity.ap, dtype=np.float32)
    if ap_values.ndim == 0:
        aps = np.full((count, 7), float(ap_values), dtype=np.float32, order="F")
    elif ap_values.shape == (7,):
        aps = np.asfortranarray(np.broadcast_to(ap_values, (count, 7)))
    else:
        raise ValueError("ap must be a scalar or a seven-element MSIS history")
    library = pymsis_api.msis21f
    try:
        with library._lock:
            if library._last_used_options != _MSIS_OPTIONS:
                library.pyinitswitch(_MSIS_OPTIONS, parmpath=pymsis_api._MSIS_PARAMETER_PATH)
                library._last_used_options = _MSIS_OPTIONS
            output = library.pymsiscalc(
                days,
                ut_seconds,
                longitudes,
                latitudes,
                altitudes,
                f107,
                f107a,
                aps,
            )
    except (AttributeError, TypeError):
        return None
    if not isinstance(output, np.ndarray) or output.shape != (count, 11):
        return None
    density = np.asarray(output[:, 0], dtype=float)
    density[(density >= 9.9e-38) & (density < 1.0e-37)] = np.nan
    return density


@dataclass(slots=True)
class SolarActivity:
    f10: float = 150.0
    f10_mean: float = 150.0
    ap: float | np.ndarray = 4.0


class CelestrakSpaceWeather:
    """Historical MSIS solar and geomagnetic inputs from ``SW-All.csv``."""

    def __init__(self, rows: dict[date, dict[str, str]]):
        self._rows = rows

    @classmethod
    def from_csv(cls, path: str | Path) -> "CelestrakSpaceWeather":
        with Path(path).open("rt", encoding="utf-8", newline="") as handle:
            rows = {
                datetime.strptime(row["DATE"], "%Y-%m-%d").date(): row
                for row in csv.DictReader(handle)
            }
        return cls(rows)

    def __call__(self, instant: datetime) -> SolarActivity:
        utc = instant.astimezone(timezone.utc)
        row = self._rows.get(utc.date())
        if row is None:
            raise ValueError(f"space weather is unavailable for {utc.date().isoformat()}")
        bin_index = utc.hour // 3
        recent = []
        cursor = utc.date()
        cursor_bin = bin_index
        while len(recent) < 20:
            cursor_row = self._rows.get(cursor)
            if cursor_row is None:
                raise ValueError(f"space weather history is unavailable before {utc.date().isoformat()}")
            recent.append(float(cursor_row[f"AP{cursor_bin + 1}"]))
            cursor_bin -= 1
            if cursor_bin < 0:
                cursor -= timedelta(days=1)
                cursor_bin = 7
        ap = np.asarray(
            [
                float(row["AP_AVG"]),
                recent[0],
                recent[1],
                recent[2],
                recent[3],
                float(np.mean(recent[4:12])),
                float(np.mean(recent[12:20])),
            ]
        )
        return SolarActivity(
            f10=float(row["F10.7_OBS"]),
            f10_mean=float(row["F10.7_OBS_CENTER81"]),
            ap=ap,
        )


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
    ap = float(np.mean(np.asarray(activity.ap)))
    g = 1.0 + 0.012 * max(ap - 4.0, 0.0)
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


def density_msis_like(
    position_m: np.ndarray,
    activity: SolarActivity | None = None,
    config: AtmosphereConfig | None = None,
    *,
    time_s: float = 0.0,
) -> np.ndarray:
    """Return MSIS mass density in kg/m^3 using ``pymsis``.

    Positions are interpreted as Earth-fixed coordinates and converted to WGS84
    geodetic latitude, longitude and altitude.
    TODO: add Orekit-grade inertial/body-fixed frame transforms before passing
    inertial propagation states into this model.
    """

    cfg = AtmosphereConfig(model="msis") if config is None else config
    instant = cfg.epoch + timedelta(seconds=float(time_s))
    act = SolarActivity() if activity is None else (activity(instant) if callable(activity) else activity)
    r = np.asarray(position_m, dtype=float)
    single = r.ndim == 1
    if (single and r.shape != (3,)) or (not single and (r.ndim != 2 or r.shape[1] != 3)):
        raise ValueError("position array must have shape (3,) or (N, 3)")
    if single:
        finite_position = all(math.isfinite(float(value)) for value in r)
    else:
        finite_position = bool(np.all(np.isfinite(r)))
    if not finite_position:
        raise ValueError("MSIS inputs must contain only finite values")
    lats, lons, alts_m = ecef_to_geodetic_wgs84(r, degrees=True)
    if single:
        altitude_km = min(max(alts_m, cfg.min_altitude_m), cfg.max_altitude_m) / 1000.0
        lats = np.asarray([lats])
        lons = np.asarray([lons])
        alts_km = np.asarray([altitude_km])
    else:
        alts_km = np.clip(alts_m, cfg.min_altitude_m, cfg.max_altitude_m) / 1000.0
    if not math.isfinite(act.f10) or not math.isfinite(act.f10_mean) or not np.all(np.isfinite(act.ap)):
        raise ValueError("MSIS inputs must contain only finite values")
    day, seconds, epoch = _msis_epoch_parts(instant)
    rho = _calculate_msis21_density(day, seconds, lons, lats, alts_km, act)
    if rho is None:
        count = lons.size
        dates = [epoch] * count
        f107 = np.full(count, act.f10)
        f107a = np.full(count, act.f10_mean)
        ap_values = np.asarray(act.ap, dtype=float)
        aps = (
            np.full((count, 7), float(ap_values))
            if ap_values.ndim == 0
            else np.broadcast_to(ap_values, (count, 7)).copy()
        )
        out = pymsis.calculate(
            dates,
            lons,
            lats,
            alts_km,
            f107s=f107,
            f107as=f107a,
            aps=aps,
            version=2.1,
        )
        rho = np.asarray(out[..., pymsis.Variable.MASS_DENSITY], dtype=float).reshape(-1)
    return rho[0] if single else rho
