"""Frame and time helper wrappers backed by spacecoords and Astropy."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from spacecoords import frames as spacecoords_frames


def ecef_to_geodetic_wgs84(position_m: np.ndarray, *, degrees: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF Cartesian positions to WGS84 latitude, longitude and altitude."""

    r = np.asarray(position_m, dtype=float)
    single = r.ndim == 1
    if single:
        r = r[None, :]
    lats, lons, alts_m = spacecoords_frames.ecef_to_geodetic_wgs84(r[:, 0], r[:, 1], r[:, 2], degrees=degrees)
    if single:
        return lats[0], lons[0], alts_m[0]
    return lats, lons, alts_m


def astropy_time(epoch: datetime, offset_seconds: float = 0.0):
    """Return an Astropy ``Time`` for an epoch plus propagation seconds."""

    try:
        from astropy.time import Time, TimeDelta
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("astropy is required for astropy-backed time handling") from exc
    return Time(epoch) + TimeDelta(float(offset_seconds), format="sec")
