"""Third-body point-mass perturbation kernels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np


PositionProvider = Callable[[float], np.ndarray]


@dataclass(slots=True)
class ThirdBody:
    """Point-mass third body with a fixed or time-varying geocentric position."""

    mu: float
    position_m: np.ndarray | PositionProvider
    name: str = ""

    def position(self, t: float) -> np.ndarray:
        return self.position_m(t) if callable(self.position_m) else np.asarray(self.position_m, dtype=float)


def _as_batch_position(position_m: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(position_m, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.shape[-1] != 3:
        raise ValueError("position array must have shape (3,) or (N, 3)")
    return arr, single


def third_body_acceleration(position_m: np.ndarray, t: float, bodies: ThirdBody | list[ThirdBody] | tuple[ThirdBody, ...]) -> np.ndarray:
    """Return differential third-body acceleration in an Earth-centered frame."""

    r, single = _as_batch_position(position_m)
    body_list = [bodies] if isinstance(bodies, ThirdBody) else list(bodies)
    acc = np.zeros_like(r)
    for body in body_list:
        rb = np.asarray(body.position(t), dtype=float)
        if rb.ndim == 1:
            rb = np.broadcast_to(rb, r.shape)
        if rb.shape != r.shape:
            raise ValueError("third-body position must have shape (3,) or match position batch shape")
        sat_to_body = rb - r
        body_norm = np.linalg.norm(rb, axis=1)
        sat_norm = np.linalg.norm(sat_to_body, axis=1)
        acc += body.mu * (sat_to_body / sat_norm[:, None] ** 3 - rb / body_norm[:, None] ** 3)
    return acc[0] if single else acc


def astropy_geocentric_body_provider(body: str, epoch: datetime) -> PositionProvider:
    """Create a geocentric body-position provider backed by Astropy ephemerides."""

    def provider(t: float) -> np.ndarray:
        try:
            from astropy import units as u
            from astropy.coordinates import get_body_barycentric_posvel
            from astropy.time import Time, TimeDelta
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("astropy is required for astropy-backed third-body positions") from exc

        instant = Time(epoch) + TimeDelta(float(t), format="sec")
        body_pos, _ = get_body_barycentric_posvel(body, instant)
        earth_pos, _ = get_body_barycentric_posvel("earth", instant)
        return (body_pos.xyz - earth_pos.xyz).to_value(u.m)

    return provider
