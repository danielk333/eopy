"""Batch Cartesian orbit propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .atmosphere import (
    AtmosphereConfig,
    SolarActivity,
    density_dtm2000_like,
    density_msis_like,
)
from .constants import EARTH_OMEGA
from .gravity import GravityField, gravity_acceleration
from .integrators import Dop853Options, Event, integrate_dop853
from .radiation import RadiationPressure, solar_radiation_pressure_acceleration
from .third_body import ThirdBody, third_body_acceleration


@dataclass(slots=True)
class SpacecraftProperties:
    mass_kg: float | np.ndarray = 1.0
    drag_area_m2: float | np.ndarray = 0.0
    drag_coefficient: float | np.ndarray = 2.2
    ballistic_coefficient_m2_per_kg: float | np.ndarray | None = None


def _as_batch_state(state: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(state, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.shape[-1] != 6:
        raise ValueError("state must have shape (6,) or (N, 6)")
    return arr, single


def drag_acceleration(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    spacecraft: SpacecraftProperties,
    atmosphere: AtmosphereConfig | None,
    activity: SolarActivity | None,
) -> np.ndarray:
    cfg = AtmosphereConfig() if atmosphere is None else atmosphere
    if cfg.model.lower() == "msis":
        rho = density_msis_like(position_m, activity, cfg)
    elif cfg.model.lower() == "dtm2000":
        rho = density_dtm2000_like(position_m, activity, cfg)
    else:
        raise ValueError(f"unknown atmosphere model {cfg.model!r}")

    omega_cross_r = np.column_stack((-EARTH_OMEGA * position_m[:, 1], EARTH_OMEGA * position_m[:, 0], np.zeros(position_m.shape[0])))
    v_rel = velocity_mps - omega_cross_r if cfg.co_rotate else velocity_mps
    speed = np.linalg.norm(v_rel, axis=1)
    if spacecraft.ballistic_coefficient_m2_per_kg is None:
        ballistic = np.asarray(spacecraft.drag_coefficient) * np.asarray(spacecraft.drag_area_m2) / np.asarray(spacecraft.mass_kg)
    else:
        ballistic = np.asarray(spacecraft.ballistic_coefficient_m2_per_kg)
    if ballistic.ndim == 0:
        ballistic = np.full(position_m.shape[0], float(ballistic))
    return -0.5 * rho[:, None] * ballistic[:, None] * speed[:, None] * v_rel


def make_rhs(
    *,
    gravity: GravityField | None = None,
    spacecraft: SpacecraftProperties | None = None,
    atmosphere: AtmosphereConfig | None = None,
    activity: SolarActivity | None = None,
    radiation: RadiationPressure | None = None,
    sun_position: Callable[[float], np.ndarray] | np.ndarray | None = None,
    third_bodies: ThirdBody | Sequence[ThirdBody] | None = None,
    extra_acceleration: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    props = SpacecraftProperties() if spacecraft is None else spacecraft

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        y, single = _as_batch_state(state)
        position = y[:, :3]
        velocity = y[:, 3:]
        acc = gravity_acceleration(position, gravity)
        has_ballistic = props.ballistic_coefficient_m2_per_kg is not None and np.asarray(props.ballistic_coefficient_m2_per_kg).any()
        if has_ballistic or np.asarray(props.drag_area_m2).any():
            acc = acc + drag_acceleration(position, velocity, props, atmosphere, activity)
        if radiation is not None and np.asarray(radiation.area_m2).any():
            if sun_position is None:
                raise ValueError("sun_position is required when radiation is enabled")
            sun = sun_position(t) if callable(sun_position) else sun_position
            acc = acc + solar_radiation_pressure_acceleration(position, sun, radiation)
        if third_bodies is not None:
            acc = acc + third_body_acceleration(position, t, third_bodies)
        if extra_acceleration is not None:
            acc = acc + extra_acceleration(t, position, velocity)
        dydt = np.empty_like(y)
        dydt[:, :3] = velocity
        dydt[:, 3:] = acc
        return dydt[0] if single else dydt

    return rhs


def propagate(
    state0: np.ndarray,
    t0: float,
    t1: float,
    *,
    gravity: GravityField | None = None,
    spacecraft: SpacecraftProperties | None = None,
    atmosphere: AtmosphereConfig | None = None,
    activity: SolarActivity | None = None,
    radiation: RadiationPressure | None = None,
    sun_position: Callable[[float], np.ndarray] | np.ndarray | None = None,
    third_bodies: ThirdBody | Sequence[ThirdBody] | None = None,
    options: Dop853Options | None = None,
    return_history: bool = False,
    events: Event | Sequence[Event] | None = None,
    extra_acceleration: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None,
):
    """Propagate one state ``(6,)`` or many states ``(N, 6)``.

    TODO: add solid/ocean tides, relativity, conical penumbra, maneuvers,
    covariance/STM propagation and Orekit-grade frame/time-scale handling.
    """

    state, single = _as_batch_state(state0)
    rhs = make_rhs(
        gravity=gravity,
        spacecraft=spacecraft,
        atmosphere=atmosphere,
        activity=activity,
        radiation=radiation,
        sun_position=sun_position,
        third_bodies=third_bodies,
        extra_acceleration=extra_acceleration,
    )
    result = integrate_dop853(rhs, t0, state, t1, options=options, return_history=return_history, events=events)
    if return_history:
        ts, ys, stats = result
        if single:
            ys = ys[:, 0, :]
        return ts, ys, stats
    tf, yf, stats = result
    if single:
        yf = yf[0]
    return tf, yf, stats
