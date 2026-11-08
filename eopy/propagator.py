"""Batch Cartesian orbit propagation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Sequence

import numpy as np

from .atmosphere import (
    AtmosphereConfig,
    SolarActivityProvider,
    density_dtm2000,
    density_exponential,
    density_msis,
)
from .constants import EARTH_OMEGA
from .frames import EarthOrientation
from .gravity import GravityField, gravity_acceleration
from .integrators import Dop853Options, Dop853Stats, Event, integrate_dop853
from .profiling import PropagationProfile
from .radiation import RadiationPressure, solar_radiation_pressure_acceleration
from .relativity import schwarzschild_acceleration
from .third_body import ThirdBody, third_body_acceleration
from .types import NDArray_NxM


@dataclass(slots=True)
class SpacecraftProperties:
    mass_kg: float | NDArray_NxM = 1.0
    drag_area_m2: float | NDArray_NxM = 0.0
    drag_coefficient: float | NDArray_NxM = 2.2
    ballistic_coefficient_m2_per_kg: float | NDArray_NxM | None = None


def _as_batch_state(state: NDArray_NxM) -> tuple[NDArray_NxM, bool]:
    arr = np.asarray(state, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.shape[-1] != 6:
        raise ValueError("state must have shape (6,) or (N, 6)")
    return arr, single


def drag_acceleration(
    position_m: NDArray_NxM,
    velocity_mps: NDArray_NxM,
    spacecraft: SpacecraftProperties,
    atmosphere: AtmosphereConfig | None,
    activity: SolarActivityProvider | None,
    profile: PropagationProfile | None = None,
    time_s: float = 0.0,
    earth_orientation: EarthOrientation | None = None,
    sun_position_m: NDArray_NxM | None = None,
) -> NDArray_NxM:
    cfg = AtmosphereConfig() if atmosphere is None else atmosphere
    force_position = position_m
    force_velocity = velocity_mps
    if earth_orientation is not None:
        fixed_position, fixed_velocity = earth_orientation.state_to_fixed(
            time_s, position_m, velocity_mps
        )
        force_position = fixed_position
        force_velocity = (
            fixed_velocity
            if cfg.co_rotate
            else np.asarray(velocity_mps) @ earth_orientation.rotation(time_s).T
        )
    density_measure = (
        profile.measure("atmosphere_density") if profile is not None else nullcontext()
    )
    with density_measure:
        if cfg.model.lower() == "msis":
            rho = density_msis(force_position, activity, cfg, time_s=time_s)
        elif cfg.model.lower() == "dtm2000":
            if sun_position_m is None:
                raise ValueError("sun_position is required for DTM2000")
            fixed_sun = np.asarray(sun_position_m)
            if earth_orientation is not None:
                fixed_sun = earth_orientation.position_to_fixed(time_s, fixed_sun)
            rho = density_dtm2000(
                force_position,
                activity,
                cfg,
                time_s=time_s,
                sun_position_m=fixed_sun,
            )
        elif cfg.model.lower() == "exponential":
            rho = density_exponential(
                force_position,
                activity(cfg.epoch + timedelta(seconds=time_s)) if callable(activity) else activity,
                cfg,
            )
        else:
            raise ValueError(f"unknown atmosphere model {cfg.model!r}")

    kinematics_measure = (
        profile.measure("drag_kinematics") if profile is not None else nullcontext()
    )
    with kinematics_measure:
        if earth_orientation is None:
            omega_cross_r = np.column_stack(
                (
                    -EARTH_OMEGA * position_m[:, 1],
                    EARTH_OMEGA * position_m[:, 0],
                    np.zeros(position_m.shape[0]),
                )
            )
            v_rel = velocity_mps - omega_cross_r if cfg.co_rotate else velocity_mps
        else:
            v_rel = force_velocity
        speed = np.linalg.norm(v_rel, axis=1)
        if spacecraft.ballistic_coefficient_m2_per_kg is None:
            ballistic = (
                np.asarray(spacecraft.drag_coefficient)
                * np.asarray(spacecraft.drag_area_m2)
                / np.asarray(spacecraft.mass_kg)
            )
        else:
            ballistic = np.asarray(spacecraft.ballistic_coefficient_m2_per_kg)
        if ballistic.ndim == 0:
            ballistic = np.full(position_m.shape[0], float(ballistic))
        acceleration = -0.5 * np.asarray(rho)[:, None] * ballistic[:, None] * speed[:, None] * v_rel
        if earth_orientation is not None:
            acceleration = earth_orientation.vector_to_inertial(time_s, acceleration)
        return acceleration


def make_rhs(
    gravity: GravityField | None = None,
    spacecraft: SpacecraftProperties | None = None,
    atmosphere: AtmosphereConfig | None = None,
    activity: SolarActivityProvider | None = None,
    radiation: RadiationPressure | None = None,
    sun_position: Callable[[float], NDArray_NxM] | NDArray_NxM | None = None,
    third_bodies: ThirdBody | Sequence[ThirdBody] | None = None,
    extra_acceleration: Callable[[float, NDArray_NxM, NDArray_NxM], NDArray_NxM] | None = None,
    earth_orientation: EarthOrientation | None = None,
    relativity: bool = False,
    profile: PropagationProfile | None = None,
) -> Callable[[float, NDArray_NxM], NDArray_NxM]:
    props = SpacecraftProperties() if spacecraft is None else spacecraft
    atmosphere_model = (AtmosphereConfig() if atmosphere is None else atmosphere).model.lower()
    has_drag = (
        props.ballistic_coefficient_m2_per_kg is not None
        and np.asarray(props.ballistic_coefficient_m2_per_kg).any()
    ) or np.asarray(props.drag_area_m2).any()
    if has_drag and atmosphere_model == "dtm2000":
        if earth_orientation is None:
            raise ValueError("earth_orientation is required for DTM2000 propagation")
        if sun_position is None:
            raise ValueError("sun_position is required for DTM2000 propagation")

    def rhs(t: float, state: NDArray_NxM) -> NDArray_NxM:
        y, single = _as_batch_state(state)
        position = y[:, :3]
        velocity = y[:, 3:]
        sun = None
        if sun_position is not None and (atmosphere_model == "dtm2000" or radiation is not None):
            if profile is None:
                sun = sun_position(t) if callable(sun_position) else sun_position
            else:
                with profile.measure("sun_position"):
                    sun = sun_position(t) if callable(sun_position) else sun_position
        if profile is None:
            if earth_orientation is not None and gravity is not None and gravity.c is not None:
                fixed_position = earth_orientation.position_to_fixed(t, position)
                fixed_acceleration = gravity_acceleration(fixed_position, gravity)
                acc = earth_orientation.vector_to_inertial(t, fixed_acceleration)
            else:
                acc = gravity_acceleration(position, gravity)
        else:
            with profile.measure("gravity"):
                if earth_orientation is not None and gravity is not None and gravity.c is not None:
                    fixed_position = earth_orientation.position_to_fixed(t, position)
                    fixed_acceleration = gravity_acceleration(fixed_position, gravity)
                    acc = earth_orientation.vector_to_inertial(t, fixed_acceleration)
                else:
                    acc = gravity_acceleration(position, gravity)
        if has_drag:
            if profile is None:
                drag = drag_acceleration(
                    position,
                    velocity,
                    props,
                    atmosphere,
                    activity,
                    time_s=t,
                    earth_orientation=earth_orientation,
                    sun_position_m=sun,
                )
            else:
                with profile.measure("drag_atmosphere"):
                    drag = drag_acceleration(
                        position,
                        velocity,
                        props,
                        atmosphere,
                        activity,
                        profile,
                        time_s=t,
                        earth_orientation=earth_orientation,
                        sun_position_m=sun,
                    )
            acc = acc + drag
        if radiation is not None and np.asarray(radiation.area_m2).any():
            if sun is None:
                raise ValueError("sun_position is required when radiation is enabled")
            if profile is None:
                srp = solar_radiation_pressure_acceleration(position, sun, radiation)
            else:
                with profile.measure("solar_radiation_pressure"):
                    srp = solar_radiation_pressure_acceleration(position, sun, radiation)
            acc = acc + srp
        if third_bodies is not None:
            if profile is None:
                third_body = third_body_acceleration(position, t, third_bodies)
            else:
                with profile.measure("third_bodies"):
                    third_body = third_body_acceleration(position, t, third_bodies)
            acc = acc + third_body
        if relativity:
            acc = acc + schwarzschild_acceleration(position, velocity)
        if extra_acceleration is not None:
            if profile is None:
                extra = extra_acceleration(t, position, velocity)
            else:
                with profile.measure("extra_acceleration"):
                    extra = extra_acceleration(t, position, velocity)
            acc = acc + extra
        dydt = np.empty_like(y)
        dydt[:, :3] = velocity
        dydt[:, 3:] = acc
        return dydt[0] if single else dydt

    return rhs


def propagate(
    state0: NDArray_NxM,
    t0: float,
    t1: float,
    gravity: GravityField | None = None,
    spacecraft: SpacecraftProperties | None = None,
    atmosphere: AtmosphereConfig | None = None,
    activity: SolarActivityProvider | None = None,
    radiation: RadiationPressure | None = None,
    sun_position: Callable[[float], NDArray_NxM] | NDArray_NxM | None = None,
    third_bodies: ThirdBody | Sequence[ThirdBody] | None = None,
    options: Dop853Options | None = None,
    return_history: bool = False,
    sample_times: NDArray_NxM | None = None,
    events: Event | Sequence[Event] | None = None,
    extra_acceleration: Callable[[float, NDArray_NxM, NDArray_NxM], NDArray_NxM] | None = None,
    profile: PropagationProfile | bool | None = None,
    earth_orientation: EarthOrientation | None = None,
    relativity: bool = False,
) -> tuple[float | NDArray_NxM, NDArray_NxM, Dop853Stats]:
    """Propagate one state ``(6,)`` or many states ``(N, 6)``.

    Solid/ocean tides, maneuver scheduling, and covariance/STM propagation are
    intentionally outside the compact force-model interface.
    """

    state, single = _as_batch_state(state0)
    profiler = PropagationProfile() if profile is True else (None if profile is False else profile)
    rhs = make_rhs(
        gravity=gravity,
        spacecraft=spacecraft,
        atmosphere=atmosphere,
        activity=activity,
        radiation=radiation,
        sun_position=sun_position,
        third_bodies=third_bodies,
        extra_acceleration=extra_acceleration,
        earth_orientation=earth_orientation,
        relativity=relativity,
        profile=profiler,
    )
    if profiler is None:
        result = integrate_dop853(
            rhs,
            t0,
            state,
            t1,
            options=options,
            return_history=return_history,
            sample_times=sample_times,
            events=events,
        )
    else:
        with profiler.measure("propagation"):
            result = integrate_dop853(
                rhs,
                t0,
                state,
                t1,
                options=options,
                return_history=return_history,
                sample_times=sample_times,
                events=events,
            )
        result[-1].profile = profiler
    stats = result[-1]
    if single and stats.event_states is not None:
        stats.event_states = tuple(states[:, 0, :] for states in stats.event_states)
    if return_history or sample_times is not None:
        ts, ys, stats = result
        if single:
            ys = ys[:, 0, :]
        return ts, ys, stats
    tf, yf, stats = result
    if single:
        yf = yf[0]
    return tf, yf, stats
