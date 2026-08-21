"""Event helper callables for propagation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .constants import EARTH_RADIUS

Event = Callable[[float, np.ndarray], float]


def _single_state(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=float)
    if arr.shape == (6,):
        return arr
    if arr.shape == (1, 6):
        return arr[0]
    raise ValueError("built-in events support a single propagated state")


def altitude_crossing_event(
    altitude_m: float,
    *,
    earth_radius_m: float = EARTH_RADIUS,
    terminal: bool = True,
    direction: float = 0.0,
) -> Event:
    """Return an event that triggers when geocentric altitude crosses a level."""

    def event(_t: float, state: np.ndarray) -> float:
        y = _single_state(state)
        return float(np.linalg.norm(y[:3]) - earth_radius_m - altitude_m)

    event.terminal = terminal
    event.direction = direction
    return event


def cylindrical_eclipse_event(
    sun_position: Callable[[float], np.ndarray] | np.ndarray,
    *,
    earth_radius_m: float = EARTH_RADIUS,
    terminal: bool = False,
    direction: float = 0.0,
) -> Event:
    """Return an event with zero at the cylindrical umbra boundary."""

    def event(t: float, state: np.ndarray) -> float:
        y = _single_state(state)
        sun = sun_position(t) if callable(sun_position) else sun_position
        sun_arr = np.asarray(sun, dtype=float)
        sun_hat = sun_arr / np.linalg.norm(sun_arr)
        along_sun = float(np.dot(y[:3], sun_hat))
        perpendicular = y[:3] - along_sun * sun_hat
        boundary_distance = float(np.linalg.norm(perpendicular) - earth_radius_m)
        if along_sun >= 0.0:
            return abs(along_sun) + max(boundary_distance, 0.0)
        return boundary_distance

    event.terminal = terminal
    event.direction = direction
    return event
