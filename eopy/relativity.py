"""First-order relativistic corrections for Earth satellite motion."""

from __future__ import annotations

import numpy as np

from .constants import EARTH_MU


SPEED_OF_LIGHT_MPS = 299_792_458.0


def schwarzschild_acceleration(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    mu: float = EARTH_MU,
) -> np.ndarray:
    """Return the first post-Newtonian Schwarzschild acceleration."""

    position = np.asarray(position_m, dtype=float)
    velocity = np.asarray(velocity_mps, dtype=float)
    radius = np.linalg.norm(position, axis=-1)
    speed_squared = np.sum(velocity * velocity, axis=-1)
    radial_speed = np.sum(position * velocity, axis=-1)
    scale = mu / (SPEED_OF_LIGHT_MPS**2 * radius**3)
    return scale[..., None] * (
        (4.0 * mu / radius - speed_squared)[..., None] * position
        + 4.0 * radial_speed[..., None] * velocity
    )
