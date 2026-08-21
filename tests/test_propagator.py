import numpy as np
import pymsis
from spacecoords import frames

from eopy.atmosphere import AtmosphereConfig, SolarActivity
import eopy.atmosphere as atmosphere_module
from eopy.constants import ASTRONOMICAL_UNIT, EARTH_MU, SOLAR_RADIATION_PRESSURE_1AU
from eopy.gravity import GravityField, gravity_acceleration
from eopy.integrators import Dop853Options
from eopy.propagator import SpacecraftProperties, drag_acceleration, propagate
from eopy.radiation import (
    RadiationPressure,
    cylindrical_shadow_factor,
    solar_radiation_pressure_acceleration,
)


def test_gravity_accepts_batch_positions():
    r = np.array([[7000e3, 0.0, 0.0], [0.0, 7000e3, 0.0]])
    acc = gravity_acceleration(r, GravityField(degree=2))
    assert acc.shape == (2, 3)
    assert acc[0, 0] < 0.0
    assert acc[1, 1] < 0.0


def test_propagate_two_particles_keeps_batch_shape():
    radius = 7000e3
    speed = np.sqrt(EARTH_MU / radius)
    states = np.array(
        [
            [radius, 0.0, 0.0, 0.0, speed, 0.0],
            [radius + 1e3, 0.0, 0.0, 0.0, np.sqrt(EARTH_MU / (radius + 1e3)), 0.0],
        ]
    )

    _t, out, stats = propagate(
        states,
        0.0,
        60.0,
        gravity=GravityField(degree=2),
        options=Dop853Options(rtol=1e-10, atol=1e-3, max_step=30.0),
    )

    assert out.shape == states.shape
    assert stats.accepted_steps > 0
    assert np.all(np.isfinite(out))


def test_pymsis_density_is_positive_for_leo_point():
    position = np.array([6778e3, 0.0, 0.0])
    cfg = AtmosphereConfig(model="msis")
    _t, out, _stats = propagate(
        np.array([*position, 0.0, np.sqrt(EARTH_MU / np.linalg.norm(position)), 0.0]),
        0.0,
        1.0,
        spacecraft=SpacecraftProperties(ballistic_coefficient_m2_per_kg=0.01),
        atmosphere=cfg,
        activity=SolarActivity(f10=160.0, f10_mean=155.0, ap=8.0),
        options=Dop853Options(rtol=1e-8, atol=1e-3, max_step=1.0),
    )
    assert np.all(np.isfinite(out))


def test_solar_radiation_pressure_cylindrical_umbra():
    sun = np.array([ASTRONOMICAL_UNIT, 0.0, 0.0])
    lit = np.array([[0.0, 7200e3, 0.0]])
    eclipsed = np.array([[-7200e3, 0.0, 0.0]])

    assert cylindrical_shadow_factor(lit, sun)[0] == 1.0
    assert cylindrical_shadow_factor(eclipsed, sun)[0] == 0.0

    radiation = RadiationPressure(
        area_m2=2.5, reflection_coefficient=0.7, mass_kg=1000.0
    )
    np.testing.assert_allclose(
        solar_radiation_pressure_acceleration(eclipsed, sun, radiation), 0.0
    )
